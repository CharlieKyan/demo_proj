from tqdm import tqdm
import sys
import torch
import torch.nn as nn
from transformers import set_seed, GPT2Config, GPT2Tokenizer
from transformers import AutoTokenizer, AutoModel, pipeline
from transformers.models.biogpt import BioGptTokenizer
import os
import pandas as pd
from torch.utils.data import Dataset
import json
import numpy as np
import pdb
from kg import KnowledgeGraph, Entity, EntityResponse
from ques_classifier import VisionKnowledgeClassifier
from obj_recog import MedicalObjDetection

class SlakeDataset(Dataset):
    def __init__ (self, data_path, kg_path, kg_name_list =['en_disease', 'en_organ_rel', 'en_organ'],  split = 'train', like_test = False, model_type = 'gpt2', prefix_length = 8, kg_len = 16):
        super().__init__()
        self.split = split
        self.data_path = data_path
        self.like_test = like_test
        complete_data_path = os.path.join(data_path, split + '_preprocessed.json')
        with open(complete_data_path, 'rb') as f:
            self.data = json.load(f)

        # for debugging only, remove later
        # self.data = self.data[:20]

        self.questions = []
        self.answers = []
        self.img_prefixes = []
        self.img_ids = []
        self.kg_list = []
        self.triplet = []
        for d in self.data:
            self.questions.extend(d['questions'])
            self.answers.extend(d['answers'])
            self.img_prefixes.extend([d['img_prefix']] * len(d['questions']))
            self.img_ids.extend([d['img_id']] * len(d['questions']))
            self.triplet.extend(d['triple'])

        self.model_type = model_type
        self.tokenizer = AutoTokenizer.from_pretrained(model_type)
        self.train_setting = True if (split!='test'and like_test==False) else False
        self.prefix_len = prefix_length
        self.kg_len = kg_len
        self.kg_path = kg_path
        self.kg = KnowledgeGraph(kg_path, kg_name_list)
        self.classifier = VisionKnowledgeClassifier()
        self.med_obj_recog = MedicalObjDetection()
        self.max_q_len, self.max_a_len = self.get_max_sq_len(self.questions, self.answers)

        self.kg_len_dist = []
        count = 0

        for question in self.questions:
            if self.classifier.classify(question):
                index = self.questions.index(question)
                kg = self.obtain_kg(index)
                self.kg_list.append(kg)
                self.kg_len_dist.append(kg.size(0))
                count += 1
            else:
                self.kg_list.append(torch.zeros(self.kg_len))
            
        print(f"Number of questions with KG: {count}")
            

    
    def get_max_sq_len(self, questions, answers):
        q_lens = [len(self.tokenizer.tokenize(q)) for q in questions]
        a_lens = [len(self.tokenizer.tokenize(a)) for a in answers]
        max_q_len = int(np.mean(q_lens) + 2*np.std(q_lens))
        max_a_len = int(np.mean(a_lens) + 2*np.std(a_lens))
        return max_q_len, max_a_len
    
    def make_padding(self, max_len, tokens, question = False, leftover_tokens = 0):
        padding = max_len - tokens.size(0) 
        if padding > 0:
            if question:
                leftover_tokens = padding
                mask = torch.ones(tokens.size(0))
            else:
                tokens = torch.cat((tokens, torch.zeros(padding+leftover_tokens)))
                mask = torch.zeros(max_len+leftover_tokens)    
              
        elif padding==0:
            if question:
                mask = torch.ones(tokens.size(0)) 
            else:
                mask = torch.zeros(tokens.size(0)+leftover_tokens)
                tokens = torch.cat((tokens,torch.zeros(leftover_tokens)))
                
        
        elif padding < 0:
            if question:
                tokens = tokens[:max_len]
                mask = torch.ones(max_len)
            else:
                tokens = torch.cat((tokens[:max_len], torch.zeros(leftover_tokens)))
                mask = torch.zeros(max_len+ leftover_tokens)
        return tokens, mask, leftover_tokens

    def make_padding_test_setting(self, max_len, tokens,do_padding=False):
        padding = max_len - tokens.size(0)
        padding_len = 0
        if padding > 0:
            if do_padding:
                mask = torch.cat((torch.ones(tokens.size(0)),torch.zeros(padding)))
                tokens = torch.cat((tokens,torch.zeros(padding)))
                padding_len = padding
            else:
                mask = torch.ones(tokens.size(0))
        elif padding ==0:
            mask = torch.ones(max_len)
        elif padding < 0:
            tokens = tokens[:max_len]
            mask = torch.ones(max_len)
        return tokens, mask, padding_len
    

    def obtain_kg(self, index):
        question = self.questions[index]
        q_verb = self.triplet[index][1]
        attribute_names = []
        best_kg, ques_verbs, kg_keyword = self.kg.select_kg(question, q_verb)
        for verb in ques_verbs:
            attribute_name = self.kg.get_attribute_name(verb, kg_keyword)
            attribute_names.append(attribute_name)
        kg_full_path = os.path.join(self.kg_path, best_kg + '.json')
        knowledge_graph = self.kg.parse_json_to_kg(kg_full_path)
        # entity_response = self.kg.extract_entities(self.questions[index])
        # if "image" or "picture" in self.questions[index]:
        #     img_prefix = torch.tensor(self.img_prefixes[index])
        #     obj_in_img = self.med_obj_recog.predict(img_prefix)
        #     for obj in obj_in_img:
        #         entity_response.entities.append(Entity(name = obj))
        img_prefix = torch.tensor(self.img_prefixes[index])
        obj_in_img = self.med_obj_recog.predict(img_prefix)
        # entity_names = [entity.name for entity in entity_response.entities]
        if len(obj_in_img) > 0:
            relationships = self.kg.get_relationships(knowledge_graph, obj_in_img, attribute_names)
            kg_list = []
            for relation in relationships:
                kg_str = ' '.join(relation)
                kg = self.kg.encode(kg_str).squeeze()
                kg_list.append(kg)
            kg_list = torch.cat(kg_list)
            print(f"kg_list: {kg_list.size()}")
        else: # kg_list is empty
            kg_list = torch.zeros(self.kg_len)
        return kg_list
    
    def pad_kg(self, kg, max_kg_len=12288):
        padding = max_kg_len - kg.size(0)
        if padding > 0:
            kg = torch.cat((kg, torch.zeros(padding)))
        elif padding < 0:
            kg = kg[:max_kg_len]
        return kg

    
    def pad_sequences(self,index):
        m = [torch.tensor(self.tokenizer.encode('question:')), torch.tensor(self.tokenizer.encode(' knowledge: ')), torch.tensor(self.tokenizer.encode(' context: ')), torch.tensor(self.tokenizer.encode(' answer: ')), torch.tensor(self.tokenizer.encode(self.tokenizer.eos_token))] #m[0] is question, m[1] is knowledge, m[2] is context, m[3] is answer, m[4] is end of text token
        m_mask = [torch.ones(len(m[0])), torch.ones(len(m[1])), torch.ones(len(m[2])), torch.ones(len(m[3])), torch.zeros(len(self.tokenizer.encode(self.tokenizer.eos_token)))]

        if self.train_setting:
            q = torch.tensor(self.tokenizer.encode(self.questions[index]))
            a = torch.tensor(self.tokenizer.encode(self.answers[index]))
                
            max_q_len, max_a_len = self.get_max_sq_len(self.questions, self.answers)
            q, q_mask, leftover_tokens = self.make_padding(max_q_len, q, question = True)
            q_len = m[0].size(0) + q.size(0) + m[1].size(0)
            a, a_mask, _ = self.make_padding(max_a_len, a, leftover_tokens = leftover_tokens)
            if len((a==0).nonzero())!=0:
                pad_start = (a==0).nonzero()[0]
            else:
                pad_start=[]
            a = torch.cat((a, m[4])) if len(pad_start)==0 else torch.cat((a[:pad_start],m[4],a[pad_start:]))
            q = torch.cat((m[0],q,m[1],torch.ones(self.kg_len), m[2], torch.ones(self.prefix_len),m[3],a))
            if self.classifier.classify(self.questions[index]):
                q_mask = torch.cat((m_mask[0],q_mask,m_mask[1],torch.ones(self.kg_len),m_mask[2],torch.ones(self.prefix_len),m_mask[3],a_mask,m_mask[4]))
            else:
                q_mask = torch.cat((m_mask[0],q_mask,m_mask[1],torch.zeros(self.kg_len),m_mask[2],torch.ones(self.prefix_len),m_mask[3],a_mask,m_mask[4]))

            return q, q_mask, q_len
        else:
            # in the test stage we do not have acces to the answer, so we just load the question, knowledge and context
            q = torch.tensor(self.tokenizer.encode(self.questions[index]))
            max_q_len, _ = self.get_max_sq_len(self.questions, self.answers)
            q, q_mask, _ = self.make_padding_test_setting(max_q_len, q)
            q_len = m[0].size(0) + q.size(0) + m[1].size(0)
            q = torch.cat((m[0],q,m[1],torch.ones(self.kg_len), m[2], torch.ones(self.prefix_len),m[3]))
            if self.classifier.classify(self.questions[index]):
                q_mask = torch.cat((m_mask[0],q_mask,m_mask[1],torch.ones(self.kg_len),m_mask[2],torch.ones(self.prefix_len),m_mask[3]))
            else:
                q_mask = torch.cat((m_mask[0],q_mask,m_mask[1],torch.zeros(self.kg_len),m_mask[2],torch.ones(self.prefix_len),m_mask[3]))
            # q_mask = torch.cat((m_mask[0],q_mask,m_mask[1]),torch.ones(self.kg_len),m_mask[2],torch.ones(self.prefix_len),m_mask[3])

            return q, q_mask, q_len


    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, index):
        prefix = torch.tensor(self.img_prefixes[index])
        kg = self.pad_kg(self.kg_list[index])
        tokens, mask, seq_len = self.pad_sequences(index)
        return prefix, kg, tokens, mask, seq_len
    

if __name__ == "__main__":
    slake = SlakeDataset(data_path = 'datasets/slake', kg_path = 'datasets/slake/KG', split = 'train')
    print(slake[0][1].size())
