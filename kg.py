from pydantic import BaseModel, Field
from typing import List
from transformers import AutoTokenizer, AutoModel, AutoModelForTokenClassification, pipeline
import torch
import json
import networkx as nx
import matplotlib.pyplot as plt
from flair.data import Sentence
from flair.models import SequenceTagger
from obj_recog import MedicalObjDetection
from ques_classifier import VisionKnowledgeClassifier
import os
from obj_recog import MedicalObjDetection


class Entity(BaseModel):
    name: str = Field(..., description="Name of the entity")


class EntityResponse(BaseModel):
    entities: List[Entity] = Field(..., description="List of entities")

class KnowledgeGraph():
    def __init__(self, kg_path, kg_name_list =['en_disease', 'en_organ_rel', 'en_organ']):
        self.kg_path = kg_path
        self.kg_name_list = kg_name_list
        self.pipeline = pipeline("token-classification", model="Clinical-AI-Apollo/Medical-NER", aggregation_strategy='simple')
        self.model = AutoModel.from_pretrained("neuml/pubmedbert-base-embeddings")
        self.tokenizer = AutoTokenizer.from_pretrained("neuml/pubmedbert-base-embeddings")
        # self.tagger = SequenceTagger.load('flair/pos-english')
        self.embedding_cache = {}
        self.obj_recog = MedicalObjDetection()
        self.classifier = VisionKnowledgeClassifier()
        self.kg_size = 768


    def extract_entities(self, sentence, score_threshold=0.2):
        # Ask the LLM to identify entities in the sentence
        entities = self.pipeline(sentence)
        print(f"entities: {entities}")
        filtered_entities = [entity for entity in entities if entity['score'] > score_threshold]
        result = []
        for entity in filtered_entities:
            result.append(Entity(name=entity['word']))         
        return EntityResponse(entities=result)
    
    def meanpooling(self, output, mask):
        embeddings = output[0]
        mask = mask.unsqueeze(-1).expand(embeddings.size()).float()
        return torch.sum(embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
    
    def encode(self, text):
        if isinstance(text, list):
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=768)
            with torch.no_grad():
                outputs = self.model(**inputs)
            embeddings = self.meanpooling(outputs, inputs['attention_mask'])
            return embeddings
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        else:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
            embeddings = self.meanpooling(outputs, inputs['attention_mask'])
            self.embedding_cache[text] = embeddings
            return embeddings
    
    def precompute_node_embeddings(self, graph):
        node_embeddings = {}
        for node in graph.nodes:
            node_embeddings[node] = self.encode(node)
        return node_embeddings

    
    def compute_similarity(self, entity_embedding, node_embedding):
        similarity_tensor = torch.cosine_similarity(entity_embedding, node_embedding, dim=1)
        return similarity_tensor.item()
    
    def get_top_k_similar_nodes(self, entity, graph, k=2):
        entity_embedding = self.encode(entity)
        node_embeddings = self.precompute_node_embeddings(graph)
        similarities = {}
        for node, node_embedding in node_embeddings.items():
            similarity = self.compute_similarity(entity_embedding, node_embedding)
            similarities[node] = similarity
        return sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:k]


    
    def is_exact_match(self, entity1, entity2):
        return entity1.lower() == entity2.lower()
    
    def get_relationships(self, graph, entity_names, attribute_names):
        relationships = []
        for entity in entity_names:
            similar_entities = self.get_top_k_similar_nodes(entity, graph)

            # For each semantically similar entity, retrieve its relationships
            for similar_entity in similar_entities:
                for _, target, data in graph.out_edges(similar_entity, data=True):
                    relation = data.get('attribute')
                    if relation and relation in attribute_names:
                        relationships.append((similar_entity[0], relation, target))
                    elif relation:
                        relationships.append((similar_entity[0], relation, target))

                for source, _, data in graph.in_edges(similar_entity, data=True):
                    relation = data.get('attribute')
                    if relation and relation in attribute_names:
                        relationships.append((source, relation, similar_entity[0]))
                    elif relation:
                        relationships.append((source, relation, similar_entity[0]))


        return relationships
    
    def parse_json_to_kg(self, json_file):
        with open(json_file, 'r') as f:
            knowledge_graph = json.load(f)
        graph = nx.DiGraph()
        for entity, attributes in knowledge_graph.items():
            if not graph.has_node(entity):
                graph.add_node(entity)
            
            for attribute, values in attributes.items():
                for value in values:
                    if isinstance(value, str):
                        graph.add_node(value)
                        graph.add_edge(entity, value, attribute=attribute)
                    elif isinstance(value, list):
                        for v in value:
                            graph.add_node(v)
                            graph.add_edge(entity, v, attribute=attribute)

        return graph
    
    def enhanced_answer_query(self, llm, knowledge_graph, query):
        entities = self.extract_entities(query)
        all_relations = {}
        for entity in entities:
            relationships = self.get_relationships(knowledge_graph, entity)
            if relationships:
                all_relations[entity] = relationships

        compiled_infor = []
        for entity, relations in all_relations.items():
            compiled_infor.append(f"{relations[0]} {relations[1]} {relations[2]}")

        info_string = '. '.join(compiled_infor)

        if info_string:
            prompt = f"Based on this: {info_string}. Give a response to this query: {query}"
            response = llm.generate_text(prompt)[0]
        else:
            response = "I couldn't find a direct relationship for the entities in your query based on the provided data."

        return response
    
    def extract_kg_keywords(self, kg_path, kg_name): #extract all keywords from a kg
        with open(f'{kg_path}/{kg_name}.json', 'r') as f:
            kg = json.load(f)
        keywords = []
        for _, attributes in kg.items():
            for attribute, _ in attributes.items():
                if attribute not in keywords:
                    keywords.append(attribute)
        return keywords
    
    def select_kg(self, question, ques_verb):
        kg_keywords = {}
        for kg_name in self.kg_name_list:
            kg_keywords[kg_name] = self.extract_kg_keywords(self.kg_path, kg_name)

        question = Sentence(question)
        # tagger = SequenceTagger.load('flair/pos-english')
        # tagger.predict(question)
        # ques_keywords = [token.text for token in question if any(tag.value.startswith('VB') for tag in token.get_labels('pos'))]
        # auxiliary_verbs = {"Do", "Does", "Did", "do", "does", "did"}
        # ques_keywords = [word for word in ques_keywords if word not in auxiliary_verbs]
        print(f"ques_keywords: {ques_verb}")
        keyword_embeddings = self.encode(ques_verb)
        highest_similarity = 0
        best_kg = None
        for kg_name, keywords in kg_keywords.items():
            kg_embeddings = self.encode(keywords)
            similarities = torch.cosine_similarity(keyword_embeddings.unsqueeze(1), kg_embeddings.unsqueeze(0), dim=2)
            max_similarity = torch.max(similarities).item()
            if max_similarity > highest_similarity:
                highest_similarity = max_similarity
                best_kg = kg_name

        return best_kg, ques_verb, kg_keywords[best_kg]

    def get_attribute_name(self, question_verb, kg_attributes):
        max_similarity = 0
        for attribute in kg_attributes:
            attribute_embd = self.encode(attribute)
            verb_embd = self.encode(question_verb)
            similarity = torch.cosine_similarity(attribute_embd, verb_embd)
            if similarity > max_similarity:
                max_similarity = similarity
                best_attribute = attribute
        return best_attribute

    def obtain_kg(self, question, img_prefix, q_verb):
        attribute_names = []
        ranked_relationships = []
        best_kg, ques_verbs, kg_keyword = self.select_kg(question, q_verb)
        for verb in ques_verbs:
            attribute_name = self.get_attribute_name(verb, kg_keyword)
            print(f"attribute_name: {attribute_name}")
            attribute_names.append(attribute_name)
        kg_full_path = os.path.join(self.kg_path, best_kg + '.json')
        knowledge_graph = self.parse_json_to_kg(kg_full_path)
        entity_names = self.obj_recog.predict(img_prefix)
        print(f"entity_names: {entity_names}")
        if entity_names:
            relationships = self.get_relationships(knowledge_graph, entity_names, attribute_names)
            # rank the relationships
            for relation in relationships:
                relation_str = ' '.join(relation)
                relation_embd = self.encode(relation_str)
                similarity = torch.cosine_similarity(relation_embd, self.encode(question))
                ranked_relationships.append((relation, similarity))
            ranked_relationships = sorted(ranked_relationships, key=lambda x: x[1], reverse=True)
            print(f"relationships: {relationships}")
            print(f"ranked_relationships: {ranked_relationships}")
            kg_list = []
            for relation in relationships:
                kg_str = ' '.join(relation)
                kg = self.encode(kg_str).squeeze()
                kg_list.append(kg)
            kg_list = torch.cat(kg_list)
            print(f"kg_list: {kg_list.size()}")
        else: # kg_list is empty
            kg_list = torch.zeros(self.kg_size)
        return kg_list

    
    

if __name__ == '__main__':
    kg = KnowledgeGraph('datasets/slake/KG')
    question = "Does the picture contain the organ which has the effect of sensing light?"
    print(kg.select_kg(question))
    # data_path = "datasets/slake/try.json" #datasets/slake/try.json
    # with open(data_path, 'r') as f:
    #     data = json.load(f)
    # question = []
    # img_prefix = []
    # img_ids = []
    # for d in data:
    #     question.extend(d['questions'])
    #     img_prefix.extend([d['img_prefix']]*len(d['questions']))
    #     img_ids.extend([d['img_id']]*len(d['questions']))
    # print(question[1])
    # print(kg.classifier.classify(question[1]))
    # print(kg.select_kg(question[1]))
    # print(kg.obtain_kg(question[1], img_prefix[0]))
    # entity_response = kg.extract_entities(question[1])
    # if "image" or "picture" in question[1]:
    #     img_prefix = torch.tensor(img_prefix[1])
    #     obj_in_img = obj_recog.predict(img_prefix)
    #     for obj in obj_in_img:
    #         entity_response.entities.append(Entity(name = obj))
    # entity_names = [entity.name for entity in entity_response.entities]
    # print(entity_names)
