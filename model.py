from peft import LoraConfig, LoraModel, get_peft_model, AutoPeftModelForCausalLM, PromptEncoderConfig, IA3Config
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from mapper import PrefixMapper
from argparse import Namespace
from mapper import PrefixMapper, MLP, KGMapperWithAttention, KGMapper, KGMapperMLP
from prefix_mapper import MLP, TransformerMapper

class VQAModelWithKG(nn.Module):
    def __init__(self, prefix_len, clip_len, prefix_size, kg_size, kg_len, num_layers, setting, clip_mapping, args = None):
        super(VQAModelWithKG, self).__init__()
        self.clip_len = clip_len
        self.prefix_len = prefix_len
        self.kg_len = kg_len
        gpt_type = args.model_type if args is not None else 'gpt2'
        self.gpttype = gpt_type
        self.device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(self.device)
        self.model = AutoModelForCausalLM.from_pretrained(gpt_type, load_in_8bit=True, device_map = self.device)
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt_type)
        self.gpt_embedding_size = self.model.transformer.wte.weight.shape[1]

        # different settings of adapter
        if setting == 'lora':
            lora_config = LoraConfig(r=8, lora_alpha=32, lora_dropout=0.1, task_type='CAUSAL_LM')
            self.model = get_peft_model(self.model, lora_config)   
            self.model.print_trainable_parameters()
        elif setting == 'IA3':
            ia3_config = IA3Config(ia3_type='IA3', task_type='CAUSAL_LM')
            self.model = get_peft_model(self.model, ia3_config)
            self.model.print_trainable_parameters()
        elif setting == 'p-tuning':
            p_config = PromptEncoderConfig(peft_type='P_TUNING', task_type='CAUSAL_LM', num_virtual_tokens=30)
            self.model = get_peft_model(self.model, p_config)
            self.model.print_trainable_parameters()
        elif setting == 'prefix_tuning':
            prefix_config = PromptEncoderConfig(peft_type='PREFIX_TUNING', task_type='CAUSAL_LM', num_virtual_tokens=30)
            self.model = get_peft_model(self.model, prefix_config)
            self.model.print_trainable_parameters()
        elif setting == 'prompt_tuning':
            prompt_config = PromptEncoderConfig(peft_type='PROMPT_TUNING', task_type='CAUSAL_LM', num_virtual_tokens=30)
            self.model = get_peft_model(self.model, prompt_config)
            self.model.print_trainable_parameters()
        elif setting == 'frozen':
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            raise ValueError('Invalid setting')

        
        if clip_mapping == 'transformer':
            self.clip_mapping = PrefixMapper(prefix_size, self.gpt_embedding_size, prefix_len, clip_len, num_layers).to(self.device)
        elif clip_mapping == 'mlp':
            dim_size = [prefix_size, (self.gpt_embedding_size * prefix_len) // 2,self.gpt_embedding_size * prefix_len, self.gpt_embedding_size * prefix_len]
            self.clip_mapping = MLP(dim_size).to(self.device)
        else:
            raise ValueError('Invalid clip_mapping')
        
        self.kg_mapping = KGMapper(kg_size, self.gpt_embedding_size, kg_len, num_layers)
        # self.kg_mapping = KGMapperMLP(kg_size, self.gpt_embedding_size, kg_len, hidden_dim=1024)
        
    def forward(self, prefix, kg_list, tokens, mask, q_len, batch_size):
        prefix = self.clip_mapping(prefix).view(-1, self.prefix_len, self.gpt_embedding_size)
        if kg_list is not None:
            if not torch.all(kg_list == 0):
                kg = self.kg_mapping(kg_list).view(-1, self.kg_len, self.gpt_embedding_size)
            else:
                kg = torch.zeros((batch_size, self.kg_len, self.gpt_embedding_size))
        
        if self.gpttype=='microsoft/biogpt':
            embedding = self.model.transformer.embed_tokens(tokens)
        else:
            embedding = self.model.transformer.wte(tokens)
        for b in range(batch_size):
            embedding[b, q_len[b]:q_len[b]+self.kg_len, :] = kg[b]
            embedding[b, q_len[b]+self.kg_len:q_len[b]+self.kg_len+self.prefix_len, :] = prefix[b]
        embedding = embedding.to(self.device)
        mask = mask.to(self.device)
        # print(f"embedding shape: {embedding.shape}, mask shape: {mask.shape}") 
        return self.model(inputs_embeds=embedding, attention_mask=mask)

    def generate(self, prefix, kg_list, tokens, mask, q_len):
        prefix = self.clip_mapping(prefix.view(1,-1)).view(-1, self.prefix_len, self.gpt_embedding_size)
        if kg_list is not None:
            kg_list.to(self.device)
            if not torch.all(kg_list == 0):
                kg = self.kg_mapping(kg_list.view(1,-1)).view(-1, self.kg_len, self.gpt_embedding_size)
            else:
                kg = torch.zeros((1, self.kg_len, self.gpt_embedding_size))
            # kg = self.kg_mapping(kg_list.view(1,-1)).view(-1, self.kg_len, self.gpt_embedding_size)
        if self.type=='microsoft/biogpt':
            embedding_txt = self.model.transformer.embed_tokens(tokens)
        else:
            embedding_txt = self.model.transformer.wte(tokens)
        #question: ..., knowledge: ..., prefix: ..., answer: ..., <eos>
        embedding_txt[q_len:q_len+self.kg_len, :] = kg
        embedding_txt[q_len+self.kg_len:q_len+self.kg_len+self.prefix_len, :] = prefix
        return embedding_txt

class MedVQAModelNoKG(nn.Module):
    def __init__(
        self,
        prefix_length=2,
        clip_length=2,
        prefix_size=512,
        num_layers=8,
        setting="frozen",
        mapping_type="MLP",
        args=None,
    ):
        super(MedVQAModelNoKG, self).__init__()
        gpttype = "gpt2-xl"
        self.model_type = gpttype
        self.setting = setting
        self.prefix_length = prefix_length
        self.gpt = GPT2LMHeadModel.from_pretrained(gpttype,load_in_8bit=True,device_map='auto')
        if setting == 'lora':
            lora_config = LoraConfig(r=8, lora_alpha=32, lora_dropout=0.1, task_type='CAUSAL_LM')
            self.model = get_peft_model(self.model, lora_config)   
            self.model.print_trainable_parameters()
        elif setting == 'IA3':
            ia3_config = IA3Config(ia3_type='IA3', task_type='CAUSAL_LM')
            self.model = get_peft_model(self.model, ia3_config)
            self.model.print_trainable_parameters()
        elif setting == 'p-tuning':
            p_config = PromptEncoderConfig(peft_type='P_TUNING', task_type='CAUSAL_LM', num_virtual_tokens=30)
            self.model = get_peft_model(self.model, p_config)
            self.model.print_trainable_parameters()
        elif setting == 'prefix_tuning':
            prefix_config = PromptEncoderConfig(peft_type='PREFIX_TUNING', task_type='CAUSAL_LM', num_virtual_tokens=30)
            self.model = get_peft_model(self.model, prefix_config)
            self.model.print_trainable_parameters()
        elif setting == 'prompt_tuning':
            prompt_config = PromptEncoderConfig(peft_type='PROMPT_TUNING', task_type='CAUSAL_LM', num_virtual_tokens=30)
            self.model = get_peft_model(self.model, prompt_config)
            self.model.print_trainable_parameters()
        elif setting == 'frozen':
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            raise ValueError('Invalid setting')
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpttype)
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        # for the replace_visual ablation study we replace the visual tokens with learnable parameters 
        self.nv_tokens = torch.nn.Parameter(torch.randn(args.batch_size,prefix_length,self.gpt_embedding_size),requires_grad=True).cuda()
        if mapping_type == "MLP":
            self.clip_project = MLP((prefix_size,
                    (self.gpt_embedding_size * prefix_length) // 2,
                    self.gpt_embedding_size * prefix_length,
                    self.gpt_embedding_size * prefix_length))
        elif mapping_type == "Transformer":
            self.clip_project = TransformerMapper(
                prefix_size,
                self.gpt_embedding_size,
                prefix_length,
                clip_length,
                num_layers)
        else:
            raise ValueError("select valid mapping type: MLP or Transformer")
if __name__ == "__main__":
    gpt_type = 'gpt2-xl'
    model = AutoModelForCausalLM.from_pretrained(gpt_type, load_in_8bit=True, device_map = 'auto')
    print(model)