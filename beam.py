import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from model import VQAModelWithKG
from argparse import Namespace

def generate_beam(model, tokenizer, beam_size: int = 5, generated=None, entry_length=65, temperature=1.0):
    """
    Generate text using beam search with length normalization.
    """
    model.eval()
    scores = None
    stop_token_id = tokenizer.eos_token_id
    device = next(model.parameters()).device
    is_stopped = torch.zeros(beam_size).bool().to(device) # whether beams have reached the stop token
    seq_len = torch.ones(beam_size).long().to(device) # length of each beam
    tokens = None
    
    with torch.no_grad():
        for i in range(entry_length):
            if is_stopped.all():
                break
            outputs = model.model(inputs_embeds=generated)
            logits = outputs.logits[:, -1, :] / temperature
            logits = torch.nn.functional.log_softmax(logits, dim=-1)

            if scores is None:
                scores, next_tokens = logits.topk(beam_size, -1)
                generated = generated.expand(beam_size, *generated.shape[1:])
                next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                if tokens is None:
                    tokens = next_tokens
                else:
                    tokens = tokens.expand(beam_size, *tokens.shape[1:])
                    tokens = torch.cat((tokens, next_tokens), dim=1)
            else:
                logits[is_stopped] = -float(np.inf)
                logits[is_stopped, 0] = 0
                scores_sum = scores[:, None] + logits
                seq_len[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_len[:, None]
                scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(
                    beam_size, -1
                )
                next_tokens_source = next_tokens // scores_sum.shape[1]
                seq_len = seq_len[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
                tokens = tokens[next_tokens_source]
                tokens = torch.cat((tokens, next_tokens), dim=1)
                generated = generated[next_tokens_source]
                scores = scores_sum_average * seq_len
                is_stopped = is_stopped[next_tokens_source]
            if model.gpttype == "biogpt":
                next_token_embed = model.model.biogpt.embed_tokens(
                    next_tokens.squeeze()
                ).view(generated.shape[0], 1, -1)
            elif model.gpttype == "gpt2":
                next_token_embed = model.model.transformer.wte(
                    next_tokens.squeeze()
                ).view(generated.shape[0], 1, -1)
            else:
                next_token_embed = model.model.get_input_embeddings()(tokens[:,-1])
                next_token_embed=next_token_embed.squeeze().view(generated.shape[0], 1, -1)
            generated = torch.cat((generated, next_token_embed), dim=1)
            is_stopped = is_stopped + next_tokens.eq(stop_token_id).squeeze()
            if is_stopped.all():
                break
            scores = scores / seq_len
            output_list = tokens.cpu().numpy()
            output_texts = [
                tokenizer.decode(output[: int(length)])
                for output, length in zip(output_list, seq_len)
            ]
            order = scores.argsort(descending=True)
            output_texts = [output_texts[i] for i in order]
            return output_texts

# if __name__ == "__main__":
#     args = Namespace(gpt_type='gpt2')
#     prefix_len = 2
#     clip_len = 2
#     prefix_size = 512
#     num_layers = 8
#     setting = 'Lora'
#     clip_mapping = 'transformer'
#     model = VQAModelWithKG(prefix_len, clip_len, prefix_size, num_layers, setting, clip_mapping, args)
#     tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
#     generated = tokenizer.encode("Hello, my name is")
#     generated = torch.tensor(generated).unsqueeze(0)
#     output_texts = generate_beam(model, tokenizer, generated=generated)
#     print(output_texts)
