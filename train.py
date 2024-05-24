from sklearn.metrics import accuracy_score, roc_auc_score
import torch
from tqdm import tqdm
import copy
import os
import numpy as np
import time
import random
import torch.nn as nn
import torch.nn.functional as nnf
import os
import numpy as np
import random
import pandas as pd
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
)
from torch.nn import functional as nnf
from accelerate import Accelerator
import pdb

def traingpt_open_ended(train_loader, valid_loader, model, args):
    accelerator = Accelerator()
    device = accelerator.device
    if str(device) == "cuda":
        device = torch.device("cuda:5")
    model = model.to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.num_warmup_steps, num_training_steps=args.epochs)
    model, optimizer, train_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, scheduler)
    valid_loader = accelerator.prepare(valid_loader)

    best_valid_loss = float('inf')
    counter = 0
    n_epoch = args.epochs
    for epoch in range(n_epoch):
        with tqdm(total=args.batch_size * len(train_loader), desc=f'Train Epoch {epoch}', unit='img') as pbar:
            start_time = time.time()
            total_loss = 0

            for i, (prefix, kg, tokens, mask, q_len) in enumerate(train_loader):
                with accelerator.accumulate(model):
                    print("prefix", prefix.shape)
                    prefix = prefix.type(torch.float32).to(device)
                    kg = kg.type(torch.float32).to(device)
                    tokens = tokens.type(torch.long).to(device)
                    mask = mask.type(torch.long).to(device)
                    q_len = q_len.type(torch.long).to(device)
                    output = model(prefix, kg, tokens, mask, q_len, args.batch_size)
                    logits = output.logits
                    loss = 0
                    shift = 10 if args.setting=="p_tuning" or args.setting=="prompttuning" else 0 
                    # calculate loss
                    for b in range(logits.size(0)):
                        condensed_tokens = tokens[b,q_len[b]+model.kg_len+model.prefix_len+1:]
                        condensed_logits = logits[b,shift+q_len[b]+model.kg_len+model.prefix_len:-1]

                        loss+= nnf.cross_entropy(condensed_logits.reshape(-1,logits.shape[-1]), condensed_tokens.flatten(), ignore_index=0)
                    loss=loss/logits.size(0) 
                    accelerator.backward(loss)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    total_loss += loss.item()
                    avg_loss = total_loss / (i+1)   
                    pbar.set_postfix(**{'loss (batch)': {avg_loss} })
                    pbar.update(args.batch_size)

        model.eval()
        valid_loss = 0
        with tqdm(total=args.batch_size * len(valid_loader), desc=f'Val Epoch {epoch}', unit='img') as pbar:
            for i, (prefix, kg, tokens, mask, q_len) in enumerate(valid_loader):
                torch.cuda.empty_cache()
                prefix = prefix.type(torch.float32)
                kg = kg.type(torch.float32).to(device)
                tokens = tokens.type(torch.long)
                mask = mask.type(torch.long)
                q_len = q_len.type(torch.long)
                with torch.no_grad():
                    output = model(prefix, kg, tokens, mask, q_len, args.batch_size)
                    logits = output.logits
                    loss = 0
                    shift = 10 if args.setting=="p_tuning" or args.setting=="prompttuning" else 0 
                    # calculate loss
                    for b in range(logits.size(0)):
                        condensed_tokens = tokens[b,q_len[b]+model.prefix_len+1:]
                        condensed_logits = logits[b,shift+q_len[b]+model.prefix_len:-1]

                        loss+= nnf.cross_entropy(condensed_logits.reshape(-1,logits.shape[-1]), condensed_tokens.flatten(), ignore_index=0)
                loss=loss/logits.size(0)    
                valid_loss += loss.item()
                avg_val_loss = valid_loss / (i+1)
                pbar.set_postfix(**{'loss (batch)': avg_val_loss})
                pbar.update(args.batch_size)
            
            if avg_val_loss < best_valid_loss:
                best_valid_loss = avg_val_loss
                torch.save(model.state_dict(), os.path.join(args.out_dir, f"open_ended_latest.pt"))
            
            elapsed_time = time.time() - start_time
            print(f'Epoch {epoch} took {elapsed_time} seconds')
            print(
            "VAL epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t time={:.2f}s".format(
                epoch + 1, n_epoch, avg_loss, avg_val_loss, elapsed_time
                )   
            )
            if avg_val_loss > best_valid_loss:
                counter += 1
                if counter > args.patience:
                    print('Early stopping')
                    break
            
    return model

