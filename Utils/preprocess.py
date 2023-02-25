import pandas as pd
import numpy as np
import os
import sentencepiece as spm
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.utils.data

def preprocess(data):
    
    q = []
    for i in data['Q']:
        i = re.sub(r"([?.!,])", r" \1 ", i)
        i = i.strip()
        q.append(i)
    
    a = []
    for j in data['A']:
        j = re.sub(r"([?.!,])", r" \1 ", j)
        j = j.strip()
        a.append(j)

    with open('total.txt', 'w', encoding='utf8') as f:
        f.write('\n'.join(q))
        f.write('\n'.join(a))
        
    return q, a


def create_masks(question, reply_input, reply_target):
    
    def subsequent_mask(size):
        mask = torch.triu(torch.ones(size, size)).transpose(0, 1).type(dtype=torch.uint8)
        return mask.unsqueeze(0)
    
    question_mask = (question!=0).to(device)
    question_mask = question_mask.unsqueeze(1).unsqueeze(1)         # (batch_size, 1, 1, max_words)
     
    reply_input_mask = reply_input!=0
    reply_input_mask = reply_input_mask.unsqueeze(1)  # (batch_size, 1, max_words)
    reply_input_mask = reply_input_mask & subsequent_mask(reply_input.size(-1)).type_as(reply_input_mask.data) 
    reply_input_mask = reply_input_mask.unsqueeze(1) # (batch_size, 1, max_words, max_words)
    reply_target_mask = reply_target!=0              # (batch_size, max_words)
    
    return question_mask, reply_input_mask, reply_target_mask