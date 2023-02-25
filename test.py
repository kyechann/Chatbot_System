import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import random
import pandas as pd
import glob
import re
import sentencepiece as spm
from torch.utils.data import Dataset
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from Utils.preprocess import *
from Data.dataset import *
from Model.model_transformer import *

d_model = 512
heads = 8
num_layers = 6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 10

train_loader = torch.utils.data.DataLoader(Dataset(),
                                           batch_size = 100, 
                                           shuffle=True, 
                                           pin_memory=True)

def load_split(file_path):
    data = pd.read_csv(file_path, encoding = 'utf8')
    print(len(data))

    Q, A = preprocess(data)
    return Q, A

def test_vocab():
    corpus = "total.txt"
    prefix = "chatbot"
    vocab_size = 16000

    spm.SentencePieceTrainer.train(
        f"--input={corpus} --model_prefix={prefix} --vocab_size={vocab_size + 7}" + 
        " --model_type=bpe" +
        " --max_sentence_length=100000" + # 문장 최대 길이
        " --pad_id=0 --pad_piece=[PAD]" + # pad (0)
        " --unk_id=1 --unk_piece=[UNK]" + # unknown (1)
        " --bos_id=2 --bos_piece=[BOS]" + # begin of sequence (2)
        " --eos_id=3 --eos_piece=[EOS]" + # end of sequence (3)
        " --user_defined_symbols=[SEP],[CLS],[MASK]") # 사용자 정의 토큰


    vocab_file = "chatbot.model"
    vocab = spm.SentencePieceProcessor()
    vocab.load(vocab_file)
    line = "안녕하세요. 만나서 반갑습니다. 저는 홍길동입니다."
    pieces = vocab.encode_as_pieces(line)
    ids = vocab.encode_as_ids(line)

    return line, pieces, ids

def train(train_loader, transformer, criterion, epoch):
    
    transformer.train()
    sum_loss = 0
    count = 0

    for i, (question, reply) in enumerate(train_loader):
        
        samples = question.shape[0]

        # Move to device
        question = question.to(device)
        reply = reply.to(device)

        # Prepare Target Data
        reply_input = reply[:, :-1]
        reply_target = reply[:, 1:]

        # Create mask and add dimensions
        question_mask, reply_input_mask, reply_target_mask = create_masks(question, reply_input, reply_target)

        # Get the transformer outputs
        out = transformer(question, question_mask, reply_input, reply_input_mask)

        # Compute the loss
        loss = criterion(out, reply_target, reply_target_mask)
        
        # Backprop
        transformer_optimizer.optimizer.zero_grad()
        loss.backward()
        transformer_optimizer.step()
        
        sum_loss += loss.item() * samples
        count += samples
        
        if i % 100 == 0:
            print("Epoch [{}][{}/{}]\tLoss: {:.3f}".format(epoch, i, len(train_loader), sum_loss/count))



if __name__ == '__main__':
    d_model = 512
    heads = 8
    num_layers = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 10
    Q, A = load_split('Data/gamsung.csv')
    line, pieces, ids = test_vocab()
    lr = 1e-4
    with open('Data/gamsung.json', 'r') as j:
        word_map = json.load(j)
    model = Transformer(d_model = d_model, heads = heads, num_layers = num_layers, word_map = word_map)
    model = model.to(device)
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    transformer_optimizer = AdamWarmup(model_size = d_model, warmup_steps = 4000, optimizer = adam_optimizer)
    criterion = LossWithLS(len(word_map), 0.1)
    criterion_reduction_none = nn.MSELoss(reduction='none')
    

    for epoch in range(epochs):
        
        train(train_loader, model, criterion_reduction_none, epoch)
        
        state = {'epoch': epoch, 'transformer': model, 'transformer_optimizer': transformer_optimizer}
        torch.save(state, 'checkpoint_' + str(epoch) + '.pth.tar')