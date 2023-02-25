import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import random
import pandas as pd
import glob
import re
import json
import sentencepiece as spm
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from Utils.preprocess import preprocess
from torch.utils.data import Dataset, DataLoader

# 최대 길이를 40으로 정의
MAX_LENGTH = 50

START_TOKEN = [2]
END_TOKEN = [3]

# 토큰화 / 정수 인코딩 / 시작 토큰과 종료 토큰 추가 / 패딩
def tokenizeer(inputs, outputs):
    inputs, outputs = [], []

    for (sentence1, sentence2) in zip(inputs, outputs):
    # encode(토큰화 + 정수 인코딩), 시작 토큰과 종료 토큰 추가
        zeros1 = np.zeros(MAX_LENGTH, dtype=int)
        zeros2 = np.zeros(MAX_LENGTH, dtype=int)
        sentence1 = START_TOKEN + vocab.encode_as_ids(sentence1) + END_TOKEN
        zeros1[:len(sentence1)] = sentence1[:MAX_LENGTH]

        sentence2 = START_TOKEN + vocab.encode_as_ids(sentence2) + END_TOKEN
        zeros2[:len(sentence2)] = sentence2[:MAX_LENGTH]

        inputs.append(zeros1)
        outputs.append(zeros2)
    return inputs, outputs


class SequenceDataset(Dataset):
    def __init__(self, questions, answers):
        questions = np.array(questions)
        answers = np.array(answers)
        self.inputs = questions
        self.dec_inputs = answers[:,:-1]
        self.outputs = answers[:,1:]
        self.length = len(questions)
    
    def __getitem__(self,idx):
        return (self.inputs[idx], self.dec_inputs[idx], self.outputs[idx])

    def __len__(self):
        return self.length
    


class Dataset(Dataset):

    def __init__(self):

        self.pairs = pd.read_csv('Data/gamsung.csv', encoding='utf-8')
        self.dataset_size = len(self.pairs)

    def __getitem__(self, i):
        
        question = torch.LongTensor(self.pairs[i][0])
        reply = torch.LongTensor(self.pairs[i][1])
            
        return question, reply

    def __len__(self):
        return self.dataset_size
    