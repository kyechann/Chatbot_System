import torch
import random
import pandas as pd
import glob
import re
import numpy as np
import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from konlpy.tags import Okt
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings(action='ignore')

FILTERS = "([~.,!?\"':;)(])"
CHANGE_FILTER = re.compile(FILTERS)
okt = Okt()

# 로드 후 전처리
def data_load(file_path):
    data = pd.read_csv(file_path, encoding='utf-8')
    # Null 값 제거
    data = data.dropna(inplace=True)
    # 중복 제거
    data = data.drop_duplicated(subset=['text'])
    # 특수 문자 제거
    data = re.sub(CHANGE_FILTER, "", data)
    
    return data

# 데이터 분할
def split_data(data):
    question, answer = list(data['Q']), list(data['A'])
    # skleran에서 지원하는 함수를 통해서 학습 셋과
    # 테스트 셋을 나눈다.
    train_input, eval_input, train_label, eval_label = train_test_split(question, answer, test_size=0.33,
                                                                        random_state=42)
    # 원본 데이터 모두를 학습시키기 위해서는 아래 두 줄의 주석을 해제한다.
    # train_input = question
    # train_label = answer
    # 그 값을 리턴한다.
    return train_input, train_label, eval_input, eval_label

# 데이터 토크나이저
def tokenizing_data(data):
    morph_analyzer = Okt()
    f = lambda x: x in tags
    # 형태소 토크나이즈 결과 문장을 받을 리스트를 생성.
    result_data = list()

    for seq in tqdm(data):
        seq = okt.pos(seq)
        if f(seq) in False:
            morphlized_seq = " ".join(morph_analyzer.morphs(seq.replace(' ', '')))
            result_data.append(morphlized_seq)

    return result_data