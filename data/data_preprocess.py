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
tags = [
        'JKS', 'JKC', 'JKG', 'JKO', 'JKB', 'JKV', 'JKQ',
        # 주격조사, 보격조사, 관형격조사, 목적격조사, 부사격조사, 호격조사, 인용격조사
        'JX', 'JC',
        # 보조사, 접속조사
        'SF', 'SP', 'SS', 'SE', 'SO',
        # 마침표,물음표,느낌표(SF), 쉼표,가운뎃점,콜론,빗금(SP), 따옴표,괄호표,줄표(SS), 줄임표(SE), 붙임표(물결,숨김,빠짐)(SO)
        'EP', 'EF', 'EC', 'ETN', 'ETM',
        # 선어말어미, 종결어미, 연결어미, 명사형전성어미, 관형형전성어미
        'XSN', 'XSV', 'XSA'
        # 명사파생접미사, 동사파생접미사, 형용사파생접미사
    ]

def data_load(file_path):
    data = pd.read_csv(file_path, encoding='utf-8')
    data = data.dropna(inplace=True)
    data = data.drop_duplicated(subset=['text'])
    data = re.sub(CHANGE_FILTER, "", data)
    
    return data

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


def pos(self, sentence):
    return okt.pos(sentence)

# 불용어 제거 후 필요한 품사 정보만 가져오기
def get_keywords(self, pos, without_tag=False):
    f = lambda x: x in self.exclusion_tags
    word_list = []
    for p in pos:
        if f(p[1]) is False:
            word_list.append(p if without_tag is False else p[0])
    return word_list

# 키워드를 단어 인덱스 시퀀스로 변환
def get_wordidx_sequence(self, keywords):
    if self.word_index is None:
        return []
    w2i = []
    for word in keywords:
        try:
            w2i.append(self.word_index[word])
        except KeyError:
            # 해당 단어가 사전에 없는 경우 OOV 처리
            w2i.append(self.word_index['OOV'])
    return w2i