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
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings(action='ignore')

def data_load(file_path):
    data = pd.read_csv(file_path, encoding='utf-8')
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

