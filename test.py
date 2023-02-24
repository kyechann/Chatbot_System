import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import random
import pandas as pd
import glob
import re
import sentencepiece as spm
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from Utils.preprocess import preprocess

def test_vocab(file_path):
    data = pd.read_csv(file_path, encoding = 'utf8')
    print(len(data))

    Q, A = preprocess(data)

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


if __name__ == '__main__':
    line, pieces, ids = test_sentencepiece('Data/gamsung.csv')
    print(line)
    print(pieces)
    print(ids)