import pandas as pd
import numpy as np
import os
import sentencepiece as spm
import re

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
