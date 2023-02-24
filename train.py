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
import cv2
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

