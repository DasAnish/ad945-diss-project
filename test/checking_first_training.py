from src.TransformerLayers import Transformer
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import pickle

vocab_size=2000
model_dim = 512
heads = 8
N = 1
args = (vocab_size, vocab_size, model_dim, model_dim*4, heads, N, max_len)

import sentencepiece as spm, os, io

if os.getcwd() == r"D:\Desktop\Diss\ad945-diss-project\test":
    os.chdir("..")

model_names = 'test/models/de-en-model_'

with io.open('data/pairs.de', 'r', encoding='utf-8') as f:
    pairs_de = f.read()

encoder_de = spm.SentencePieceProcessor()
encoder_de.Init(model_file='data/pairs.de.model')

src_list = encoder_de.encode(pairs_de.split('\n'))

encoder_en = spm.SentencePieceProcessor()
encoder_en.Init(model_file='data/pairs.en.model')

for model_num in range(30, 330, 30):
    model_name = f"{model_names}{model_num}"

    # loading model
    model = Transformer(*args)
    model.load_state_dict(torch.load(model_name))

