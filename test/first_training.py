from src.TransformerLayers import Transformer
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import pickle


import tokenizer, os, io

if os.getcwd() == r"D:\Desktop\Diss\ad945-diss-project\test":
    os.chdir("..")

with io.open('data/new_sentences.dat', mode='rb') as f:
    pairs = pickle.load(f)

pairs = pairs[:-1]

pairs = [(pairs[i], pairs[i+1]) for i in range(0, len(pairs), 2)]

model_dim = 512
heads = 8
N = 1
src_vocab = 1100+2
src_pad = src_vocab-1
trg_vocab = 1000+2
trg_pad = trg_vocab-1

model = Transformer(src_vocab, trg_vocab, model_dim, model_dim*4, heads, N)

for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

optim = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

with open('data/list_of_idx.dat', 'rb') as f:
    import pickle
    en_list = pickle.load(f)
    de_list = pickle.load(f)

en_list = [i.lower() for i in en_list]
de_list = [i.lower() for i in de_list]

en_func = lambda i: en_list.index(i)
de_func = lambda i: de_list.index(i)


def train_model(epochs, print_every=100):
    model.train()

    start = time.time()
    temp = start
    total_loss = 0

    for epoch in range(epochs):
        optim.zero_grad()
        tl = 0
        for i, pair in enumerate(pairs):
            src, trg = pair
            src = [de_func(i.txt.lower()) for i in tokenizer.tokenize(src) if i.txt is not None]
            trg = [en_func(i.txt.lower()) for i in tokenizer.tokenize(trg) if i.txt is not None]

            # print(i, len(src), len(trg), end=" | ")

            if len(src) < len(trg):
                for i in range(len(src), len(trg)):
                    src.append(src_pad)
            else:
                for i in range(len(trg), len(src)):
                    trg.append(trg_pad)

            src_mask = torch.tensor(np.array(src) != src_pad).unsqueeze(1)
            trg_mask = np.array(trg) != trg_pad

            nopeak_mask = np.triu(np.ones((len(trg), len(trg))), k=1).astype(np.uint8)
            nopeak_mask = (nopeak_mask+1) % 2

            for i in range(len(nopeak_mask)):
                nopeak_mask[i] = nopeak_mask[i] & trg_mask
            # print(nopeak_mask)

            trg_mask = torch.from_numpy(nopeak_mask)
            # print(nopeak_mask.shape)

            src_tensor = torch.tensor([
                src for _ in range(len(trg))
            ])

            trg_tensor = torch.tensor([
                trg for _ in range(len(trg))
            ])

            preds = model(src_tensor, trg_tensor, src_mask, trg_mask)
            # print(preds.view(-1, preds.size(-1)).shape)
            # print(preds)
            target = trg_tensor.contiguous().view(-1)

            loss = F.cross_entropy(preds.view(-1, preds.size(-1)), target,
                                   ignore_index=trg_pad)
            loss.backward()
            total_loss += loss.item()
            # print(loss)

        optim.step()

        if (epoch + 1) % print_every == 0:
            avg = total_loss / print_every
            print(f"time: {time.time() - temp}, loss = {avg}, epoch = {epoch}")
            total_loss = 0
            temp = time.time()


train_model(1000, 1)


