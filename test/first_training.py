from src.TransformerLayers import Transformer
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import pickle


import sentencepiece as spm, os, io

if os.getcwd() == r"D:\Desktop\Diss\ad945-diss-project\test":
    os.chdir("..")

with io.open('data/new_sentences.dat', mode='rb') as f:
    pairs = pickle.load(f)

pairs = pairs[:-1]

pairs = [(pairs[i], pairs[i+1]) for i in range(0, len(pairs), 2)]

model_dim = 512
heads = 8
N = 1
src_vocab = 2000
src_pad = src_vocab-1
trg_vocab = 2000
trg_pad = trg_vocab-1

model = Transformer(src_vocab, trg_vocab, model_dim, model_dim*4, heads, N)

for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

optim = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)



with io.open('data/pairs.de', 'r', encoding='utf-8') as f:
    pairs_de = f.read()
with open('data/pairs.en', 'r', encoding='utf-8') as f:
    pairs_en = f.read()


encoder_de = spm.SentencePieceProcessor()
encoder_de.Init(model_file='data/pairs.de.model')
encoder_en = spm.SentencePieceProcessor()
encoder_en.Init(model_file='data/pairs.en.model')
pairs_de_list = pairs_de.split("\n")
pairs_en_list = pairs_en.split("\n")

src_list = encoder_de.encode(pairs_de_list, out_type=int)
trg_list = encoder_en.encode(pairs_en_list, out_type=int)

src_pad = encoder_de.piece_to_id('<pad>')
trg_pad = encoder_en.piece_to_id('<pad>')

for i in range(len(pairs_de_list)):
    src = src_list[i]
    trg = trg_list[i]

    if len(src) < len(trg):
        for _ in range(len(src), len(trg)):
            src.append(src_pad)
    elif len(trg) < len(src):
        for _ in range(len(trg), len(src)):
            trg.append(trg_pad)


def batch(src_list, trg_list):

    for i in range(len(src_list)):
        yield src_list[i], trg_list[i]


if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"

device = torch.device(dev)


def train_model(epochs, print_every=100):
    model.train()

    start = time.time()
    temp = start
    total_loss = 0

    for epoch in range(epochs):
        optim.zero_grad()
        tl = 0
        for i, pair in enumerate(batch(src_list, trg_list)):
            src, trg = pair

            trg_mask = np.array(trg) != trg_pad

            nopeak_mask = np.triu(np.ones((len(trg), len(trg))), k=1).astype(np.uint8)
            nopeak_mask = (nopeak_mask+1) % 2

            # nopeak_mask = nopeak_mask & trg_mask

            for i in range(len(nopeak_mask)):
                nopeak_mask[i] = nopeak_mask[i] & trg_mask
            # print(nopeak_mask)
            #
            trg_mask = torch.from_numpy(nopeak_mask).to(device)
            # print(nopeak_mask.shape)

            src_mask = torch.tensor(np.array(src) != src_pad).to(device)#.unsqueeze(1)

            src_tensor = torch.tensor([
                src #for _ in range(len(trg))
            ])

            trg_tensor = torch.tensor([
                trg for _ in range(len(trg))
            ])

            enc_output = model.encoder(src_tensor, src_mask)
            enc_output = enc_output.repeat(len(src), 1, 1)

            src_mask = src_mask.unsqueeze(1)

            dec_output = model.decoder(trg_tensor, enc_output, src_mask, trg_mask)

            preds = model.linear(dec_output)

            # preds = model(src_tensor, trg_tensor, src_mask, trg_mask)
            # print(preds.view(-1, preds.size(-1)).shape)
            # print(preds)
            target = trg_tensor.contiguous().view(-1)

            loss = F.cross_entropy(preds.view(-1, preds.size(-1)), target,
                                   ignore_index=trg_pad)
            loss.backward()
            total_loss += loss.item()

            if (i+1) % print_every == 0:
                print(i+1//print_every, total_loss / (i+1))
            # print(loss)

        optim.step()

        if (epoch + 1) % (print_every//10) == 0:
            avg = total_loss / print_every * 10
            print(f"time: {time.time() - temp}, loss = {avg}, epoch = {epoch}")
            total_loss = 0
            temp = time.time()


train_model(1000, 10)


