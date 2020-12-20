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

# print("DEBUG:: RUNNING SMALLER ATTRIBUTES")
print("DEBUG:: RUNNING NORMAL ATTRIBUTES")
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

src_pad = encoder_de.piece_to_id('<pad>')
trg_pad = encoder_en.piece_to_id('<pad>')

src_start = encoder_en.piece_to_id('<s>')
trg_start = encoder_de.piece_to_id('<s>')
src_end = encoder_en.piece_to_id('</s>')
trg_end = encoder_de.piece_to_id('</s>')

max_len = 80
src_list = encoder_de.encode(pairs_de.split('\n'))
trg_list = encoder_en.encode(pairs_en.split('\n'))

final_src_list = []
final_trg_list = []

for i in range(len(src_list)):
    src = src_list[i]
    trg = trg_list[i]

    if len(src) > max_len or len(trg) > max_len:
        continue

    src = [src_start] + src + [src_end]
    trg = [trg_start] + trg + [trg_end]

    for _ in range(len(src), max_len):
        src.append(src_pad)

    for _ in range(len(trg), max_len):
        trg.append(trg_pad)

    final_src_list.append(src)
    final_trg_list.append(trg)





#
# for i in range(len(pairs_de_list)):
#     src = src_list[i]
#     trg = trg_list[i]
#
#     for _ in range(len(src), max_len):
#         src.append(src_pad)
#     for _ in range(len(trg), max_len):
#         trg.append(trg_pad)


if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"

device = torch.device(dev)

k = 10

def train_model(epochs, print_every=k):
    model.train()

    start = time.time()
    temp = start
    total_loss = 0

    for epoch in range(epochs):
        optim.zero_grad()

        src_mask = torch.tensor(np.array(src_list) != src_pad).to(device)
        src_tensor = torch.LongTensor(src_list).to(device)
        src_tensor.requires_grad = False
        # print(src_tensor.shape)

        trg_np = np.array(trg_list)
        trg_tensor = torch.LongTensor(trg_np)
        trg_tensor.rquires_grad = False
        trg_mask_ = trg_np != trg_pad

        # target = trg_tensor.contiguous().view(-1)
        # target.requires_grad = False

        no_peak_mask = np.triu(np.ones((max_len, max_len)), k=1).astype(np.uint8)
        no_peak_mask = (no_peak_mask == 0)

        # trg_mask_np = np.empty((len(src_list), max_len))

        enc_output = model.encoder(src_tensor, src_mask)

        for i in range(0, max_len-1):
            print(i, "%.3f"%(time.time() - start))

            trg_mask_np = np.array([no_peak_mask[i] & mask for mask in trg_mask_])

            trg_mask = torch.LongTensor(trg_mask_np).to(device)

            dec_output = model.decoder(trg_tensor, enc_output,
                                       src_mask, trg_mask)

            preds = model.linear(dec_output)

            target_np =(trg_mask_np==0)*3 + trg_np * trg_mask_np
            target = torch.LongTensor(target_np).contiguous().view(-1)
            target.require_grad = False

            preds = preds.view(-1, preds.size(-1))

            loss = F.cross_entropy(preds, target, ignore_index=trg_pad)
            loss.backward(retain_graph=True)

            total_loss += loss.item()

            del dec_output, preds, loss
            torch.cuda.empty_cache()

        optim.step()
        # del src_tensor, src_mask, trg_tensor, enc_output
        # torch.cuda.empty_cache()

        if True:#(epoch + 1) % (print_every//10) == 0:
            avg = total_loss / print_every * 10
            print(f"time: {time.time() - temp}, loss = {avg}, epoch = {epoch}")
            total_loss = 0
            temp = time.time()

        if (epoch + 1) % print_every:
            print("************SAVING MODEL*****************")
            model_name = f'{epoch+1}'
            model.save(model_name)


train_model(1000, k)


