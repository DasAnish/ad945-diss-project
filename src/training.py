import torch
import torch.nn as nn
import torch.nn.functional as F
Tensor = torch.Tensor

from TransformerLayers import Transformer
from utils import Log

import sacrebleu

import pickle
import numpy as np
import time
import os
import sentencepiece as spm

Log().close()
log = Log()

input_file = 'data/News-Commentary.de-en.'
model_file = 'data/SPM.de-en.'

vocab_size = 10000

# making the vocab file if it hasn't been made
if not os.path.exists(f"{model_file}de.model") or not os.path.exists(f"{model_file}en.model"):
    for l in ['de', 'en']:
        spm.SentencePieceTrainer.Train(
            input=f"{input_file}{l}",
            model_prefix=f"{model_file}{l}",
            model_type='bpe',
            vocab_size=vocab_size,
            pad_id=3,
            pad_piece='<p>',
            bos_piece='<s>',
            eos_piece='</s>'
            )
# raise Exception
# Initializing the sentencepiece encoders
encoder_de = spm.SentencePieceProcessor()
encoder_de.load(model_file=f'{model_file}de.model')
encoder_en = spm.SentencePieceProcessor()
encoder_en.load(model_file=f'{model_file}en.model')

src_pad = encoder_de.pad_id()
trg_pad = encoder_en.pad_id()

src_start = encoder_de.bos_id()
trg_start = encoder_en.bos_id()

src_end = encoder_de.eos_id()
trg_end = encoder_en.eos_id()

tokenized_split_data_file = 'data/tokenized_split_data'

max_len = 75

# loading the tokenized bins
if os.path.exists(tokenized_split_data_file):
    log.print('loading from tokenized bins file')
    with open(tokenized_split_data_file, 'rb') as f:
        train_src_bins = pickle.load(f)
        train_trg_bins = pickle.load(f)

else:
    # Tokenizing the raw files
    with open(f'{input_file}de', encoding='utf-8') as f:
        pairs_de = f.read()
    with open(f'{input_file}en', encoding='utf-8') as f:
        pairs_en = f.read()

    _src_list = pairs_de.split("\n")
    _trg_list = pairs_en.split('\n')

    _src_list = [[src_start] + encoder_de.encode(s) + [src_end]
                 for s in _src_list]
    _trg_list = [[trg_start] + encoder_en.encode(s) + [trg_end]
                 for s in _trg_list]

    log.print('Tokenized')

    train_src_bins = {30: [], 40: [], 50: [], 60: [], 75: []}
    train_trg_bins = {30: [], 40: [], 50: [], 60: [], 75: []}

    for src, trg in zip(_src_list, _trg_list):

        # filtering out the long sentences
        if len(src) > max_len or len(trg) > max_len:
            continue

        lsrc = len(src)
        ltrg = len(trg)

        # sorting the sentences according to size
        for v in [30, 40, 50, 60, 75]:
            if lsrc <= v and ltrg <= v:
                for _ in range(lsrc, v):
                    src.append(src_pad)
                for _ in range(ltrg, v):
                    trg.append(trg_pad)

                train_src_bins[v].append(src)
                train_trg_bins[v].append(trg)
                break

    log.print("Filtered long sentences")

log.print({v: len(train_src_bins[v]) for v in train_src_bins})


# Function to save the model and optimizer
def save(model, model_file, optim, optim_file):
    torch.save(model.state_dict(), model_file)
    torch.save(optim.state_dict(), optim_file)


# Creates shuffles the list and creates a batch of batch_size
def batch(size, src_list, trg_list, shuffle=True):
    src_np = np.array(src_list)
    trg_np = np.array(trg_list)
    indexes = np.arange(len(src_np))
    # indexes = np.roll(indexes, 1800+500)
    while True:
        if not shuffle:
            np.random.shuffle(indexes)
            src_np = src_np[indexes]
            trg_np = trg_np[indexes]

        for i in range(0, len(src_np), size):
            temp_src = src_np[i:i+size]
            temp_trg = trg_np[i:i+size]

            yield (temp_src, temp_trg)


# creates a fixed token_size batches
def batch_for_bins(token_size, src_bins, trg_bins, shuffle=True):
    bins = [30, 40, 50, 60, 75]

    src_bins_np = {v: np.array(src_bins[v]) for v in bins}
    trg_bins_np = {v: np.array(trg_bins[v]) for v in bins}

    arange = {v: np.arange(len(src_bins[v])) for v in src_bins}

    while True:
        for v in bins:
            src_lis = src_bins_np[v]
            trg_lis = trg_bins_np[v]
            if shuffle:
                np.random.shuffle(arange[v])
                src_lis = src_lis[arange[v]]
                trg_lis = trg_lis[arange[v]]

            batch_size = token_size // v

            for i in range(0, len(src_lis), batch_size):
                # print(i, end=', ')
                src = src_lis[i: i + batch_size]
                trg = trg_lis[i: i + batch_size]

                yield src, trg, v


# if the tokenized file isn't made then we save it
if not os.path.exists(tokenized_split_data_file):
    with open(tokenized_split_data_file, 'wb') as f:
        pickle.dump(train_src_bins, f)
        pickle.dump(train_trg_bins, f)


# Defining the model variables
path = 'models'
model_prefix = 'de-en-model-'
optim_file = 'data/optim_state_dict'
model_dim = 512
heads = 8
N = 6
args = (vocab_size, vocab_size, model_dim, model_dim*4, heads, N, max_len)

# checking the dev
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"

device = torch.device(dev)

model = Transformer(*args)
starting_index = 0

# initializing the parameters of the model.
if not os.path.exists(path):
    if not os.path.exists(path):
        os.mkdir(path)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    optim = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    log.print(f"No {path} found. Created a new path directory and started using xavier_uniform")
else:
    for i in os.walk(path):
        break
    i = i[2]
    m = 0
    mf = None
    for file in i:
        num = int(file[len(model_prefix):])
        if num > m:
            m = num
            mf = file

    if mf is not None:
        log.print(f"Starting from last saved {mf}")
        model.load_state_dict(torch.load(f"{path}/{mf}", map_location=device))
        optim = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
        starting_index = m
    else:
        log.print(f"Starting from xavier_uniform distribution")
        optim = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)


# The evaluation function that will predict the sentence
def eval(batch_size=5, model_file=None, src_list=train_src_bins[30], trg_list=train_trg_bins[30]):
    if model_file is not None:
        model.load_state_dict(torch.load(model_file))
    model.eval()
    model.to(device)

    batches = batch(batch_size, src_list, trg_list, shuffle=False)

    src_lis_, trg_lis_ = next(batches)
    trg_sentences = []
    out_sentences = []
    blue_score = 0
    for i, (src_lis, trg_lis) in enumerate(batch(1, src_lis_, trg_lis_, shuffle=False)):
        if i >= batch_size: break

        src_mask = torch.tensor(np.array(src_lis) != src_pad).to(device).unsqueeze(1)

        src_tensor = torch.LongTensor(src_lis).to(device)

        trg = np.array([[trg_start] for _ in range(1)])
        trg_pads = np.array([[trg_pad] for _ in range(1)])

        enc_output = model.encoder(src_tensor, src_mask)

        for i in range(1, max_len):
            trg_tensor = torch.LongTensor(np.concatenate([trg, trg_pads], axis=1)).to(device)
            trg_mask = ((trg_tensor != trg_pad) * 1).unsqueeze(1)

            dec_output = model.decoder(trg_tensor, enc_output, src_mask, trg_mask)
            preds = model.linear(dec_output)
            preds = F.softmax(preds, dim=-1)

            output = torch.argmax(preds[:, i], dim=1).unsqueeze(1).tolist()
            output = np.array(output)
            trg = np.concatenate([trg, output], axis=1)

            o = output[0]

            del trg_mask, dec_output, preds, output, trg_tensor
            torch.cuda.empty_cache()

            if o == trg_end: break

        del enc_output
        torch.cuda.empty_cache()

        for i in range(1):

            trg_sentence = ''.join(encoder_en.decode(trg_lis[i].tolist())).replace('_', ' ')
            out_sentence = ''.join(encoder_en.decode(trg[i].tolist())).replace('_', ' ')

            trg_sentences.append(trg_sentence)
            out_sentences.append(out_sentence)

        del src_mask, src_tensor
        torch.cuda.empty_cache()

    blue_score = sacrebleu.corpus_bleu(out_sentences, [trg_sentences])
    print(blue_score)


model_prefix = 'de-en-model-'


# The algorithm that trains the batch_size
def train_model(batch_size, epochs, print_every, save_every, eval_every):

    model.train()

    start = time.time()
    temp = start
    total_loss = 0
    total_loss_at_save = 0
    last_loss = 0
    r = save_every / print_every

    batches = batch_for_bins(batch_size, train_src_bins, train_trg_bins, shuffle=True)

    for epoch, (src_lis, trg_lis, ml) in enumerate(batches):
        if epoch > epochs: break

        optim.zero_grad()

        src_mask = torch.tensor(np.array(src_lis) != src_pad).to(device).unsqueeze(1)
        src_tensor = torch.LongTensor(src_lis).to(device)
        src_tensor.requires_grad = False

        trg_np = np.array(trg_lis)

        no_peak_mask = np.triu(np.ones((ml - 1, ml - 1)), k=1).astype(np.uint8)
        no_peak_mask = (no_peak_mask == 0) * 1

        trg_mask = torch.tensor(trg_np[:, :-1] != trg_pad).unsqueeze(1)
        trg_mask = (trg_mask & no_peak_mask).to(device)

        trg_tensor = torch.LongTensor(trg_np[:, :-1]).to(device)
        target = torch.LongTensor(trg_np[:, 1:]).to(device).contiguous().view(-1)

        preds = model(src_tensor, trg_tensor, src_mask, trg_mask)
        preds = preds.view(-1, preds.size(-1))

        loss = F.cross_entropy(preds, target, ignore_index=trg_pad, reduction='sum')

        loss.backward()

        total_loss += loss.item()

        optim.step()

        del src_mask, src_tensor, trg_mask, trg_tensor, preds, loss
        torch.cuda.empty_cache()

        if (epoch + 1) % print_every == 0:
            avg = "%.3f" % (total_loss)
            t = "%.3f" % (time.time() - temp)
            tt = time.strftime('%H:%M:%S', time.gmtime(time.time() - start))
            log.print(f"time: {t}s, total: {tt}, cat_loss = {avg}, epoch = {epoch + 1}")
            total_loss_at_save += total_loss
            total_loss = 0
            temp = time.time()

        if (epoch + 1) % save_every == 0:
            model_name = f'{path}/{model_prefix}{starting_index + epoch + 1}'

            save(model, model_name, optim, optim_file)

            avg = (total_loss_at_save / r)
            total_loss_at_save = 0
            diff = avg - last_loss
            last_loss = avg

            avg = '%.3f' % avg
            diff = '%.3f' % diff

            log.print(f"****Saving model: {model_name} | avg_loss: {avg} | diff: {diff}****")

        if (epoch + 1) % eval_every == 0:
            eval(src_list=train_src_bins[30], trg_list=train_trg_bins[30])
            model.train()

        log.flush()

# print("*******************HERE********************")
if __name__ == '__main__':
    try:
        eval()
        log.print("Starting the training")
        bs = 200 * 75
        train_model(bs, 200000, eval_every=3000, save_every=3000, print_every=300)

    except Exception as e:
        log.print(e, type=Log.ERROR)
        log.flush()
        raise e
