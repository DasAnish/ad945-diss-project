from src.TransformerLayers import Transformer
from src.utils import Log
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

log = Log()#file_name)

model = Transformer(src_vocab, trg_vocab, model_dim, model_dim*4, heads, N)

for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

optim = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

input_file = 'data/pairs.'
model_file = 'data/pairs.'

with io.open('data/pairs.de', 'r', encoding='utf-8') as f:
    pairs_de = f.read()
with open('data/pairs.en', 'r', encoding='utf-8') as f:
    pairs_en = f.read()

encoder_de = spm.SentencePieceProcessor()
encoder_de.Init(model_file=f'{model_file}de.model')
encoder_en = spm.SentencePieceProcessor()
encoder_en.Init(model_file=f'{model_file}en.model')

src_pad = encoder_de.piece_to_id('<pad>')
trg_pad = encoder_en.piece_to_id('<pad>')

src_start = encoder_de.piece_to_id('<s>')
trg_start = encoder_en.piece_to_id('<s>')

trg_end = encoder_en.piece_to_id('</s>')
src_end = encoder_en.piece_to_id('</s>')

tokenized_split_data_file = 'data/tokenized_split_data'

max_len = 75

if os.path.exists(tokenized_split_data_file):
    with open(tokenized_split_data_file, 'rb') as f:
        train_src_list = pickle.load(f)
        train_trg_list = pickle.load(f)
        test_src_list = pickle.load(f)
        test_trg_list = pickle.load(f)

else:
    with open(f'{input_file}de', encoding='utf-8') as f:
        pairs_de = f.read()
    with open(f'{input_file}en', encoding='utf-8') as f:
        pairs_en = f.read()

    _src_list = encoder_de.encode(
        pairs_de.split('\n')
    )
    _trg_list = encoder_en.encode(
        pairs_en.split('\n')
    )

    log.print('Tokenized')

    src_list = []
    trg_list = []
    for src, trg in zip(_src_list, _trg_list):
        if len(src) > max_len - 2 or len(trg) > max_len - 2:
            continue

        src = [src_start] + src + [src_end]
        trg = [trg_start] + trg + [trg_end]

        for _ in range(len(src), max_len):
            src.append(src_pad)

        for _ in range(len(trg), max_len):
            trg.append(trg_pad)

        src_list.append(src)
        trg_list.append(trg)

    log.print("Filtered long sentences")


def train_test_split(src_list, trg_list):
    indexes = np.arange(len(src_list))
    np.random.shuffle(indexes)
    src_np = np.array(src_list)
    src_np = src_np[indexes]
    trg_np = np.array(trg_list)
    trg_np = trg_np[indexes]

    l = (len(src_list) * 4) // 5
    return src_np[:l], trg_np[:l], src_np[l:], trg_np[l:]


def batch(size, src_list, trg_list):
    src_np = np.array(src_list)
    trg_np = np.array(trg_list)
    indexes = np.arange(len(src_np))
    # indexes = np.roll(indexes, 1800+500)
    while True:
        np.random.shuffle(indexes)
        src_np = src_np[indexes]
        trg_np = trg_np[indexes]

        for i in range(0, len(src_np), size):
            temp_src = src_np[i:i+size]
            temp_trg = trg_np[i:i+size]

            yield (temp_src, temp_trg)


if not os.path.exists(tokenized_split_data_file):
    train_src_list, train_trg_list, test_src_list, test_trg_list = train_test_split(src_list, trg_list)
    log.print("Done the train/test split 80/20")
    with open('data/tokenized_split_data', 'wb') as f:
        # pickle.dump(['train_src_list', 'train_trg_list', 'test_src_list', 'test_trg_list'], f)
        pickle.dump(train_src_list, f)
        pickle.dump(train_trg_list, f)
        pickle.dump(test_src_list, f)
        pickle.dump(test_trg_list, f)


if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"

device = torch.device(dev)
k = 10


def train_model(epochs, print_every=k, save_every=3 * k):
    model.train()
    path = 'models'
    if not os.path.exists(path):
        os.mkdir(path)

    start = time.time()
    temp = start
    total_loss = 0

    batches = batch(100, train_src_list, train_trg_list)

    for epoch, (src_lis, trg_lis) in enumerate(batches):
        if epoch > epochs: break

        optim.zero_grad()

        src_mask = torch.tensor(np.array(src_lis) != src_pad).to(device)
        src_tensor = torch.LongTensor(src_lis).to(device)
        src_tensor.requires_grad = False
        # print(src_tensor.shape)

        trg_np = np.array(trg_lis)
        trg_tensor = torch.LongTensor(trg_np).to(device)
        trg_tensor.rquires_grad = False
        trg_mask_ = trg_np != trg_pad

        # target = trg_tensor.contiguous().view(-1)
        # target.requires_grad = False

        no_peak_mask = np.triu(np.ones((max_len, max_len)), k=1).astype(np.uint8)
        no_peak_mask = (no_peak_mask == 0)

        # trg_mask_np = np.empty((len(src_list), max_len))

        enc_output = model.encoder(src_tensor, src_mask)

        for i in range(0, max_len - 1):
            # print(i, "%.3f"%(time.time() - start))

            trg_mask_np = np.array([no_peak_mask[i] & mask for mask in trg_mask_])

            trg_mask = torch.LongTensor(trg_mask_np).to(device)

            dec_output = model.decoder(trg_tensor, enc_output, src_mask, trg_mask)

            preds = model.linear(dec_output)

            # target_np =(trg_mask_np==0)*3 + trg_np * trg_mask_np
            # target = torch.LongTensor(target_np).contiguous().view(-1).to(device)
            # target.require_grad = False

            target = trg_tensor[:, i + 1].contiguous().view(-1)

            _preds = preds[:, i + 1, :]

            loss = F.cross_entropy(_preds, target, ignore_index=trg_pad)
            loss.backward(retain_graph=True)

            total_loss += loss.item()

            del dec_output, preds, loss, _preds
            torch.cuda.empty_cache()

        optim.step()
        del src_tensor, src_mask, trg_tensor, enc_output
        torch.cuda.empty_cache()

        if True:  # (epoch + 1) % print_every == 0:
            avg = "%.3f" % (total_loss / print_every * 10)
            t = "%.3f" % (time.time() - temp)
            tt = time.strftime('%H:%M:%S', time.gmtime(time.time() - start))

            log.print(f"time: {t}s, total: {tt}m, cat_loss = {avg}, epoch = {epoch}")
            total_loss = 0
            temp = time.time()

        if (epoch + 1) % save_every == 0:
            model_name = f'{path}/de-en-model-{epoch + 1}'
            model.save_model(model_name)
            log.print(f"***Saving model: {model_name}***")


try:
    train_model(2000)
except Exception as e:
    log.print(e, type=Log.ERROR)

log.close()

