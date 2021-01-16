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

print("DEBUG:: RUNNING SMALLER ATTRIBUTES")
# print("DEBUG:: RUNNING NORMAL ATTRIBUTES")
model_dim = 4
heads = 1
N = 1
src_vocab = 2000
src_pad = src_vocab-1
trg_vocab = 2000
trg_pad = trg_vocab-1

log = Log()#file_name)

model = Transformer(src_vocab, trg_vocab, model_dim, model_dim*4, heads, N)
# print(model.state_dict().keys())
# raise Exception

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


def batch(size, src_list, trg_list, shuffle=True):
    print(size)
    src_np = np.array(src_list)
    trg_np = np.array(trg_list)
    indexes = np.arange(len(src_np))
    # indexes = np.roll(indexes, 1800+500)
    while True:
        if shuffle:
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
model_prefix = 'de-en-model-'
path='models'
starting_index = 0


def train_model(batch_size, epochs, print_every=k, save_every=5 * k, eval_every=10 * k):
    model.train()

    start = time.time()
    temp = start
    total_loss = 0
    total_loss_at_save = 0
    last_loss = -1
    r = save_every / print_every

    batches = batch(batch_size, train_src_list, train_trg_list, shuffle=False)

    no_peak_mask = np.triu(np.ones((max_len - 1, max_len - 1)), k=1).astype(np.uint8)
    no_peak_mask = (no_peak_mask == 0) * 1
    # _src_list, _trg_list = next(batches)

    # batches = batch(batch_size, _src_list, _trg_list)

    for epoch, (src_lis, trg_lis) in enumerate(batches):
        if epoch > epochs: break

        optim.zero_grad()

        src_mask = torch.tensor(np.array(src_lis) != src_pad).to(device).unsqueeze(1)
        src_tensor = torch.LongTensor(src_lis).to(device)
        src_tensor.requires_grad = False

        trg_np = np.array(trg_lis)

        trg_mask = torch.tensor(trg_np[:, :-1] != trg_pad).unsqueeze(1)
        trg_mask = (trg_mask & no_peak_mask).to(device)

        trg_tensor = torch.LongTensor(trg_np[:, :-1]).to(device)
        target = torch.LongTensor(trg_np[:, 1:]).to(device).contiguous().view(-1)

        preds = model(src_tensor, trg_tensor, src_mask, trg_mask)
        preds = preds.view(-1, preds.size(-1))

        loss = F.cross_entropy(preds, target, ignore_index=trg_pad)

        loss.backward()
        optim.step()

        del src_mask, src_tensor, trg_mask, trg_tensor, preds, loss
        torch.cuda.empty_cache()

        if (epoch + 1) % print_every == 0:
            avg = "%.3f" % (total_loss)
            t = "%.3f" % (time.time() - temp)
            tt = time.strftime('%H:%M:%S', time.gmtime(time.time() - start))
            print(f"time: {t}s, total: {tt}, cat_loss = {avg}, epoch = {epoch + 1}")
            total_loss_at_save += total_loss
            total_loss = 0
            temp = time.time()

        if (epoch + 1) % save_every == 0:
            model_name = f'{path}/{model_prefix}{starting_index + epoch + 1}'
            # model.save_model(model_name)
            # save(model, model_name, optim, optim_file)
            avg = (total_loss_at_save / save_every)
            total_loss_at_save = 0
            diff = avg - last_loss
            avg = '%.3f' % avg
            diff = '%.3f' % diff
            last_loss = avg
            log.print(f"****Saving model: {model_name} | avg_loss: {avg} | diff: {diff}****")

        if (epoch + 1) % eval_every == 0:
            eval(src_list=src_lis, trg_list=trg_lis)
            model.train()

        log.flush()


# try:
#     train_model(2, 1000)
# except Exception as e:
#     log.print(e, type=Log.ERROR)
#     raise e
#
# log.close()

def eval(batch_size=5, model_file=None, src_list=test_src_list, trg_list=test_trg_list):
    if model_file is not None:
        model.load_state_dict(torch.load(model_file))
    model.eval()
    model.to(device)

    # batch_size = 5

    batches = batch(batch_size, src_list, trg_list, shuffle=False)

    src_lis, trg_lis = next(batches)

    trg_np = np.array(trg_lis)
    src_np = np.array(src_lis)

    src_mask = torch.tensor(np.array(src_lis) != src_pad).to(device).unsqueeze(1)

    src_tensor = torch.LongTensor(src_lis).to(device)

    trg_tensor = torch.LongTensor([[trg_start]
                                   for j in range(batch_size)])
    trg_tensor = trg_tensor.to(device)
    trg = np.array([[trg_start] for _ in range(batch_size)])
    trg_pads = np.array([[trg_pad] for _ in range(batch_size)])

    enc_output = model.encoder(src_tensor, src_mask)

    for i in range(1, max_len):
        trg_tensor = torch.LongTensor(np.concatenate([trg, trg_pads], axis=1)).to(device)
        trg_mask = ((trg_tensor != trg_pad) * 1).unsqueeze(1)

        dec_output = model.decoder(trg_tensor, enc_output, src_mask, trg_mask)
        preds = model.linear(dec_output)
        preds = F.softmax(preds)

        output = torch.argmax(preds[:, i], dim=1).unsqueeze(1).tolist()
        output = np.array(output)
        trg = np.concatenate([trg, output], axis=1)

        del trg_mask, dec_output, preds, output, trg_tensor
        torch.cuda.empty_cache()
    print("This output is not SAVED")
    for i in range(batch_size):
        print("TRG:", ''.join(encoder_en.decode(trg_lis[i].tolist())).replace('_', ' '))
        print("OUT:", ''.join(encoder_en.decode(trg[i].tolist())).replace('_', ' '))


eval()
