import torch
import torch.nn.functional as F
from torch.autograd import Variable
from datetime import datetime
from tqdm.notebook import tnrange
from src.opt import Opt
import re
import math
import numpy as np
import os
from src.transformer_layers import Transformer
import torch.nn as nn


class Log:
    """
    A logger that notes the date/time with the text provided.
    """

    LOG, ERROR = 0, 1

    def __init__(self, outfile='data/.log', filename='data/logfile.log'):
        self.filename = filename
        self.outfile = outfile
        self.file_object = open(filename, 'a+', encoding='utf-8')
        self.line_num = 0
        print("LOGGING For seesion on: " + str(datetime.now()), file=self.file_object)

    def print(self, txt, type=LOG, shell=True):
        if shell: print(txt)
        prefix = "LOG ::" if type==Log.LOG else "ERROR ::"
        txt = f"{prefix} {str(datetime.now())} :: {txt}"
        print(txt, file=self.file_object)

    def close(self):
        self.file_object.seek(0, 0)
        text = self.file_object.read()
        text = text.split('\n')
        text.reverse()
        output = '\n'.join(text)
        self.file_object.close()

        with open(self.outfile, 'w') as f:
            f.write(output)

    def flush(self):
        self.file_object.close()
        self.file_object = open(self.filename, 'a+', encoding = 'utf-8')


def move():
    """The function that is used to construct the dataset"""
    opt = Opt.get_instance()

    def move_lang(lang):
        inpFile1 = open(opt.pc_input_file + lang, 'r', encoding='utf-8')
        inpFile2 = open(opt.nc_input_file + lang, 'r', encoding='utf-8')
        intFile = open(opt.interim_file + lang, 'w', encoding='utf-8')

        for _ in tnrange(int(opt.num_mil * 2 * 10**5)):
            intFile.write(inpFile2.readline())
        for _ in tnrange(int(opt.num_mil * 8 * 10**5)):
            intFile.write(inpFile1.readline())

        inpFile1.close()
        inpFile2.close()
        intFile.close()
    move_lang(opt.src_lang)
    move_lang(opt.trg_lang)


def load_dev_dataset():
    """The function which loads the dev dataset"""
    opt = Opt.get_instance()
    opt.dev_dataset = f'data/{opt.src_lang}/DEV-{opt.src_lang}-{opt.trg_lang}.'

    with open(opt.dev_dataset + opt.src_lang, 'r', encoding='utf-8') as f:
        opt.dev_src_sentences = f.read().split('\n')[:2000]
    with open(opt.dev_dataset + opt.trg_lang, 'r', encoding='utf-8') as f:
        opt.dev_trg_sentences = f.read().split('\n')[:2000]


def load_model():
    """The function used to load the model's parameters"""
    opt = Opt.get_instance()

    model = Transformer(*opt.args)
    optim = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    starting_index = 0
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    # initializing the parameters of the model.

    if not os.path.exists(opt.path):
        if not os.path.exists(opt.path):
            os.mkdir(opt.path)
        opt.log.print(f"No {opt.path} found. Created a new path directory and started using xavier_uniform")
    else:
        for i in os.walk(opt.path):
            break
        i = i[2]
        m = 0
        mf = None
        suffix_len = len('.model')
        for file in i:
            if opt.model_prefix not in file or '.model' not in file:
                continue

            num = int(file[len(opt.model_prefix):-suffix_len])
            if num > m:
                m = num
                mf = file[:-suffix_len]

        if mf is not None:
            opt.log.print(f"Starting from last saved {mf}")
            opt.save_model.load(f'{opt.path}/{mf}')
            model.load_state_dict(opt.save_model.model_state_dict)
            optim.load_state_dict(opt.save_model.optim_state_dict)
            starting_index = m
        else:
            opt.log.print(f"Starting from xavier_uniform distribution")
    opt.starting_index = starting_index

    return model, optim


def batch():
    """The batching generator"""
    opt = Opt.get_instance()
    max_count = {v: (len(opt.src_bins[v])*v) // opt.tokensize for v in opt.bins}
    # print(max_count)
    cur_count = {v: 0 for v in opt.bins}
    batch_sizes = {v: opt.tokensize // v for v in opt.bins}

    step = 0
    while len(max_count):
        v = np.random.choice(list(max_count.keys()))
        i = cur_count[v]
        cur_count[v] += 1
        j = cur_count[v]

        step += 1
        if step < opt.starting_index:
            continue

        size = batch_sizes[v]
        src_list = opt.src_bins[v][i*size: j*size]
        trg_list = opt.trg_bins[v][i*size: j*size]

        if j > max_count[v]:
            if opt.keep_training:
                cur_count[v] = 0
            else:
                del max_count[v]

        if len(src_list) == 0:
            continue

        yield src_list, trg_list


def nopeak_mask(size):
    """The function which generates an upper triangular matrix"""
    opt = Opt.get_instance()
    np_mask = np.triu(np.ones((1, size, size)),
                      k=1).astype('uint8')
    np_mask = Variable(torch.from_numpy(np_mask) == 0).to(opt.device)
    return np_mask


def create_masks(src, trg):
    """
    The function that makes the source and target mask
    :param src: The source batch
    :param trg: The target batch
    :return: the source and target mask
    """
    opt = Opt.get_instance()
    src_mask = (src != opt.src_pad).unsqueeze(-2)

    if trg is not None:
        trg_mask = (trg != opt.trg_pad).unsqueeze(-2)
        size = trg.size(1)  # get seq_len for matrix
        np_mask = nopeak_mask(size).to(opt.device)
        trg_mask = trg_mask & np_mask

    else:
        trg_mask = None
    return src_mask, trg_mask


def k_best_outputs(outputs, pred, log_probs, i, k):
    """The function that picks the top k output (part of beam search)"""
    probs, idx = pred[:, -1].data.topk(k)
    log_probs = torch.Tensor([math.log(p) for p in probs.data.view(-1)]).view(k, -1) + log_probs.transpose(0, 1)
    k_probs, k_ix = log_probs.view(-1).topk(k)

    row = k_ix // k
    col = k_ix % k

    outputs[:, :i] = outputs[row, :i]
    outputs[:, i] = idx[row, col]

    log_probs = k_probs.unsqueeze(0)

    return outputs, log_probs


def beam_search(src, model):
    """The function that implements the beam search (pruned breadth-first search)"""
    opt = Opt.get_instance()
    bos_token = opt.trg_bos
    src_mask = (src != opt.src_pad).unsqueeze(-2)
    encoder_output = model.encoder(src, src_mask)

    outputs = torch.LongTensor([[bos_token]]).to(opt.device)

    trg_mask = nopeak_mask(1)

    pred = model.linear(model.decoder(outputs, encoder_output, src_mask, trg_mask))
    pred = F.softmax(pred, dim=-1)

    probs, idx = pred[:, -1].data.topk(opt.k)
    log_probs = torch.Tensor([math.log(prob) for prob in probs.data[0]]).unsqueeze(0)

    outputs = torch.zeros(opt.k, opt.max_len).long().to(opt.device)
    outputs[:, 0] = bos_token
    outputs[:, 1] = idx[0]

    encoder_outputs = torch.zeros(opt.k, encoder_output.size(-2), encoder_output.size(-1)).to(opt.device)
    encoder_outputs[:, :] = encoder_output[0]

    eos_token = opt.trg_eos
    src_mask = (src != opt.src_pad).unsqueeze(-2)
    ind = None
    for i in range(2, opt.max_len):
        trg_mask = nopeak_mask(i)
        pred = model.linear(model.decoder(outputs[:, :i],
                                          encoder_outputs, src_mask, trg_mask))
        pred = F.softmax(pred, dim=-1)
        outputs, log_probs = k_best_outputs(outputs, pred, log_probs, i, opt.k)
        ones = torch.nonzero(outputs == eos_token)
        # ones = (outputs==eos_token).nonzero() # Occurrences of end symbols for all input sentences.
        sequence_lengths = torch.zeros(len(outputs)).to(opt.device)
        for vec in ones:
            i = vec[0]
            if sequence_lengths[i] == 0:  # First end symbol has not been found yet
                sequence_lengths[i] = vec[1]  # Position of first end symbol

        complete_sentence_count = len([s for s in sequence_lengths if s > 0])

        if complete_sentence_count == opt.k:
            alpha = 0.7
            div = 1 / (sequence_lengths.type_as(log_probs) ** alpha)
            _, ind = torch.max(log_probs * div, 1)
            ind = ind.data[0]
            break

    if ind is None:
        length = (outputs[0] == eos_token).nonzero()[0]
        sentence_list = (outputs[0][1:length]).tolist()
        return ''.join(opt.trg_processor.decode(sentence_list)).replace('_', " ")

    else:
        length = (outputs[ind] == eos_token).nonzero()[0]
        sentence_list = (outputs[0][1:length]).tolist()
        return ''.join(opt.trg_processor.decode(sentence_list)).replace('_', " ")


def multiple_replace(dict, text):
    """A function that uses regex to replace certain parts of the sentence to make it suitable for printing"""
    # compiling the regex based on dictionary
    regex = re.compile("(%s)" % "|".join(map(re.escape, dict.keys())))

    # For each match i.e. x look up the value in the dictionary to replace
    return regex.sub(lambda x: dict[x.string[x.start():x.end()]], text)


def translate_sentence(sentence, model):
    """The function that uses beam search to translate the sentences"""
    opt = Opt.get_instance()
    model.eval()
    sentence = Variable(torch.LongTensor([opt.src_processor.encode(sentence)])).to(opt.device)

    sentence = beam_search(sentence, model)

    return multiple_replace({' ?': '?', ' !': '!', ' .': '.', '\' ': '\'', ' ,': ','}, sentence)






