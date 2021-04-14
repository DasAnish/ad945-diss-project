import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Variable
from math import sqrt, sin, cos
from datetime import datetime
from tqdm.notebook import tnrange
import re, math
import numpy as np


class PositionalEncoding(nn.Module):

    """A parameter less module that concatenates a number of sine signals at the end of the embedded vectors."""

    def __init__(self, model_dim: int, max_length: int, dropout: float = 0.1):
        super().__init__()

        if torch.cuda.is_available():
            dev = "cuda:0"
        else:
            dev = "cpu"

        device = torch.device(dev)

        self.model_dim: int = model_dim
        self.dropout: nn.Dropout = nn.Dropout(dropout)
        self.add_module('dropout', self.dropout)

        position_vector: Tensor = torch.zeros(max_length, model_dim, requires_grad=False).to(device)
        # arange = torch.arange(max_length)

        # note to self: this appear to work right now.
        for pos in range(max_length):
            for i in range(0, model_dim, 2):
                # Follwing the formula provided in the paper.
                position_vector[pos, i] = sin(pos / (10000 ** ((2 * i) / model_dim)))
                position_vector[pos, i+1] = cos(pos / (10000 ** ((2 * (i + 1)) / model_dim)))

        # position_vector: max_seq_len x model_dim
        position_vector = position_vector.unsqueeze(0)

        # position_vector: 1 x max_seq_len x model_dim
        self.register_buffer('position_vector', position_vector)

    def forward(self, x: Tensor) -> Tensor:
        """
        The forward pass implementation of the Positional Embedding step.
        :param x: the tensor containing Batch x Seq_len x model_dim embeddings.
        :return: the embedded vector with some alterations.
        """

        x = x * sqrt(self.model_dim)
        sequence_length = x.size(1)
        x = x + Variable(self.position_vector[:, :sequence_length], requires_grad=False)
        x = self.dropout(x)
        return x


class FeedForward(nn.Module):

    """The utility module that is used to implement a feed-forward fully connected Neural Net for the
    encoder and decoder layers."""

    def __init__(self, model_dim: int, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()

        if torch.cuda.is_available():
            dev = "cuda:0"
        else:
            dev = "cpu"

        device = torch.device(dev)

        self.fc1: nn.Linear = nn.Linear(model_dim, d_ff).to(device)
        self.dropout: nn.Dropout = nn.Dropout(dropout).to(device)
        self.fc2: nn.Linear = nn.Linear(d_ff, model_dim).to(device)

        self.add_module('fc1', self.fc1)
        self.add_module('dropout', self.dropout)
        self.add_module('fc2', self.fc2)

    def forward(self, x: Tensor) -> Tensor:

        """
        Implements Feed-Forward algorithm with dropout.
        :param x: a tensor with last dim = model_dim
        :return: output from the NN
        """

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)

        return x


class Log:
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


def move(opt):
    def move_lang(lang):
        inpFile1 = open(opt.pc_input_file + lang, 'r', encoding='utf-8')
        inpFile2 = open(opt.nc_input_file + lang, 'r', encoding='utf-8')
        intFile = open(opt.interim_file + lang, 'w', encoding='utf-8')

        for i in tnrange(int(opt.num_mil * 2 * 10**5)):
            intFile.write(inpFile2.readline())
        for i in tnrange(int(opt.num_mil * 8 * 10**5)):
            intFile.write(inpFile1.readline())

        inpFile1.close()
        inpFile2.close()
        intFile.close()
    move_lang(opt.src_lang)
    move_lang(opt.trg_lang)


def load_dev_dataset(opt):
    opt.dev_dataset = f'data/{opt.src_lang}/DEV-{opt.src_lang}-{opt.trg_lang}.'

    with open(opt.dev_dataset + opt.src_lang, 'r', encoding='utf-8') as f:
        opt.dev_src_sentences = f.read().split('\n')[:2000]
    with open(opt.dev_dataset + opt.trg_lang, 'r', encoding='utf-8') as f:
        opt.dev_trg_sentences = f.read().split('\n')[:2000]


def batch(opt):
    max_count = {v:(len(opt.src_bins[v])*v) // opt.tokensize for v in opt.bins}
    # print(max_count)
    cur_count = {v:0 for v in opt.bins}
    batch_sizes = {v: opt.tokensize // v for v in opt.bins}

    step = 0
    while len(max_count):
        v = np.random.choice(list(max_count.keys()))
        i = cur_count[v]
        cur_count[v] += 1
        j = cur_count[v]

        step += 1
        if step < opt.starting_index: continue

        size = batch_sizes[v]
        src_list = opt.src_bins[v][i*size: j*size]
        trg_list = opt.trg_bins[v][i*size: j*size]

        if j > max_count[v]:
            if opt.keep_training: cur_count[v] = 0
            else: del max_count[v]

        if len(src_list) == 0:
            continue

        yield src_list, trg_list


def nopeak_mask(size, opt):
    np_mask = np.triu(np.ones((1, size, size)),
                      k=1).astype('uint8')
    np_mask = Variable(torch.from_numpy(np_mask) == 0).to(opt.device)
    return np_mask


def create_masks(src, trg, opt):
    src_mask = (src != opt.src_pad).unsqueeze(-2)

    if trg is not None:
        trg_mask = (trg != opt.trg_pad).unsqueeze(-2)
        size = trg.size(1)  # get seq_len for matrix
        np_mask = nopeak_mask(size, opt).to(opt.device)
        trg_mask = trg_mask & np_mask

    else:
        trg_mask = None
    return src_mask, trg_mask


def k_best_outputs(outputs, out, log_scores, i, k):
    probs, ix = out[:, -1].data.topk(k)
    log_probs = torch.Tensor([math.log(p) for p in probs.data.view(-1)]).view(k, -1) + log_scores.transpose(0, 1)
    k_probs, k_ix = log_probs.view(-1).topk(k)

    row = k_ix // k
    col = k_ix % k

    outputs[:, :i] = outputs[row, :i]
    outputs[:, i] = ix[row, col]

    log_scores = k_probs.unsqueeze(0)

    return outputs, log_scores


def beam_search(src, model, opt):
    init_tok = opt.trg_bos
    src_mask = (src != opt.src_pad).unsqueeze(-2)
    e_output = model.encoder(src, src_mask)

    outputs = torch.LongTensor([[init_tok]]).to(opt.device)

    trg_mask = nopeak_mask(1, opt)

    out = model.linear(model.decoder(outputs, e_output, src_mask, trg_mask))
    out = F.softmax(out, dim=-1)

    probs, ix = out[:, -1].data.topk(opt.k)
    log_scores = torch.Tensor([math.log(prob) for prob in probs.data[0]]).unsqueeze(0)

    outputs = torch.zeros(opt.k, opt.max_len).long().to(opt.device)
    outputs[:, 0] = init_tok
    outputs[:, 1] = ix[0]

    e_outputs = torch.zeros(opt.k, e_output.size(-2), e_output.size(-1)).to(opt.device)
    e_outputs[:, :] = e_output[0]

    eos_tok = opt.trg_eos
    src_mask = (src != opt.src_pad).unsqueeze(-2)
    ind = None
    for i in range(2, opt.max_len):

        trg_mask = nopeak_mask(i, opt)

        out = model.linear(model.decoder(outputs[:, :i],
                                         e_outputs, src_mask, trg_mask))

        out = F.softmax(out, dim=-1)

        outputs, log_scores = k_best_outputs(outputs, out, log_scores, i, opt.k)

        ones = torch.nonzero(outputs == eos_tok)
        # ones = (outputs==eos_tok).nonzero() # Occurrences of end symbols for all input sentences.
        sentence_lengths = torch.zeros(len(outputs), dtype=torch.long).to(opt.device)
        for vec in ones:
            i = vec[0]
            if sentence_lengths[i] == 0:  # First end symbol has not been found yet
                sentence_lengths[i] = vec[1]  # Position of first end symbol

        num_finished_sentences = len([s for s in sentence_lengths if s > 0])

        if num_finished_sentences == opt.k:
            alpha = 0.7
            div = 1 / (sentence_lengths.type_as(log_scores) ** alpha)
            _, ind = torch.max(log_scores * div, 1)
            ind = ind.data[0]
            break

    if ind is None:
        length = (outputs[0] == eos_tok).nonzero()[0]
        sentence_list = (outputs[0][1:length]).tolist()
        return ''.join(opt.trg_processor.decode(sentence_list)).replace('_', " ")

    else:
        length = (outputs[ind] == eos_tok).nonzero()[0]
        sentence_list = (outputs[0][1:length]).tolist()
        return ''.join(opt.trg_processor.decode(sentence_list)).replace('_', " ")


def multiple_replace(dict, text):
    # Create a regular expression  from the dictionary keys
    regex = re.compile("(%s)" % "|".join(map(re.escape, dict.keys())))

    # For each match, look-up corresponding value in dictionary
    return regex.sub(lambda mo: dict[mo.string[mo.start():mo.end()]], text)


def translate_sentence(sentence, model, opt):
    model.eval()
    sentence = Variable(torch.LongTensor([opt.src_processor.encode(sentence)])).to(opt.device)

    sentence = beam_search(sentence, model, opt)

    return multiple_replace({' ?': '?', ' !': '!', ' .': '.', '\' ': '\'', ' ,': ','}, sentence)






