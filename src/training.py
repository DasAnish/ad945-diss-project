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


class Opt:
    pass


opt = Opt()
opt.src_lang = 'es'
opt.trg_lang = 'en'
opt.num_mil = 1
opt.input_file = f'data/{opt.src_lang}/ParaCrawl.{opt.src_lang}-{opt.trg_lang}.'
opt.model_file = f'data/{opt.src_lang}/SPM-{opt.num_mil}m-8k.{opt.src_lang}-{opt.trg_lang}.'
opt.interim_file = f'data/{opt.src_lang}/ParaCrawl.{opt.src_lang}-{opt.trg_lang}.{opt.num_mil}m.'
opt.dataset = f'data/{opt.src_lang}/tokenized_dataset_{opt.src_lang}_{opt.num_mil}m'
opt.dev_dataset = f'data/{opt.src_lang}/DEV-{opt.src_lang}-{opt.trg_lang}.'
# opt.model_file = 'data/SPM-1m-8k.fr-en.'
opt.max_len = 150
opt.dev_dataset = f'data/{opt.src_lang}/DEV-{opt.src_lang}-{opt.trg_lang}.'


opt.src_data_path = opt.interim_file + opt.src_lang
opt.trg_data_path = opt.interim_file + opt.trg_lang
opt.vocab_size = 8000
opt.tokensize = 2048
opt.print_every = 200
opt.save_every = 5000
opt.epochs = 10
opt.warmup_steps = 16000
opt.keep_training = False

opt.path = f'{opt.src_lang}-en-models'
opt.model_prefix = f'{opt.src_lang}-en-model-'
# optim_file = 'data/optim_state_dict'
opt.model_dim = 512
opt.heads = 8
opt.N = 6
opt.args = (opt.vocab_size, opt.vocab_size,
            opt.model_dim, opt.model_dim*4,
            opt.heads, opt.N, opt.max_len, 0.1, True)
opt.log = log

