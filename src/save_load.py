import torch
import torch.nn as nn
from src.TransformerLayers import Transformer
import os


class Save:
    def __init__(self, model_state_dict=None, optim_state_dict=None):
        self.model_state_dict = model_state_dict
        self.optim_state_dict = optim_state_dict

    def save(self, filename):
        torch.save(self.model_state_dict, f'{filename}.model')
        torch.save(self.optim_state_dict, f'{filename}.optim')

    def load(self, filename):

        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')

        self.model_state_dict = torch.load(f'{filename}.model', map_location=device)
        self.optim_state_dict = torch.load(f'{filename}.optim', map_location=device)


def load_model(opt):

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
            if opt.model_prefix not in file or '.model' not in file: continue

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