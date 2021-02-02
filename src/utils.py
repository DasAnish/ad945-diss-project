import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Variable
from math import sqrt
from datetime import datetime


class PositionalEncoding(nn.Module):

    """A parameter less module that concatenates a number of sine signals at the end of the embedded vectors."""

    def __init__(self, model_dim: int, max_length: int):
        super().__init__()

        if torch.cuda.is_available():
            dev = "cuda:0"
        else:
            dev = "cpu"

        device = torch.device(dev)

        self.model_dim: int = model_dim

        position_vector: Tensor = torch.zeros(max_length, model_dim, requires_grad=False).to(device)
        arange = torch.arange(max_length)

        # note to self: this appear to work right now.
        for i in range(model_dim//2):
            # Follwing the formula provided in the paper.
            position_vector[:, 2*i] = torch.sin(arange / (10000 ** (2*i / model_dim)))
            position_vector[:, 2*i+1] = torch.cos(arange / (10000 ** (2*i / model_dim)))

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

    def print(self, txt, type=LOG):
        print(txt)
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




