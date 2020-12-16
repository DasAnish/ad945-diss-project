import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Variable
from math import sqrt


# class WordEmbeddings(nn.Module):
#
#     def __init__(self, size_of_vocab: int, model_dim: int):
#         super().__init__()
#         self.embedder: nn.Embedding = nn.Embedding(size_of_vocab, model_dim)
#
#     def forward(self, x: Tensor) -> Tensor:
#         return self.embedder(x)


class PositionalEncoding(nn.Module):

    '''A parameter less module that concatenates a number of sine signals at the end of the embedded vectors.'''

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
        for i in range(0, model_dim, 2):
            # Follwing the formula provided in the paper.
            position_vector[:, i] = torch.sin(arange / (10000 ** ((2*i) / model_dim)))
            position_vector[:, i+1] = torch.cos(arange / (10000 ** ((2*i+1) / model_dim)))

        position_vector.unsqueeze(0)
        self.register_buffer('position_vector', position_vector)

    def forward(self, x: Tensor) -> Tensor:
        """
        The forward pass implementation of the Positional Embedding step.
        :param x: the tensor containing Batch x Seq_len x model_dim embeddings.
        :return: the embedded vector with some alterations.
        """

        x = x * sqrt(self.model_dim)
        sequence_length = x.size(1)

        x = x + Variable(self.position_vector[:sequence_length, :], requires_grad=False).expand_as(x)
        return x


class Norm(nn.Module):

    """The utility module to normalize the outputs."""

    def __init__(self, model_dim: int, eps: float = 1e-5):
        super().__init__()

        if torch.cuda.is_available():
            dev = "cuda:0"
        else:
            dev = "cpu"

        device = torch.device(dev)

        self.model_dim = model_dim
        self.eps = eps

        # two learnable parameters to get better normalizations
        self.alpha: Tensor = nn.Parameter(torch.ones(self.model_dim)).to(device)
        self.bias: Tensor = nn.Parameter(torch.zeros(self.model_dim)).to(device)

    def forward(self, x: Tensor) -> Tensor:
        """
        The forward pass algorithm to nomralize the input. Uses two torch.Parameters which are learnt over time.
        Those are: alpha (initially 1s) and bias (initially 0s).
        :param x: The tensor we want to normalize w.r.t. dim=-1
        :return: The normalized output alpha * (x - mean(x)) / (std(x)+eps) + bias
        """
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias

        return norm


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

    def forward(self, x: Tensor) -> Tensor:

        """
        Implements Feed-Forward algorithm with dropout.
        :param x: a tensor with last dim = model_dim
        :return: output from the NN
        """

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)

        return x


