import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


class WordEmbeddings(nn.Module):

    def __init__(self, size_of_vocab, model_dim):
        super().__init__()
        self.embedder = nn.Embedding(size_of_vocab, model_dim)

    def forward(self, x):
        return self.embedder(x)


class PositionalEncoding(nn.Module):

    def __init__(self, model_dim, max_length):
        super().__init__()
        self.model_dim = model_dim

        position_vector = torch.zeros(max_length, model_dim, requires_grad=False)
        arange = torch.arange(max_length)

        for i in range(0, model_dim, 2):
            position_vector[:, i] = torch.sin(arange / 10000 ** ((2*i) / model_dim))
            position_vector[:, i+1] = torch.cos(arange / 10000 ** ((2*i+1) / model_dim))

        position_vector.unsqueeze(0)
        self.register_buffer('position_vector', position_vector)

    def forward(self, x):
        x = x * torch.sqrt(self.model_dim)
        sequence_length = x.size(1)

        x = x + Variable(self.position_vector[:, :sequence_length], requires_grad=False)


class Norm(nn.Module):
    def __init__(self, model_dim, eps = 1e-5):
        super().__init__()

        self.model_dim = model_dim
        self.eps = eps

        # two learnable parameters to get better normalizations
        self.alpha = nn.Parameter(torch.ones(self.model_dim))
        self.bias = nn.Parameter(torch.zeros(self.model_dim))

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias

        return norm


class FeedForward(nn.Module):
    def __init__(self, model_dim, d_ff = 2048, dropout=0.1):
        super().__init__()

        self.fc1 = nn.Linear(model_dim, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_ff, model_dim)

    def forward(self, x):

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)

        return x


