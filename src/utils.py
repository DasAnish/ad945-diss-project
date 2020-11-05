import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


class WordEmbedder(nn.Module):

    def __init__(self, size_of_vocab, embedding_dim):
        super().__init__()
        self.embedder = nn.Embedding(size_of_vocab, embedding_dim)

    def forward(self, x):
        return self.embedder(x)


class PositionalEncoding(nn.Module):

    def __init__(self, embedding_dim, max_length):
        super().__init__()
        self.embedding_dim = embedding_dim

        position_vector = torch.zeros(max_length, embedding_dim, requires_grad=False)
        arange = torch.arange(max_length)

        for i in range(0, embedding_dim, 2):
            position_vector[:, i] = torch.sin(arange / 10000 ** ((2*i) / embedding_dim))
            position_vector[:, i+1] = torch.cos(arange / 10000 ** ((2*i+1) / embedding_dim))

        position_vector.unsqueeze(0)
        self.register_buffer('position_vector', position_vector)

    def forward(self, x):
        x = x * torch.sqrt(self.embedding_dim)
        sequence_length = x.size(1)

        x = x + Variable(self.position_vector[:, :sequence_length], requires_grad=False)



