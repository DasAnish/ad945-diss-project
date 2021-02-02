import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

Tensor = torch.Tensor
from typing import Optional


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim: int, heads: int, dropout: float = 0.1):
        super().__init__()

        if torch.cuda.is_available():
            dev = "cuda:0"
        else:
            dev = "cpu"

        device = torch.device(dev)

        self.model_dim: int = model_dim
        self.heads: int = heads
        self.d_k: int = model_dim // heads
        self.dropout: nn.Dropout = nn.Dropout(dropout).to(device)
        self.add_module('dropout', self.dropout)

        self.w_key: nn.Linear = nn.Linear(model_dim, model_dim).to(device)
        self.w_query: nn.Linear = nn.Linear(model_dim, model_dim).to(device)
        self.w_value: nn.Linear = nn.Linear(model_dim, model_dim).to(device)
        self.add_module('w_key', self.w_key)
        self.add_module('w_query', self.w_query)
        self.add_module('w_value', self.w_value)

        self.final_layer: nn.Linear = nn.Linear(model_dim, model_dim).to(device)
        self.add_module('final_layer', self.final_layer)

    def forward(self,
                q: Tensor,
                k: Tensor,
                v: Tensor,
                mask: Optional[Tensor] = None):

        # if encoder_output is not None: print("enc_output == True")
        # print(f'x.shape: {x.shape}')

        batch_size = q.size(0)
        queries = self.w_query(q).view(batch_size, -1, self.heads, self.d_k)
        keys = self.w_key(k).view(batch_size, -1, self.heads, self.d_k)
        values = self.w_value(v).view(batch_size, -1, self.heads, self.d_k)

        # we have batch_size, seq_len, heads, d_k tensor

        keys.transpose_(1, 2)
        queries.transpose_(1, 2)
        values.transpose_(1, 2)

        interim_result = torch.matmul(queries, keys.transpose(-2, -1)) / sqrt(self.d_k)

        if mask is not None:
            mask = mask.unsqueeze(1)
            interim_result = interim_result.masked_fill(mask == 0, -1e9)

        interim_result = F.softmax(interim_result, dim=-1)

        if self.dropout is not None:
            interim_result = self.dropout(interim_result)

        interim_result = torch.matmul(interim_result, values)

        concatenated_result = interim_result.transpose(1, 2).contiguous().view(batch_size, -1, self.model_dim)

        final_result = self.final_layer(concatenated_result)

        return final_result
