import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
Tensor = torch.Tensor
from typing import Optional


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim: int, heads: int, dropout: float = 0.1):
        super().__init__()

        self.model_dim: int = model_dim
        self.heads: int = heads
        self.d_k: int = model_dim // heads
        self.dropout: nn.Dropout = nn.Dropout(dropout)

        self.w_key: nn.Linear = nn.Linear(model_dim, model_dim)
        self.w_query: nn.Linear = nn.Linear(model_dim, model_dim)
        self.w_value: nn.Linear = nn.Linear(model_dim, model_dim)

        self.final_layer: nn.Linear = nn.Linear(model_dim, model_dim)

    def forward(self, x: Tensor,
                mask: Optional[Tensor] = None,
                encoder_output: Optional[Tensor]=None):

        # X has the shape batch_size, seq_len, model_dim

        batch_size = x.size(0)
        keys = self.w_key(x).view(batch_size, -1, self.heads, self.d_k)
        if encoder_output is None:
            queries = self.w_query(x).view(batch_size, -1, self.heads, self.d_k)
            values = self.w_value(x).view(batch_size, -1, self.heads, self.d_k)
        else:
            queries = self.w_query(encoder_output).view(batch_size, -1, self.heads, self.d_k)
            values = self.w_value(encoder_output).view(batch_size, -1, self.heads, self.d_k)

        # we have batch_size, seq_len, heads, d_k tensor

        keys.transpose_(1, 2)
        queries.transpose_(1, 2)
        values.transpose_(1, 2)

        # after transposing we have batch_size, heads, seq_len, d_k tensor

        interim_result = torch.matmul(queries, keys.transpose(-2, -1)) / sqrt(self.d_k)

        if mask is not None:
            mask = mask.unsqueeze(1)
            mask = mask.unsqueeze(3)
            interim_result = interim_result.masked_fill(mask == 0, -1e9)

        interim_result = F.softmax(interim_result, dim=-1)

        if self.dropout is not None:
            interim_result = self.dropout(interim_result)

        # if mask is not None: print(f"interim_result.shape: {interim_result.shape} | values.shape: {values.shape}")
        interim_result = torch.matmul(interim_result, values)

        concatenated_result = interim_result.transpose(1, 2).contiguous().view(batch_size, -1, self.model_dim)

        final_result = self.final_layer(concatenated_result)

        return final_result



