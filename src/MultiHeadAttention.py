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

    def forward(self, x: Tensor,
                mask: Optional[Tensor] = None,
                encoder_output: Optional[Tensor] = None):

        # if encoder_output is not None: print("enc_output == True")
        # print(f'x.shape: {x.shape}')

        batch_size = x.size(0)
        queries = self.w_query(x).view(batch_size, -1, self.heads, self.d_k)

        if encoder_output is None:
            keys = self.w_key(x).view(batch_size, -1, self.heads, self.d_k)
            values = self.w_value(x).view(batch_size, -1, self.heads, self.d_k)
        else:
            keys = self.w_key(encoder_output).view(batch_size, -1, self.heads, self.d_k)
            values = self.w_value(encoder_output).view(batch_size, -1, self.heads, self.d_k)
        # print(f'keys.shape: {keys.shape}, \nqueries.shape: {queries.shape}, \nvalues.shape: {values.shape}')

        # we have batch_size, seq_len, heads, d_k tensor

        keys.transpose_(1, 2)
        queries.transpose_(1, 2)
        values.transpose_(1, 2)

        # after transposing we have batch_size, heads, seq_len, d_k tensor
        # if encoder_output is not None: print(queries, keys.transpose(-2, -1).shape)

        # print(f'queries.shape: {queries.shape}, \nkeys.shape: {keys.shape}, \nvalues.shape: {values.shape}')

        interim_result = torch.matmul(queries, keys.transpose(-2, -1)) / sqrt(self.d_k)

        # print(f'interim result.shape: {interim_result.shape}')

        if mask is not None:
            # print(f'mask_0.shape: {mask.shape}')
            mask = mask.unsqueeze(1)
            # print(f'mask_1.shape: {mask.shape}')
            # if encoder_output is None: mask = mask.unsqueeze(3)
            # print(f'mask_2.shape: {mask.shape}')

            # print(f'{interim_result.shape}: {interim_result}\n************************************\n')
            interim_result = interim_result.masked_fill(mask == 0, -1e9)
            # print(f'{interim_result.shape}: {interim_result}\n*************************************\n')

        interim_result = F.softmax(interim_result, dim=-1)

        if self.dropout is not None:
            interim_result = self.dropout(interim_result)

        interim_result = torch.matmul(interim_result, values)
        # print(f"shape after matmul values: {interim_result.shape}")

        concatenated_result = interim_result.transpose(1, 2).contiguous().view(batch_size, -1, self.model_dim)

        final_result = self.final_layer(concatenated_result)
        # print(f"final_result.shape: {final_result.shape}\n")

        return final_result
