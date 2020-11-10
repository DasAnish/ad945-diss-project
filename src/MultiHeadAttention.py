import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, heads, dropout=0.1):
        super().__init__()

        self.model_dim = model_dim
        self.heads = heads
        self.d_k = model_dim // heads
        self.dropout = nn.Dropout(dropout)

        self.w_key = nn.Linear(model_dim, model_dim)
        self.w_query = nn.Linear(model_dim, model_dim)
        self.w_value = nn.Linear(model_dim, model_dim)

        self.final_layer = nn.Linear(model_dim, model_dim)

    def forward(self, x, mask = None, encoder_output=None):
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

        # TODO: add the mask operation
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



