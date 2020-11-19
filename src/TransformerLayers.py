import torch
import torch.nn as nn
import torch.nn.functional as F
from src.MultiHeadAttention import MultiHeadAttention
from src.utils import *
from typing import List, Optional

"""In this file, everything has been put together to build the Transformer architecture."""


class TransformerEncoderLayer(nn.Module):
    def __init__(self, model_dim: int, heads: int, d_ff: int,
                 dropout: Optional[float] = 0.1, norm_before: bool = False):
        super().__init__()

        self.model_dim: int = model_dim
        self.heads: int = heads
        self.dropout: float = dropout
        self.norm_before: bool = norm_before

        self.self_attn: MultiHeadAttention = MultiHeadAttention(model_dim, heads, dropout)

        self.attn_dropout: nn.Dropout = nn.Dropout(dropout)
        self.ffn_dropout: nn.Dropout = nn.Dropout(dropout)

        self.ffn: FeedForward = FeedForward(model_dim, d_ff, dropout)

        self.attn_norm: Norm = Norm(model_dim)
        self.ffn_norm: Norm = Norm(model_dim)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        y = x
        if self.norm_before:
            y = self.attn_norm(x)
        y = self.self_attn(y, mask)
        y = self.attn_dropout(y)
        x = x + y
        if not self.norm_before:
            x = self.attn_norm(x)

        y = x
        if self.norm_before:
            y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        if not self.norm_before:
            x = self.ffn_norm(x)

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, model_dim: int, d_ff: int, heads: int, num_blocks: int,
                 vocab_size: int, max_seq_len: Optional[int] = 80,
                 dropout: Optional[float] = 0.1, norm_before: bool = False):
        super().__init__()

        self.model_dim: int = model_dim
        self.heads: int = heads
        self.num_blocks: int = num_blocks
        self.vocab_size: int = vocab_size
        self.max_seq_len: int = max_seq_len

        self.word_embeddings: nn.Embedding = nn.Embedding(vocab_size, model_dim)
        self.positional_encoding: PositionalEncoding = PositionalEncoding(model_dim, max_seq_len)

        encoding_layer: TransformerEncoderLayer = TransformerEncoderLayer(model_dim,
                                                                          heads,
                                                                          d_ff,
                                                                          dropout=dropout,
                                                                          norm_before=norm_before)
        self.encoding_layers: List[TransformerEncoderLayer] = [encoding_layer.deepcopy()
                                                               for _ in range(num_blocks)]

    def forward(self, _input: Tensor, mask: Tensor):
        x = self.word_embeddings(_input)
        x = self.positional_encoding(x)
        for layer in self.encoding_layers:
            x = layer(x, mask)

        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, model_dim: int, heads: int, d_ff: int,
                 dropout: Optional[float] = 0.1, norm_before: bool = False):
        super().__init__()

        self.model_dim: int = model_dim
        self.heads: int = heads
        self.d_ff: int = d_ff
        self.dropout: float = dropout
        self.norm_before: bool = norm_before

        self.self_attn_norm: Norm = Norm(model_dim)
        self.enc_dec_norm: Norm = Norm(model_dim)
        self.ffn_norm: Norm = Norm(model_dim)

        self.self_attn_dropout: nn.Dropout = nn.Dropout(dropout)
        self.enc_dec_dropout: nn.Dropout = nn.Dropout(dropout)
        self.ffn_dropout: nn.Dropout = nn.Dropout(dropout)

        self.self_attn: MultiHeadAttention = MultiHeadAttention(model_dim, heads, dropout)
        self.enc_dec_attn: MultiHeadAttention = MultiHeadAttention(model_dim, heads, dropout)
        self.ffn: FeedForward = FeedForward(model_dim, d_ff)

    def forward(self, x: Tensor, encoder_output: Tensor, src_mask: Tensor, trg_mask: Tensor) -> Tensor:

        y = x
        if self.norm_before:
            y = self.self_attn_norm(x)
        y = self.self_attn(y)
        y = self.self_attn_dropout(y)
        x = x + y
        if not self.norm_before:
            x = self.self_attn_norm(x)

        y = x
        if self.norm_before:
            y = self.enc_dec_norm(x)
        y = self.enc_dec_attn(y, src_mask, encoder_output)
        y = self.enc_dec_dropout(y)
        x = x + y
        if not self.norm_before:
            x = self.enc_dec_norm(x)

        y = x
        if self.norm_before:
            y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        if not self.norm_before:
            x = self.ffn_norm(x)

        return x

        # x += self.dropout_self_attn(self.self_attn(x, trg_mask))
        # y = self.norm_after_self_attn(x)
        #
        # # TODO: understand and replace this.
        # x += self.dropout_encdec_attn(self.enc_dec_attn(y, src_mask, encoder_output))
        # y = self.norm_after_encdec_attn(x)
        #
        # x += self.dropout_ff(self.ff(y))
        # y = self.norm_after_ff(x)
        #
        # return y


class TransformerDecoder(nn.Module):
    def __init__(self, model_dim: int, d_ff: int, heads: int, num_blocks: int,
                 vocab_size: int, max_seq_len: Optional[int] = 80,
                 dropout: Optional[float] = 0.1, norm_before: bool = False):
        super().__init__()

        self.model_dim: int = model_dim
        self.heads: int = heads
        self.num_blocks: int = num_blocks
        self.vocab_size: int = vocab_size
        self.max_seq_len: int = max_seq_len
        self.dropout: float = dropout
        self.norm_before: bool = norm_before

        self.word_embeddings: nn.Embedding = nn.Embedding(vocab_size, model_dim)
        self.positional_embeddings: PositionalEncoding = PositionalEncoding(model_dim, max_seq_len)

        decoding_layer: TransformerDecoderLayer = TransformerDecoderLayer(model_dim, heads, d_ff, dropout=dropout)
        self.decoding_layers: List[TransformerDecoderLayer] = [decoding_layer.deepcopy() for _ in range(num_blocks)]

    def forward(self, target: Tensor, encoder_output: Tensor, src_mask: Tensor, trg_mask: Tensor) -> Tensor:
        x: Tensor = self.word_embeddings(target)
        x = self.positional_embeddings(x)

        for layer in self.decoding_layers:
            x = layer(x, encoder_output, src_mask, trg_mask)

        return x


class Transformer(nn.Module):
    def __init__(self, src_vocab_size: int, trg_vocab_size: int,
                 model_dim: int, d_ff: int, heads: int, num_blocks: int,
                 max_seq_len: Optional[int] = 80, dropout: Optional[float] = 0.1,
                 norm_before: bool = False):
        super().__init__()

        self.encoder: TransformerEncoder = TransformerEncoder(model_dim, d_ff, heads, num_blocks,
                                                              src_vocab_size, max_seq_len, dropout,
                                                              norm_before)
        self.decoder: TransformerDecoder = TransformerDecoder(model_dim, d_ff, heads, num_blocks,
                                                              trg_vocab_size, max_seq_len, dropout,
                                                              norm_before)

        self.linear: nn.Linear = nn.Linear(model_dim, trg_vocab_size)

    def forward(self, src: Tensor, trg: Tensor, src_mask: Tensor, trg_mask: Tensor):
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(trg, enc_output, src_mask, trg_mask)
        output = self.linear(dec_output)
        return output

