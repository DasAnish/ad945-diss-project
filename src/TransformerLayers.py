import torch
import torch.nn as nn
import torch.nn.functional as F
from src.MultiHeadAttention import MultiHeadAttention
from src.utils import *
from typing import List, Optional

"""In this file, everything has been put together to build the Transformer architecture."""


class TransformerEncoderLayer(nn.Module):
    def __init__(self, model_dim: int, heads: int, d_ff: int, dropout: Optional[float] = 0.1):
        super().__init__()

        self.model_dim: int = model_dim
        self.heads: int = heads
        self.dropout: float = dropout

        self.self_attn: MultiHeadAttention = MultiHeadAttention(model_dim, heads, dropout)

        self.attn_dropout: nn.Dropout = nn.Dropout(dropout)
        self.ff_dropout: nn.Dropout = nn.Dropout(dropout)

        self.ff: FeedForward = FeedForward(model_dim, d_ff, dropout)

        self.norm_after_attn: Norm = Norm(model_dim)
        self.final_norm: Norm = Norm(model_dim)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        x += self.attn_dropout(self.self_attn(x, mask))
        y = self.norm_after_attn(x)
        x += self.ff(y)
        y = self.norm_final(x)
        return y


class TransformerEncoder(nn.Module):
    def __init__(self, model_dim: int, d_ff: int, heads: int, num_blocks: int,
                 vocab_size: int, max_seq_len: Optional[int] = 80, dropout: Optional[float] = 0.1):
        super().__init__()

        self.model_dim: int = model_dim
        self.heads: int = heads
        self.num_blocks: int = num_blocks
        self.vocab_size: int = vocab_size
        self.max_seq_len: int = max_seq_len

        self.word_embeddings: nn.Embedding = nn.Embedding(vocab_size, model_dim)
        self.positional_encoding: PositionalEncoding = PositionalEncoding(model_dim, max_seq_len)

        encoding_layer: TransformerEncoderLayer = TransformerEncoderLayer(model_dim, heads, d_ff, dropout=dropout)
        self.encoding_layers: List[TransformerEncoderLayer] = [encoding_layer.deepcopy() for _ in range(num_blocks)]

    def forward(self, _input: Tensor, mask: Tensor):
        x = self.word_embeddings(_input)
        x = self.positional_encoding(x)
        for layer in self.encoding_layers:
            x = layer(x, mask)

        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, model_dim: int, heads: int, d_ff: int, dropout: Optional[float] = 0.1):
        super().__init__()

        self.model_dim: int = model_dim
        self.heads: int = heads
        self.d_ff: int = d_ff
        self.dropout: float = dropout

        self.norm_after_self_attn: Norm = Norm(model_dim)
        self.norm_after_encdec_attn: Norm = Norm(model_dim)
        self.norm_after_ff: Norm = Norm(model_dim)

        self.dropout_self_attn: nn.Dropout = nn.Dropout(dropout)
        self.dropout_encdec_attn: nn.Dropout = nn.Dropout(dropout)
        self.dropout_ff: nn.Dropout = nn.Dropout(dropout)

        self.self_attn: MultiHeadAttention = MultiHeadAttention(model_dim, heads, dropout)
        self.enc_dec_attn: MultiHeadAttention = MultiHeadAttention(model_dim, heads, dropout)
        self.ff: FeedForward = FeedForward(model_dim, d_ff)

    def forward(self, x: Tensor, encoder_output: Tensor, src_mask: Tensor, trg_mask: Tensor) -> Tensor:
        x += self.dropout_self_attn(self.self_attn(x, trg_mask))
        y = self.norm_after_self_attn(x)

        # TODO: understand and replace this.
        x += self.dropout_encdec_attn(self.enc_dec_attn(y, src_mask, encoder_output))
        y = self.norm_after_encdec_attn(x)

        x += self.dropout_ff(self.ff(y))
        y = self.norm_after_ff(x)

        return y


class TransformerDecoder(nn.Module):
    def __init__(self, model_dim: int, d_ff: int, heads: int, num_blocks: int,
                 vocab_size: int, max_seq_len: Optional[int] = 80, dropout: Optional[float] = 0.1):
        super().__init__()

        self.model_dim: int = model_dim
        self.heads: int = heads
        self.num_blocks: int = num_blocks
        self.vocab_size: int = vocab_size
        self.max_seq_len: int = max_seq_len
        self.dropout: float = dropout

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
                 max_seq_len: Optional[int] = 80, dropout: Optional[float] = 0.1):
        super().__init__()

        self.encoder: TransformerEncoder = TransformerEncoder(model_dim, d_ff, heads, num_blocks,
                                                              src_vocab_size, max_seq_len, dropout)
        self.decoder: TransformerDecoder = TransformerDecoder(model_dim, d_ff, heads, num_blocks,
                                                              trg_vocab_size, max_seq_len, dropout)

        self.linear: nn.Linear = nn.Linear(model_dim, trg_vocab_size)

    def forward(self, src: Tensor, trg: Tensor, src_mask: Tensor, trg_mask: Tensor):
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(trg, enc_output, src_mask, trg_mask)
        output = self.linear(dec_output)
        return output

