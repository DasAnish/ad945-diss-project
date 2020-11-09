import torch
import torch.nn as nn
import torch.nn.functional as F
from src.MultiHeadAttention import MultiHeadAttention
from src.utils import *


class TransformerEncoderLayer(nn.Module):
    def __init__(self, model_dim, heads, d_ff, dropout=0.1):
        super().__init__()

        self.model_dim = model_dim
        self.heads = heads
        self.dropout = dropout

        self.self_attn = MultiHeadAttention(model_dim, heads, dropout)

        self.attn_dropout = nn.Dropout(dropout)
        self.ff_dropout = nn.Dropout(dropout)

        self.ff = FeedForward(model_dim, d_ff, dropout)

        self.norm_after_attn = Norm(model_dim)
        self.final_norm = Norm(model_dim)

    def forward(self, x, mask=None):
        x += self.attn_dropout(self.self_attn(x, mask))
        y = self.norm_after_attn(x)
        x += self.ff(y)
        y = self.norm_final(x)
        return y


class TransformerEncoder(nn.Module):
    def __init__(self, model_dim, d_ff, heads, num_blocks, vocab_size, max_seq_len=80, dropout=0.1):
        super().__init__()

        self.model_dim = model_dim
        self.heads = heads
        self.num_blocks = num_blocks
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        self.word_embeddings = WordEmbeddings(vocab_size, model_dim)
        self.positional_encoding = PositionalEncoding(model_dim, max_seq_len)

        encoding_layer = TransformerEncoderLayer(model_dim, heads, d_ff, dropout=dropout)
        self.encoding_layers = [encoding_layer.deepcopy() for _ in range(num_blocks)]

    def forward(self, _input, mask):
        x = self.word_embeddings(_input)
        x = self.positional_encoding(x)
        for layer in self.encoding_layers:
            x = layer(x, mask)

        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, model_dim, heads, d_ff, dropout=0.1):
        super().__init__()

        self.model_dim = model_dim
        self.heads = heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.norm_before = False

        self.norm_after_self_attn = Norm(model_dim)
        self.norm_after_encdec_attn = Norm(model_dim)
        self.norm_after_ff = Norm(model_dim)

        self.dropout_self_attn = nn.Dropout(dropout)
        self.dropout_encdec_attn = nn.Dropout(dropout)
        self.dropout_ff = nn.Dropout(dropout)

        self.self_attn = MultiHeadAttention(model_dim, heads, dropout)
        self.enc_dec_attn = MultiHeadAttention(model_dim, heads, dropout)
        self.ff = FeedForward(model_dim, d_ff)

    def forward(self, x, encoder_output, src_mask, trg_mask):
        x += self.dropout_self_attn(self.self_attn(x, trg_mask))
        y = self.norm_after_self_attn(x)

        # TODO: understand and replace this.
        x += self.dropout_encdec_attn(self.enc_dec_attn(y, src_mask, encoder_output))
        y = self.norm_after_encdec_attn(x)

        x += self.dropout_ff(self.ff(y))
        y = self.norm_after_ff(x)

        return y


class TransformerDecoder(nn.Module):
    def __init__(self, model_dim, d_ff, heads, num_blocks, vocab_size, max_seq_len=80, dropout=0.1):
        super().__init__()

        self.model_dim = model_dim
        self.heads = heads
        self.num_blocks = num_blocks
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.dropout = dropout

        self.word_embeddings = WordEmbeddings(vocab_size, model_dim)
        self.positional_embeddings = PositionalEncoding(model_dim, max_seq_len)

        decoding_layer = TransformerDecoderLayer(model_dim, heads, d_ff, dropout=dropout)
        self.decoding_layers = [decoding_layer.deepcopy() for _ in range(num_blocks)]

    def forward(self, target, encoder_output, src_mask, trg_mask):
        x = self.word_embeddings(target)
        x = self.positional_embeddings(x)

        for layer in self.decoding_layers:
            x = layer(x, encoder_output, src_mask, trg_mask)

        return x


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size,
                 model_dim, d_ff, heads, num_blocks, max_seq_len=80, dropout=0.1):
        super().__init__()

        self.encoder = TransformerEncoder(model_dim, d_ff, heads, num_blocks,
                                          src_vocab_size, max_seq_len, dropout)
        self.decoder = TransformerDecoder(model_dim, d_ff, heads, num_blocks,
                                          trg_vocab_size, max_seq_len, dropout)

        self.linear = nn.Linear(model_dim, trg_vocab_size)

    def forward(self, src, trg, src_mask, trg_mask):
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(trg, enc_output, src_mask, trg_mask)
        output = self.linear(dec_output)
        return output

