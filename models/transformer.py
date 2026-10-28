import math  # <-- Add this line at the top
import torch
import torch.nn as nn

import torch
import torch.nn as nn
from models.transformer_parts import (
    PositionalEncoding, 
    EncoderLayer, 
    DecoderLayer
)

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_len, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_len, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        import math
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return self.norm(x)

class Transformer(nn.Module):
    """
    Complete Encoder-Decoder Transformer Architecture (~75M Parameters configured)
    """
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=768, num_layers=8, 
                 num_heads=12, d_ff=3072, max_len=512, dropout=0.1):
        super().__init__()
        
        self.encoder = TransformerEncoder(src_vocab_size, d_model, num_layers, num_heads, d_ff, max_len, dropout)
        self.decoder = TransformerDecoder(tgt_vocab_size, d_model, num_layers, num_heads, d_ff, max_len, dropout)
        self.projection = nn.Linear(d_model, tgt_vocab_size)

    def encode(self, src, src_mask):
        return self.encoder(src, src_mask)

    def decode(self, tgt, enc_output, src_mask, tgt_mask):
        return self.decoder(tgt, enc_output, src_mask, tgt_mask)

    def project(self, x):
        return self.projection(x)

