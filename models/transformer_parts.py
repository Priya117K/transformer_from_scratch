import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    """
    Injects position information into the token embeddings using sine and cosine functions.
    """
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create a matrix of shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Calculate division term for sinusoidal frequencies
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register buffer so it's saved with the model state dict but not treated as a parameter
        self.register_buffer('pe', pe.unsqueeze(0)) # Shape: (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self/Cross-Attention mechanism.
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear layers for Query, Key, Value projections
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        # 1. Linear projection & reshape for multi-head: (B, SeqLen, H, d_k) -> transpose to (B, H, SeqLen, d_k)
        q = self.q_linear(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 2. Scaled Dot-Product Attention: (Q @ K^T) / sqrt(d_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            # Fill padded or future positions with very low numbers (-1e9) so softmax zeroes them out
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 3. Multiply by Value: (B, H, SeqLen, SeqLen) @ (B, H, SeqLen, d_k) -> (B, H, SeqLen, d_k)
        output = torch.matmul(attn_weights, v)

        # 4. Concatenate heads back together and pass through final linear layer
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.out_linear(output), attn_weights

class PositionWiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network: Linear -> ReLU -> Dropout -> Linear
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, d_model)
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class ResidualConnection(nn.Module):
    """
    A residual connection followed by a layer norm: x + Sublayer(LayerNorm(x))
    """
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer) -> torch.Tensor:
        # Pre-LN Transformer style (Normalization is applied *before* the sublayer)
        return x + self.dropout(sublayer(self.norm(x)))
class EncoderLayer(nn.Module):
    """
    A single Encoder block containing Self-Attention and Feed-Forward Network.
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)
        
        self.residual_1 = ResidualConnection(d_model, dropout)
        self.residual_2 = ResidualConnection(d_model, dropout)

    def forward(self, x, src_mask):
        # Sublayer 1: Multi-Head Self Attention with Residual Connection
        x = self.residual_1(x, lambda _x: self.self_attn(_x, _x, _x, src_mask)[0])
        # Sublayer 2: Feed-Forward Network with Residual Connection
        x = self.residual_2(x, self.ffn)
        return x


class DecoderLayer(nn.Module):
    """
    A single Decoder block containing Masked Self-Attention, Encoder-Decoder Cross-Attention, and FFN.
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)
        
        self.residual_1 = ResidualConnection(d_model, dropout)
        self.residual_2 = ResidualConnection(d_model, dropout)
        self.residual_3 = ResidualConnection(d_model, dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        # Sublayer 1: Masked Self-Attention (prevents looking into the future)
        x = self.residual_1(x, lambda _x: self.self_attn(_x, _x, _x, tgt_mask)[0])
        
        # Sublayer 2: Cross-Attention (Queries from decoder, Keys/Values from encoder output)
        x = self.residual_2(x, lambda _x: self.cross_attn(_x, enc_output, enc_output, src_mask)[0])
        
        # Sublayer 3: Feed-Forward Network
        x = self.residual_3(x, self.ffn)
        return x