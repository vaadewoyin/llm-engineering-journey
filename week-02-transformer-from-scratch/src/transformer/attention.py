# Imports
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# MultiHeadAttention
class MultiHeadAttention(nn.Module):
    """
    Multi-head attention layer for the transformer model.
    Args:
        emb_dim (int): The dimensionality of the input embeddings.
        num_heads (int): The number of attention heads.
        dropout (float): The dropout probability.
    """
    def __init__(self, emb_dim: int, num_heads: int, dropout: float) -> None:
        """
        Initialises the multi-head attention layer
        Raises:
            ValueError: if emb_dim is not divisible by num_heads.
        """
        super().__init__()
        if emb_dim % num_heads != 0 :
            raise ValueError ("emb_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        self.wq = nn.Linear(emb_dim, emb_dim)
        self.wk = nn.Linear(emb_dim, emb_dim)
        self.wv = nn.Linear(emb_dim, emb_dim)
        self.proj = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the multi-head attention layer.
        Args:
            query (torch.Tensor): input query tensor of shape (batch_size, seq_len, emb_dim)
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The output tensor of shape (batch_size, seq_len, emb_dim) and attention weights.
        """
        batch_size, Lq, emb_dim = query.shape
        _, Lk, _ = key.shape
        # Linear projections
        Q = self.wq(query)
        K = self.wk(key)
        V = self.wv(value)
        # Reshape and transpose for multi-head attention
        Q = Q.view(batch_size, Lq, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, Lk, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, Lk, self.num_heads, self.head_dim).transpose(1, 2)
        # Scaled dot-product attention
        attention_score = Q @ K.transpose(2, 3) / (self.head_dim ** 0.5)
        # Apply attention mask if provided
        if attn_mask is not None:
            attention_score = attention_score.masked_fill(attn_mask, -torch.inf)
        # Apply key padding mask if provided
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attention_score = attention_score.masked_fill(mask, -torch.inf)
        # Calculate attention weights and output
        weights = F.softmax(attention_score, dim=-1)
        outputs = self.dropout(weights) @ V
        # Reshape and project output
        outputs = outputs.transpose(1, 2).contiguous().view(batch_size, Lq, emb_dim)
        outputs = self.proj(outputs)
        # Return output and attention weights
        return outputs, weights

