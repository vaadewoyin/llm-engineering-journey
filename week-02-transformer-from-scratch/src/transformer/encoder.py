# Imports
import torch
import torch.nn as nn
from attention import MultiHeadAttention
from typing import Optional


#TransformerEncoderBlock
class TransformerEncoderBlock(nn.Module):
    """
    Transformer Encoder block
    Args:
        emb_dim (int): The dimensionality of the input embeddings.
        num_heads (int): The number of attention heads.
        dropout (float): The dropout probability.
        ffn_hidden_dim (int) : Feed forward network hidden size
    """
    def __init__(self, emb_dim: int, num_heads: int, dropout: float, ffn_hidden_dim):
        """
        Initialises the Transformer encoder block
        Args:
            emb_dim (int): The dimensionality of the input embeddings.
            num_heads (int): The number of attention heads.
            dropout (float): The dropout probability.
            ffn_hidden_dim (int) : Feed forward network hidden size
        """
        super().__init__()
        # Multi-head self attention
        self.self_attn = MultiHeadAttention(emb_dim, num_heads, dropout)
        # Layer norms
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(emb_dim, ffn_hidden_dim),
            nn.ReLU(),
            nn.Linear(ffn_hidden_dim, emb_dim))
        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]=None,
                 key_padding_mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        """
        Forward pass for the transformer encoder block.
        Args:
            x (torch.Tensor): Input tensor.
            attn_mask (torch.Tensor, optional): Attention mask. Defaults to None.
            key_padding_mask (torch.Tensor, optional): Key padding mask. Defaults to None.
        Returns:
            torch.Tensor: Output tensor.
        """
        # Multi-head self attention
        attn_output, _ = self.self_attn(x, x, x, attn_mask=attn_mask,
                                        key_padding_mask=key_padding_mask)
        attn_output = self.dropout(attn_output)
        # Add & Norm
        x_norm = self.norm1(x + attn_output)
        # Feed-forward network
        ffn_output = self.feed_forward(x_norm)
        ffn_output = self.dropout(ffn_output)
        return self.norm2(x_norm + ffn_output)

# TransformerEncoder
class TransformerEncoder(nn.Module):
    """
    This class stacks multiple encoder blocks to create the full transformer encoder.
    Args:
        emb_dim (int): The dimensionality of the input embeddings.
        num_heads (int): The number of attention heads.
        dropout (float): The dropout probability.
        ffn_hidden_dim (int) : Feed forward network hidden size
        num_layers (int) : Number of encoder blocks to stack
    """
    def __init__(self, emb_dim: int, num_heads: int, dropout: float,
                 ffn_hidden_dim: int, num_layers: int):
        """
        Initialises the transformer encoder
        """
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(emb_dim, num_heads, dropout, ffn_hidden_dim) for _ in range (num_layers)
            ])

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]=None,
                 key_padding_mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        """
        Forward pass for the transformer encoder.
        Returns the output of the last encoder block (shape: (batch_size, seq_len, emb_dim))
        """
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        return x

