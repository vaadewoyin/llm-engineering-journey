# Imports
import torch
import torch.nn as nn

# Positional Encoding
class PositionalEncoding(nn.Module):
    """
    Learnable position encoding layer for the transformer model
    Args:
        context_len (int): maximum context length (sequence length) for which to learn positional embeddings.
        emb_dim (int): dimensionality of the embeddings.
    """
    def __init__(self, context_len: int, emb_dim: int) -> None:
        """
        Initialize the position encoding layer.

        Args:
            context_len (int): maximum context length.
            emb_dim (int): embedding dimension.
        """
        super().__init__()
        self.context_len = context_len
        self.emb_dim = emb_dim
        self.pos_embedding_layer = nn.Embedding(self.context_len, self.emb_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Forward pass for the position encoding layer.
        Args:
            x (torch.Tensor): input tensor of shape (batch_size, seq_len, emb_dim)

        Returns:
            torch.Tensor: positionally encoded tensor of shape (batch_size, seq_len, emb_dim)
        """
        pos_embedding = self.pos_embedding_layer(torch.arange(x.size(1), device=x.device))
        return x + pos_embedding

