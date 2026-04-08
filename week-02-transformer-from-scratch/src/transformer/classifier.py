# Imports
import torch
import torch.nn as nn
from positional import PositionalEncoding
from encoder import TransformerEncoder

class TransformerClassifier(nn.Module):
    """
    Transformer-based classifier model for sequence classification
    Args:
        vocab size (int): Vocabulary size for tokenization
        context_len (int): maximum context length (sequence length) for which to learn positional embeddings.
        emb_dim (int): dimensionality of the embeddings.
        num_heads (int): number of attention heads in the transformer encoder.
        dropout (float): dropout rate for the transformer encoder.
        ffn_hidden_dim (int): hidden dimension for the feedforward network in the transformer encoder.
        num_layers (int): number of layers in the transformer encoder.
        num_outputs (int): number of output classes for classification.
    """
    def __init__(self, vocab_size: int, context_len: int, emb_dim: int, num_heads:int,
                 dropout:float, ffn_hidden_dim: int, num_layers: int, num_outputs: int) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, emb_dim)
        self.positional_encoding = PositionalEncoding(context_len=context_len, emb_dim=emb_dim)
        self.transformer_encoder = TransformerEncoder(emb_dim, num_heads, dropout,
                                                      ffn_hidden_dim, num_layers)
        self.classifier = nn.Linear(emb_dim, num_outputs)

    def forward(self, x, attn_mask=None, key_padding_mask=None) -> torch.Tensor:
        """Forward pass for the transformer classifier model.
        Args:
            x (torch.Tensor): input tensor of shape (batch_size, seq_len)
            attn_mask (torch.Tensor, optional): attention mask for the transformer encoder.
            key_padding_mask (torch.Tensor, optional): key padding mask for the transformer encoder.
        Returns:
            torch.Tensor: output logits of shape (batch_size, num_outputs)
        """
        # Token embedding
        token_embedded_input = self.token_embedding(x)
        positional_encoded_input = self.positional_encoding(token_embedded_input)
        transformer_encoder_output = self.transformer_encoder(positional_encoded_input, attn_mask,
                                                              key_padding_mask)
        cls_token = transformer_encoder_output[:, 0, :]
        logits = self.classifier(cls_token)
        return logits

