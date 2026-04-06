# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# MultiHeadAttention
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout):
        super().__init__()
        assert emb_dim % num_heads == 0 , "Emb_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads

        self.wq = nn.Linear(emb_dim, emb_dim)
        self.wk = nn.Linear(emb_dim, emb_dim)
        self.wv = nn.Linear(emb_dim, emb_dim)
        self.proj = nn.Linear(emb_dim, emb_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
      batch_size, Lq, emb_dim = query.shape
      _, Lk, _ = key.shape

      Q = self.wq(query)
      K = self.wk(key)
      V = self.wv(value)

      Q = Q.view(batch_size, Lq, self.num_heads, self.head_dim).transpose(1, 2)
      K = K.view(batch_size, Lk, self.num_heads, self.head_dim).transpose(1, 2)
      V = V.view(batch_size, Lk, self.num_heads, self.head_dim).transpose(1, 2)

      attention_score = Q @ K.transpose(2, 3) / (self.head_dim ** 0.5)

      if attn_mask is not None:
          attention_score = attention_score.masked_fill(attn_mask, -torch.inf)

      if key_padding_mask is not None:
          mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
          attention_score = attention_score.masked_fill(mask, -torch.inf)

      weights = F.softmax(attention_score, dim=-1)
      outputs = self.dropout(weights) @ V

      outputs = outputs.transpose(1, 2).contiguous().view(batch_size, Lq, emb_dim)
      outputs = self.proj(outputs)

      return outputs, weights

