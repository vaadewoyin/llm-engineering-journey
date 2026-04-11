import torch
from transformer.attention import MultiHeadAttention

def test_multihead_attention(input_data):
    """ Test that MultiHeadAttention has same shape as input"""
    x, _, mask = input_data
    batch_size, seq_len, emb_dim = x.shape
    num_heads = 2
    attn = MultiHeadAttention(emb_dim=emb_dim, num_heads=num_heads, dropout=0.0)
    outputs, weights = attn(x, x, x, key_padding_mask=mask)

    assert outputs.shape == (batch_size, seq_len, emb_dim)
    assert weights.shape == (batch_size, num_heads, seq_len, seq_len)

def test_padding_mask_weights(input_data):
    """ Tests that attention weight at mask positions are zero. """
    x, _, mask = input_data
    batch_size, seq_len, emb_dim = x.shape
    num_heads = 2
    attn = MultiHeadAttention(emb_dim=emb_dim, num_heads=num_heads, dropout=0.0)
    _, weights = attn(x, x, x, key_padding_mask=mask)

    rows, cols = torch.where(mask)  # row(batch) & col(idx_pos) where mask is trur
    for batch, mask_true_pos in zip(rows, cols):
        # Check that attention weights for the padding cols are zero
        assert torch.allclose(input=weights[batch, :, :, [mask_true_pos]],
                              other=torch.tensor(0.0), atol=1e-8)

    print("Padding mask test passed.")

