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
