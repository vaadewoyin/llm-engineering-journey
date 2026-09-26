import torch
from transformer.encoder import TransformerEncoder
from transformer.positional import PositionalEncoding

def test_encoder_block_output(input_data):
    """ Tests that encoder block output matches input shape"""
    x, _, mask = input_data
    batch_size, seq_len, emb_dim = x.shape
    num_heads = 2
    mask = mask.clone()
    mask[:, -1] = False
    transformer_encoder = TransformerEncoder(emb_dim=emb_dim, num_heads=num_heads, dropout=0.2,
                                             ffn_hidden_dim=10, num_layers= 3)
    outputs = transformer_encoder(x, key_padding_mask = mask)

    assert outputs.shape == x.shape
    assert not torch.isnan(outputs).any()


def test_positional_encoding_diffrent_position(input_data):
    """ Tests that different positonal vectors are different"""
    x, _, _ = input_data  # x shape -> (2, 5, 12)
    positional_encoding = PositionalEncoding(context_len=5, emb_dim=12)
    pos_encoded_input = positional_encoding(x)
    assert not torch.allclose(pos_encoded_input[:, 0, :], pos_encoded_input[:, 1, :],
                              atol=1e-6)
