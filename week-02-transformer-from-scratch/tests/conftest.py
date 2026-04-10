import pytest
import torch

@pytest.fixture
def input_data():
    batch_size = 2
    seq_len = 5
    emb_dim = 12
    x = torch.randn(batch_size, seq_len, emb_dim)
    y = torch.randint(0, 4, (batch_size,))
    mask = torch.randint(0, 2, (batch_size, seq_len)).bool()
    return x, y, mask


