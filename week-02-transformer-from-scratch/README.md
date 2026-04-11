# Transformer from scratch – AG News classification

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Transformers-yellow)](https://huggingface.co/docs/transformers/index)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

> A transformer encoder built from scratch and trained on AG News (4 classes). Achieves ~91% test accuracy.

## Problem

News articles on diverse topics like sports, finance, tech and entertainment flood the internet daily. Automatic classification at scale requires a model that understands not just keywords, but relationships between words in each sentence.

The goal of this project was not to maximise accuracy but rather to build a transformer encoder with a classification head from scratch to classify the AG News dataset. The dataset, which has news articles with 4 categories (World, Sports, Business, Sci/Tech), is a popular benchmark dataset for sentence classification. The model was built from scratch in order to understand the intricacies of the transformer architecture, such as attention, positional encoding, the various parts of the encoder, and the full training pipeline.

## Architecture

- **Input**: Token IDs → Embedding (`vocab_size=30522`, `emb_dim=128`) + Positional Encoding (learnable)
- **Encoder**: 4 layers, 4 heads, `emb_dim=128`, `ffn_hidden_dim=512`
- **Classifier**: `[CLS]` token → Linear(128, 4)
- **Optimizer**: AdamW (lr=1e-3, weight_decay=0.1)
- **Scheduler**: ReduceLROnPlateau (patience=2, factor=0.5)
- **Regularisation**: Dropout=0.3, weight decay=0.1
- **Other details**: `batch_size=32`, `context_length=128` (fixed)

## Results and comparison

| Model                                 | Test Accuracy | Parameters | Training data                       |
| ------------------------------------- | ------------- | ---------- | ----------------------------------- |
| Transformer classifier (from scratch) | **90.6%**     | ~4.7M      | AG News only                        |
| BERT‑base (fine‑tuned, typical)       | ~94.5%\*      | 110M       | pre‑trained + AG News (fine‑tuning) |

\*Benchmark based on publicly available fine-tuned BERT models on AG News. Included for reference only.

## Ablation

| Model variant                               | Val accuracy (15 epochs) |
| ------------------------------------------- | ------------------------ |
| Full model                                  | 91.7%                    |
| No positional encoding (removed completely) | 90.4%                    |
| Single attention head                       | 91.5%                    |
| No layer norm                               | 89.6%                    |

## Key insights

- **Removing LayerNorm had the highest impact** on this dataset (-2.1%). Normalisation helps maintain stable gradients through the network. Without it, gradients become unstable as they propagate through the encoder blocks. LayerNorm is a necessity in stacked transformer architectures.

- **Positional encoding contributed only around -1.3% when removed.** Positional encoding helps introduce a sense of positional order for each token in the sequence, which greatly helps the model learn token representations well. However, in this case, the sequences are short and their category is largely defined by vocabulary, so removing positional encoding didn't affect the model too much. On larger documents, positional encoding has a significant effect.

- **The model started overfitting at around 6 epochs** despite having only ~4.7M parameters trained on 102k samples. This is largely due to the fact that we are training the model from scratch on a small dataset. Also, ReduceLROnPlateau didn't help; dropout and weight decay were effective regularisers during training.

## Device Support

The code is hardware-agnostic and runs on both GPU and CPU:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

## Project structure

```

week-02-transformer-from-scratch/
├── src/transformer/
│ ├── __init__.py
│ ├── attention.py
│ ├── positional.py
│ ├── encoder.py
│ ├── classifier.py
│ ├── data.py
│ └── train.py
├── tests/
│ ├── conftest.py
│ ├── test_attention.py
│ └── test_encoder_positional.py
├── outputs/ # generated plots (gitignored)
├── configs/ # optional hyperparameter configs
├── pyproject.toml
├── uv.lock
└── README.md

```

## Usage

```bash
# Clone and enter
git clone https://github.com/vaadewoyin/llm-engineering-journey.git
cd llm-engineering-journey/week-02-transformer-from-scratch

# Install dependencies & sync
uv sync

# Train
uv run python src/transformer/train.py

# Run tests
uv run pytest -v tests/

```
