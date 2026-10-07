# Efficient Fine-Tuning Analysis: Full FT vs LoRA vs QLoRA

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Unsloth](https://img.shields.io/badge/🤗%20Unsloth-LoRA-orange)](https://github.com/unslothai/unsloth)
[![Comet ML](https://img.shields.io/badge/Comet%20ML-Experiment%20Tracking-purple)](https://www.comet.com)

> Comparing full fine-tuning, LoRA, and QLoRA on generated Q&A pairs 
## What this project compares

Three fine-tuning methods, same dataset:

| Method | Model | Trainable params |
|---|---|---|
| Full FT | Qwen3-1.7B | 1.41B (82%) — measured |
| LoRA r=16 | Qwen3-1.7B | ~18M — estimated from r × (in+out) |
| QLoRA-NF4 | Qwen3-8B | ~50M — estimated from r × (in+out) |

## Dataset

- **Repo:** [`vaadewoyin/llm-engineering-journey-data`](https://huggingface.co/datasets/vaadewoyin/llm-engineering-journey-data)
- **Train / val:** 2,070 / 468 pairs, split by paper, no paper appears in both splits
- **Token length:** mean 104, p95 158, max 304 → `max_seq_length=512`
- **Domain:** QA pairs generated from concrete papers (see `concrete-papers-data-pipeline/`)

## Results

| Metric | Full FT 1.7B | LoRA r=16 1.7B | QLoRA-NF4 8B |
|---|---|---|---|
| Peak VRAM (allocated) | 8.83 GB | **3.83 GB** | 7.84 GB |
| Peak VRAM (reserved) | 8.91 GB | **3.93 GB** | 8.02 GB |
| Wall-clock | **20.1 min** | 25.5 min | 39.1 min |
| Final train loss | 1.4361 | 1.4730 | **1.2755** |
| Final eval loss | 1.4039 | 1.4020 | **1.3236** |
| Adapter size on disk | - | **66.5 MB** | 166.6 MB |
| Hardware | Modal L4 | Modal L4 | Modal L4 |

## Findings

**1. Base model capacity beats training method.** QLoRA on 8B (only ~50M trainable params, 4-bit base) reached eval loss 1.3236 — clearly better than both 1.7B runs (1.4020, 1.4039). Training ~3% of a 4.7× larger model outperforms training 82% of a small one.

**2. LoRA matches full FT at 2.3× less memory.** 1.4020 vs 1.4039 eval loss — indistinguishable — at 3.83 GB vs 8.83 GB peak. The 66.5 MB adapter is 51× smaller than full weights.

## Project Structure

```
07-efficient-finetuning-analysis/
├── run_full_finetune.py       — full FT of Qwen3-1.7B
├── run_lora.py                — LoRA r=16 on Qwen3-1.7B
├── run_qlora.py               — QLoRA-NF4 on Qwen3-8B
├── outputs/
│   ├── sft/final/             — full FT weights (~3.4 GB)
│   ├── lora_r16/final/        — LoRA adapter (66.5 MB)
│   └── qlora_nf4/final/       — QLoRA adapter (166.6 MB)
├── DESIGN.md                  — problem, components, failure modes, boundaries
├── POSTMORTEM.md              — prediction vs actual, lessons learned
└── README.md                  — this file
```

## To Reproduce This

### 1. Clone and install

```bash
git clone https://github.com/vaadewoyin/llm-engineering-journey.git
cd llm-engineering-journey/shared
uv pip install -e .

cd ../07-efficient-finetuning-analysis
uv sync
```

### 2. Set credentials

Create a `.env` file at the repo root:

```
COMET_API_KEY=your_comet_key
HF_TOKEN=your_hf_token
```

### 3. Run

```bash
uv run python run_full_finetune.py   # ~20 min on L4
uv run python run_lora.py            # ~25 min on L4
uv run python run_qlora.py           # ~39 min on L4
```

Each script:
- Loads the locked eval config (`shared/configs/baseline_eval_config.json`) for the seed
- Loads the dataset from HuggingFace Hub
- Profiles VRAM at model load, trainer init, and peak during training
- Logs to Comet project `concrete-ft-phase` under a distinct run name
- Saves the final model or adapter to `outputs/<run_name>/final/`

## Known Limitations

1. **Sequences longer than 512 tokens.** Training data seq len is capped at 512 and longer examples get truncated.
2. **Models larger than 8B.** Every config is sized to fit 24 GB VRAM on Modal L4. Larger models would need more GPU.

## AI Usage Disclosure

Per the plan's rules, core logic was written by hand. AI was used for:
- **Syntax help** on Unsloth, TRL, and Comet APIs where the documentation was incomplete
- **Boilerplate** — the final print block and Comet logging block
- **Error diagnosis** — reading stack traces for Unsloth / transformers version mismatches
- **README structure and phrasing** — the outline and wording of this document

No core logic (training loop, memory profiling, chat formatting, config loading) was AI-authored. See `DESIGN.md` for the full design rationale.