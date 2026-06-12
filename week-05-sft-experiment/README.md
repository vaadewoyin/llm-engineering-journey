# SFT on ArXiv ML Q&A – Learning Rate Comparison

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Unsloth](https://img.shields.io/badge/🤗%20Unsloth-LoRA-orange)](https://github.com/unslothai/unsloth)
[![Comet ML](https://img.shields.io/badge/Comet%20ML-Experiment%20Tracking-purple)](https://www.comet.com)

> LoRA‑based Supervised Fine‑Tuning (SFT) on 951 Q&A pairs from ArXiv ML abstracts. Comparison of two learning rates (1e‑4 vs 3e‑4) using Qwen2.5‑1.5B.

## Overview

Supervised Fine‑Tuning (SFT) teaches a base model how to respond in a structured way. This project fine‑tunes a small language model on ArXiv ML Q&A data to improve its ability to answer technical questions. The experiment compares two learning rates and evaluates the resulting models using perplexity and a manual qualitative rubric.

**Key components:**
- `main.py` – main training script using Unsloth + TRL SFTTrainer (LoRA, 4‑bit) + evaluation (perplexity answer generation)
- `tests/test_environment.py` – pre‑training environment validation
- `configs/baseline_eval_config.json` – locked hyperparameters
- `eval/qualitative_rubric.md` – scoring sheet for manual evaluation

## Key Decisions

- **LoRA instead of full fine‑tuning** – Saves memory on T4 GPU (16 GB). Only adapters are trained (0.22% of parameters).
- **Learning rates** – 1e‑4 vs 3e‑4  to observe training stability.
- **Checkpointing** – every 100 steps; `load_best_model_at_end=True` ensures best checkpoint is saved.
- **Packing disabled** – each example is a separate conversation to avoid cross‑contamination.

For the full design rationale (failure modes, production boundaries, hypothesis), see **[DESIGN.md](./DESIGN.md)**.

## Results

| Learning Rate | Final Eval Loss | Perplexity (full‑seq) | Average Qualitative Score (max 16) |
|---------------|----------------|------------------------|--------------------------------------|
| 1e‑4          | ~1.72          | ~5.54                   | 6.4                                  |
| 3e‑4          | ~1.8           | ~5.55                   | 6.3                                  |

Both learning rates produced similar loss curves and qualitative scores (average 6.4/16 for 1e-4, 6.3/16 for 3e-4). The model outputs are generally readable (fluency 2-3) with partial relevance (1-3) but scored 1/4 on factual correctness and groundedness across all evaluated examples. This observation is a strong indication that the abstract‑only dataset is insufficient for deep reasoning

## Qualitative Rubric (locked)

Four dimensions scored 1–4:
- **Factual correctness** – Is the answer factually accurate?
- **Relevance** – Does it answer the exact question?
- **Fluency** – Is the text coherent and natural?
- **Groundedness** – Are claims traceable to ML content?

See [`eval/qualitative_rubric.md`](eval/qualitative_rubric.md) for full level definitions.

## Project Structure

```
week-05-sft/
├── configs/
│   └── baseline_eval_config.json
├── eval/
│   └── qualitative_rubric.md
├── src/
│   ├── main.py
├── tests/
│   └── test_environment.py
├── outputs/                 # checkpoints and final models
├── DESIGN.md
├── POSTMORTEM.md
├── README.md
└── pyproject.toml
```

## Known Limitations

- **Abstract‑only data** – Abstracts lack methodology, experiments, and trade‑offs, limiting the depth of answers.
- **Small model** – Qwen2.5‑1.5B has limited capacity for complex reasoning.
- **No reasoning traces** – The dataset contains direct Q&A without `<think>` reasoning steps.
- **Manual scoring** – Qualitative evaluation is subjective, but the rubric keeps it consistent.

## Next Steps

- Build enriched dataset from full paper sections (introduction, methodology, conclusion).
- Add reasoning traces (`<think>...</think>`) for Thinking SFT.
- Repeat fine‑tuning with larger model 

