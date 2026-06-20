# LoRA Rank Sweep – Week 6

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Unsloth](https://img.shields.io/badge/🤗%20Unsloth-LoRA-orange)](https://github.com/unslothai/unsloth)
[![Comet ML](https://img.shields.io/badge/Comet%20ML-Experiment%20Tracking-purple)](https://www.comet.com)

> LoRA rank analysis on Qwen2.5‑1.5B using filtered 703 Q&A pairs from ArXiv ML abstracts. Evaluated ranks 8, 16, 32, 64 to find the inflection point.

## Overview

Performance of Supervised Fine‑Tuning (SFT) with LoRA depends largely on the rank (`r`) of the adapter matrices. Higher rank gives more capacity but there is the risk of overfitting, especially on small datasets. This project sweeps LoRA ranks (8, 16, 32, 64) on a 1.5B model to identify the inflection point where increasing rank stops improving perplexity. All other hyperparameters are fixed (learning rate 2e‑4, 3 epochs, filtered 703‑pair dataset).

**Key components:**
- `main.py` – main script that loops over ranks, trains, logs to Comet ML and computes perplexity.
- `tests/test_environment.py` – pre‑training environment validation.
- `config/baseline_eval_config.json` – locked hyperparameters 
- `eval/qualitative_rubric.md` – locked rubric for evaluating the best‑performing rank (reused from Week 5).

## Key Decisions

- **Rank values** – r=8, 16, 32, 64 
- **Model** – Qwen2.5‑1.5B (fits T4 & trains fast).
- **Dataset** – filtered 703 pairs (removed pairs with author/study references in original data for cleaner signal).
- **Learning rate** – fixed at 2e‑4 
- **Epochs** – 3
- **Checkpointing** – every 50 steps; `load_best_model_at_end=True`.
- **Packing disabled** – each example is a separate conversation to avoid cross‑contamination.

For the full design rationale (failure modes, production boundaries, hypothesis), see **[DESIGN.md](./DESIGN.md)**.

## Results

| Rank | Best Eval Loss | Perplexity | Peak Memory (GB) | Training Time (min) |
|------|----------------|------------|------------------|---------------------|
| 8    | 1.8355         | 6.21       | 1.88             | 8.38                |
| 16   | **1.8085**     | **6.17**   | 2.42             | 8.35                |
| 32   | 1.8101         | 6.17       | 2.96             | 8.57                |
| 64   | 1.8303         | 6.30       | 3.71             | 9.27                |

### Inflection Point

**r=16** is the optimal rank. Perplexity drops from 6.21 (r=8) to 6.17 (r=16), flattens at r=32 (6.17), and increases at r=64 (6.30). Higher ranks consume significantly more memory with no quality gain.

*My hypothesis was that r=8 would win, based on the assumption that the small dataset (703 pairs) would make higher ranks prone to overfitting. The actual result showed that r=16 performed better, with r=8 performing worse — suggesting that r=8 lacked the capacity to fully capture data patterns.*

### Qualitative Evaluation (r=16)

The best‑performing rank (r=16) was evaluated on 10 held‑out examples using the locked rubric. The average total score was **6.2/16**, similar with Week 5 results (6.4/16). This confirms that the abstract‑only dataset remains the primary bottleneck for factual correctness and groundedness.

## Qualitative Rubric (locked)

Four dimensions scored 1–4:
- **Factual correctness** – Is the answer factually accurate?
- **Relevance** – Does it answer the exact question?
- **Fluency** – Is the text coherent and natural?
- **Groundedness** – Are claims traceable to ML content?

See [`eval/qualitative_rubric.md`](eval/qualitative_rubric.md) for full level definitions.

## Project Structure

week-06-lora-rank-analysis/
├── config/
│   └── baseline_eval_config.json          # locked hyperparameters (reused from Week 5)
├── eval/
│   └── qualitative_rubric.md              # locked rubric (reused from Week 5)
├── scripts/
│   └── main.py                             # main training script
├── tests/
│   ├── test_environment.py                # environment validation
├── outputs/                               # generated during run (not committed)
│   ├── rank-8/
│   │   └── final/                         # saved model + adapter
│   ├── rank-16/
│   │   └── final/
│   ├── rank-32/
│   │   └── final/
│   └── rank-64/
│       └── final/
├── DESIGN.md                              # design document
├── POSTMORTEM.md                          # post-experiment analysis
├── README.md                              # project overview and results
├── pyproject.toml                         # dependencies and project metadata
└── uv.lock                                # pinned dependencies (if using uv)
