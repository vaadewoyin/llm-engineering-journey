# LLM Engineering Journey

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![GitHub last commit](https://img.shields.io/github/last-commit/vaadewoyin/llm-engineering-journey)](https://github.com/vaadewoyin/llm-engineering-journey)

> Hands‑on LLM engineering from first principles to production.  
> Built with PyTorch, Hugging Face, Unsloth, and modern MLOps / LLMOps tools.

---

## 📌 Overview

This repository documents a structured **13‑week journey** into LLM engineering, focusing on:

- **Foundations**: PyTorch training loops, attention mechanisms, transformers from scratch.
- **Fine‑tuning & Alignment**: LoRA, QLoRA, SFT, DPO, preference learning.
- **Retrieval & Agents**: RAG pipelines, vector databases, agentic workflows.
- **Production & MLOps**: Model serving, observability, CI/CD, containerization.

Each week has its own folder with a complete project – code, experiments, logs, and post‑mortems.

---

## 📊 Projects (weeks 1–5)

| Week | Project                  | Key Skills & Tools                                                                 |
| :--- | :----------------------- | :--------------------------------------------------------------------------------- |
| 1    | MLP Trainer              | PyTorch training loop, Typer CLI, reproducibility, 94% accuracy on CoverType.      |
| 2    | Transformer From Scratch | Scaled dot‑product attention, multi‑head, positional encoding, sentence classification. |
| 3    | LLM Inference CLI        | Hugging Face pipelines, streaming generation, `--compare` flag, token efficiency benchmark. |
| 4    | ArXiv QA Dataset         | Synthetic dataset generation: ArXiv API, Unsloth + Llama‑3‑8B (4‑bit), ChatML, Comet ML & Opik, quality filtering, Hugging Face Hub. |
| 5    | SFT on ArXiv QA          | LoRA‑based Supervised Fine‑Tuning (Qwen2.5‑1.5B), learning rate comparison (1e‑4 vs 3e‑4), Comet ML logging, checkpointing, qualitative rubric evaluation, POSTMORTEM. |

*More weeks will be added as the journey progresses (LoRA rank analysis, QLoRA, DPO, RAG, agentic systems).*

---

## 🛠️ Tools & Technologies

| Category          | Tools                                                                 |
| :---------------- | :-------------------------------------------------------------------- |
| **Core**          | Python 3.12+, PyTorch, Hugging Face Transformers, Datasets            |
| **Fine‑tuning**   | Unsloth, TRL, PEFT, bitsandbytes, Comet ML, Opik                      |
| **RAG & Agents**  | *Coming soon* – Qdrant, LangGraph, RAGAS, Gradio                      |
| **Serving**       | *Coming soon* – vLLM, FastAPI, Docker                                 |
| **Quality & CI**  | Ruff, pre‑commit, pytest, uv (package manager), Kaggle/Colab GPU      |

---

## 🚀 Quick Start

```bash
git clone https://github.com/vaadewoyin/llm-engineering-journey.git
cd llm-engineering-journey
# Each week folder contains its own pyproject.toml and README with specific instructions.