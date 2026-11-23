# LLM Engineering Journey

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![GitHub last commit](https://img.shields.io/github/last-commit/vaadewoyin/llm-engineering-journey)](https://github.com/vaadewoyin/llm-engineering-journey)

> Hands-on LLM engineering from first principles to production.  
> Built with PyTorch, Hugging Face, Unsloth, and modern LLMOps tools.

---

## 📌 Overview

This repository documents a structured, project-based journey into LLM engineering, focusing on:

- **Foundations:** PyTorch training loops, attention mechanisms, and transformers from scratch.
- **Fine-tuning:** Supervised fine-tuning, LoRA, QLoRA, and parameter-efficient training.
- **Retrieval & Evaluation:** RAG pipelines, vector databases, hybrid search, reranking, and rigorous benchmarking.
- **Agents & Production:** Agentic workflows, MCP, model serving, observability, CI/CD, and deployment.

Each module has its own numbered folder containing implementations, experiments, evaluations, and technical documentation.

---

## 📦 Supporting Artifacts

| Artifact | Purpose |
|---|---|
| `shared/` | Reusable training and evaluation utilities, configurations, and dataset helpers. |
| `concrete-papers-data-pipeline/` | Research-paper processing and QA dataset generation for concrete-materials fine-tuning experiments. |

---

## 📊 Projects

### Completed Modules (01–07)

| Module | Project | Key Skills & Tools |
|---|---|---|
| 01 | MLP Trainer | PyTorch training loops, Typer CLI, reproducibility, CoverType classification. |
| 02 | Transformer From Scratch | Scaled dot-product attention, multi-head attention, positional encoding, sequence classification. |
| 03 | LLM Inference CLI | Hugging Face pipelines, streaming generation, model comparison, inference benchmarking. |
| 04 | ArXiv QA Dataset | Synthetic QA generation, quality filtering, dataset preparation, experiment tracking. |
| 05 | SFT Experiment | Qwen2.5-1.5B, LoRA fine-tuning, learning-rate comparison, qualitative evaluation. |
| 06 | LoRA Rank Analysis | Rank sweep (r=8, 16, 32, 64), perplexity comparison, efficiency analysis. |
| 07 | Efficient Fine-Tuning Analysis | Qwen3-1.7B, Qwen3-8B, LoRA, QLoRA, memory profiling, comparative evaluation. |

### Current Roadmap (08–14)

| Module | Project | Key Skills & Tools | Status |
|---|---|---|---|
| 08 | Alignment & Preference Optimization | DPO, preference learning, alignment evaluation | Deferred |
| 09 | Retrieval-Augmented Generation System | Embeddings, chunking, Qdrant, retrieval, source grounding | In Progress |
| 10 | RAG Evaluation & Benchmarking | BM25, hybrid search, reranking, RAGAS, bootstrap confidence intervals | Planned |
| 11 | LLM Inference & Serving | FastAPI, asynchronous Python, Docker, latency and load testing | Planned |
| 12 | Agentic Workflows & MCP | LangGraph, MCP tools, deterministic routing, guardrails, evaluation | Planned |
| 13 | Production Deployment & MLOps | Modal, GitHub Actions, observability, cost controls, deployment | Planned |
| 14 | Portfolio & Applications | Reproducible demonstrations, technical documentation, portfolio preparation | Planned |

*Current development focuses on an integrated research-grounded RAG and agentic system for sustainable and alternative concrete materials.*

---

## 🛠️ Tools & Technologies

| Category | Tools |
|---|---|
| Core | Python 3.12+, PyTorch, Hugging Face Transformers, Datasets |
| Fine-tuning | Unsloth,  Comet ML |
| RAG & Evaluation | Qdrant, BM25, RAGAS, pytest |
| Agents | LangGraph, MCP |
| Serving & Deployment | FastAPI, Docker, Modal, GitHub Actions |
| Observability | Opik, Comet ML |
| Development | Git, GitHub, VS Code, uv |

*Some technologies are scheduled for upcoming modules.*

---

## 🚀 Quick Start

```bash
git clone https://github.com/vaadewoyin/llm-engineering-journey.git
cd llm-engineering-journey

# Each module has its own implementation and setup instructions.
# Example:
cd 01-mlp-trainer
```