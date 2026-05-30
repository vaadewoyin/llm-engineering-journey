# ArXiv ML Q&A Dataset

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Hugging Face Datasets](https://img.shields.io/badge/🤗%20Datasets-951%20pairs-yellow)](https://huggingface.co/datasets/vaadewoyin/arxiv-ml-qa-dataset)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-unsloth-orange)](https://github.com/unslothai/unsloth)

> A dataset of 951 high-quality Q&A pairs generated from ArXiv machine learning paper abstracts. 

## Overview
Fine-tuning language models require the use of high-quality dataset for optimal performance, this projects generates high quality Q&A pairs from ArXiv machine learning paper abstracts. 

- `ArXiv API Scraper` — Fetches paper metadata (title, abstract, ID, category) using arXiv API 
- `Cleaning Pipeline` — Deduplicates by paper ID, filters abstracts by word count (min 50 words, max 400), saves cleaned JSONL.
- `Q&A Generator` — Uses Unsloth + Llama-3-8B-Instruct (4‑bit) on Kaggle to generate two synthetic Q&A pairs per abstract in ChatML format.
- `Quality filter` — Validates ChatML structure, checks question (≥10 words) and answer (≥20 words) lengths, logs rejection rate via Comet ML.

## Key Decisions
- **Prompt strategy:** Synthetic, reasoning‑based Q&A generation – see [`configs/prompt.txt`](configs/prompt.txt).
- A 4‑bit quantised Llama‑3‑8B‑Instruct model was used for the Q&A Generator to accommodate Kaggle GPU constraints.

For the full design rationale — failure modes, limitations, and production boundaries (deterministic core, human inspection points, etc.) — see **[DESIGN.md](./DESIGN.md)**.

### Sample Q&A pair 

**Question:**  
In comparison to Gemini 3.0 Pro and Claude Opus 4.5, what advantages does the GPT-5.2-powered reviewing agent bring to the table?

**Answer:**  
GPT-5.2 outperforms its counterparts because it exhibits better contextual understanding, allowing it to identify complex relationships between different aspects of a paper. Moreover, its capacity to learn from vast amounts of data enables it to recognize subtle patterns and nuances that might elude other AI reviewers.

*(Stored in ChatML format: `{"messages": [{"role": "user", "content": "...", "role": "assistant", "content": "..."}]}`)*

### Results
Total raw pairs generated: 1044

Valid pairs after filtering: 951

Rejection rate: ~8.9%

## Project structure
```
week-04-arxiv-qa-dataset/
├── configs/
│ ├── pipeline_config.json
│ └── prompt.txt
├── notebook/
│ └── qa-generator.ipynb
├── src/
│ ├── init.py
│ ├── scraper.py
│ ├── cleaner.py
│ ├── generator.py
│ └── quality_filter.py
├── tests/
│ ├── test_cleaner.py
│ └── test_quality_filter.py
├── .env.example
├── .gitignore
├── .python-version
├── DESIGN.md
├── LICENSE
├── POSTMORTEM.md
├── README.md
├── push_to_hub.py
├── pyproject.toml
└── uv.lock
```

## Known Limitations
- No automated groundedness verification: Paper IDs were not stored alongside Q&A pairs, making it impossible to test that each answer was fully grounded in the corresponding abstracts.
- No language filtering for non-English papers – The source abstracts are from arXiv cs.LG (mostly English). No language filtering was done, as non-English papers are rare.
- Model quality variance – The generator uses a 4‑bit quantised Llama‑3‑8B‑Instruct model, which may occasionally hallucinate, produce malformed JSON, or answers that are too short.
- Synthetic bias – The Q&A pairs reflect the style & knowledge of the model (Llama‑3‑8B‑Instruct model).

## Usage

To replicate the dataset generation, run the scripts in the following order:

```bash
uv run python src/scraper.py
uv run python src/cleaner.py
# Generator requires a GPU (run on Kaggle, Colab, or local GPU) – see notebook/qa-generator.ipynb
uv run python src/quality_filter.py
```

The final filtered Q&A pairs are saved to outputs/final/qa_pairs.jsonl.