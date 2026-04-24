# LLM Inference CLI

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Transformers-yellow)](https://huggingface.co/docs/transformers/index)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

> A command-line tool for running and comparing instruction-tuned LLMs locally.

## Problem
Running and comparing instruction-tuned LLMs generally involves writing boilerplate code that loads models, generates text, and compares model outputs and various metrics (like Tokens/sec). This project helps devs run and compare models fast right from the CLI, enabling rapid experimentation and saving them from that boilerplate.

## Architecture

```
cli.py → calls config.py, generate.py, and compare.py
compare.py → calls generate.py for each model
```

- **config.py**: contains configuration used for text generation.  
- **generate.py**: contains `GenerationEngine` class for generating text  
- **compare.py**: contains function for comparing two models and their outputs and handles memory cleanup (deleting the model and forcing garbage collection) between model loads.
- **cli.py**: contains the Typer app and command definitions; calls generate.py and compare.py. 

## Features
- `llm-cli generate` — generate text from any instruction-tuned Hugging Face model  
- `llm-cli compare` — side-by-side comparison of two models with a rich table  
- Streaming and non-streaming output  
- Config file support (JSON)  
- Parameter validation before model loading  

## Key Decisions
- Configuration from the default dataclass, a JSON file, or CLI arguments all pass through the same `__post_init__` to avoid invalid inputs.  
- `ModelNotInstructionTunedError` is raised when the model lacks a chat template.
- A wrong model ID results in a clean RuntimeError message, not a traceback.
- For model comparison, after loading one model, model output is stored and memory is cleared before loading another model to avoid OOM errors and to allow working with two models in a limited memory environment.  
- If `config_file` is provided, load config from file and ignore other CLI args (with a warning). If the config file is not provided, use CLI args if provided, otherwise use defaults.  

For the full design rationale — component communication, failure modes, limitations, and production boundaries (deterministic core, human inspection points, etc.) — see **[DESIGN.md](./DESIGN.md)**.


## Results 

I compared `HuggingFaceTB/SmolLM2-135M-Instruct` and `Qwen/Qwen2-0.5B-Instruct` on the prompt:

> Explain fine-tuning in machine learning in one sentence.

### Sample Outputs

- **SmolLM2-135M-Instruct:**  
  In machine learning, fine-tuning refers to adjusting parameters...

- **Qwen2-0.5B-Instruct:**  
  Fine-tuning involves adjusting hyperparameters of an existing model...

### Performance Comparison

| Model                                | Tokens Generated | Generation Time (s) | Tokens/sec |
|:-------------------------------------|-----------------:|--------------------:|-----------:|
| HuggingFaceTB/SmolLM2-135M-Instruct  |               56 |                7.89 |       7.10 |
| Qwen/Qwen2-0.5B-Instruct             |               24 |                5.70 |       4.21 |

### Observation
SmolLM2-135M-Instruct generated more tokens at a higher throughput, while Qwen2-0.5B-Instruct produced shorter output with lower tokens/sec.

## Key Insights
- Loading two models at once in a limited-memory environment triggers an OOM error. This is prevented by loading models sequentially, storing the outputs, and clearing each model before loading the next.
- Swapping message order in chat templates leads to degraded outputs. The model's instruction‑following behavior depends on seeing the system prompt first, before the user prompt. It is important to enforce this order deterministically, without giving end users the ability to alter it.

## Project structure

```
week-03-llm-inference-cli/
├── README.md
├── DESIGN.md
├── pyproject.toml
├── configs/
│   └── default_config.json
├── src/
│   └── llm_cli/
│       ├── __init__.py
│       ├── cli.py           # Typer app, run and compare commands
│       ├── config.py        # GenConfig dataclass + validation
│       ├── generate.py      # Model loading, generation, streaming
│       └── compare.py       # Two-model comparison + memory handling
├── tests/
│   ├── test_cli.py
│   └── test_config.py
```
## Known Limitations
- The system does not compare more than two models at a time.
- Very large models cause an OOM error (but the error is caught and shown as a clean message, not a traceback).
- Gated repos without proper authentication trigger an OSError.

## Usage 
> install uv first (make sure it is installed before running the CLI)

```bash

# Clone and enter
git clone https://github.com/vaadewoyin/llm-engineering-journey.git
cd llm-engineering-journey/week-03-llm-inference-cli

# Install dependencies & sync
uv sync

# To generate text with single model with default config
uv run llm-cli generate

# To compare two models (example)
uv run llm-cli compare --model-id1 "HuggingFaceTB/SmolLM2-135M-Instruct" --model-id2 "Qwen/Qwen2-0.5B-Instruct" --prompt "Explain fine‑tuning in machine learning in one sentence."
```

