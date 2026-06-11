"""
Environment readiness test for fine-tuning.
Checks: CUDA, Unsloth model loading, HF dataset, checkpoint dir, Comet ML.

Requires: .env file with COMET_API_KEY. 
Run directly: uv run test_env.py
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import torch
import comet_ml
from datasets import load_dataset
from unsloth import FastLanguageModel

load_dotenv()

# Test funcs
def check_cuda():
    assert torch.cuda.is_available(), "CUDA not available"
    print(f"GPU: {torch.cuda.get_device_name()}")
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"Memory: {mem:.1f} GB")

def check_unsloth():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/tinyllama-bnb-4bit",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    assert model is not None
    assert tokenizer is not None
    print("Model and tokenizer loaded OK")
    
def check_hf_dataset_loading():
    data = load_dataset("vaadewoyin/arxiv-ml-qa-dataset")
    print(data["train"][0])

def check_checkpoint_dir():
    output_dir = Path("output/checkpoint/")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {"epoch": 10, "loss": 5}
    path = output_dir / f"checkpoint_{checkpoint['epoch']}.pt"
    torch.save(checkpoint, path)
    path.unlink()  
    print("Checkpoint dir is okay")

def check_comet_ml():
    comet_ml_api_key = os.getenv("COMET_API_KEY")
    assert comet_ml_api_key, "COMET_API_KEY not set"
    experiment = comet_ml.Experiment(api_key=comet_ml_api_key, 
                                     project_name="test")
    experiment.end()
    print("Comet ML initialised OK")

if __name__ == "__main__":
    check_cuda()
    check_unsloth()
    check_hf_dataset_loading()
    check_checkpoint_dir()
    check_comet_ml()
    print("\n✅ All environment checks passed!")