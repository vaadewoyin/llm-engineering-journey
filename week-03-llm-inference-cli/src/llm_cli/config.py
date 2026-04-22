# Imports 
import json
from dataclasses import dataclass

# Config dataclass
@dataclass
class GenConfig:
    model_id: str = "HuggingFaceTB/SmolLM2-135M-Instruct" #why: SmolLM2 for quick experimentation
    system_prompt: str = (
        "You are a helpful assistant. Provide a direct, final answer. "
        "Do NOT ask any questions, request clarification, or suggest follow‑up actions. "
        "End your response after giving the answer."
    )
    streamer: bool = True
    do_sample: bool = True
    temperature: float= 0.7
    top_p: float= 0.8
    max_new_tokens: int = 100
    repetition_penalty: float = 1.25

def load_gen_config(file_path: str) -> GenConfig:
    with open(file_path, "r") as f:
        data = json.load(f)
    return GenConfig(**data) #why: unpacking the json to GenConfig fields for consistency
         
# Check that config can be saved and loaded properly
if __name__ == "__main__":
    cfg = load_gen_config("configs/default_config.json")
    print(cfg)