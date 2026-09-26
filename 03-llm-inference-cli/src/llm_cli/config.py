"""Generation configuration and validation.

Created GenConfig (from either default, CLI args, or JSON file) 
pass through the same dataclass __post_init__ validation. This is
the deterministic gate that keeps invalid parameters out of the
generation engine.
"""

import json
from dataclasses import dataclass

@dataclass
class GenConfig:
    # Why: SmolLM2-135M is small enough for quick experimentation
    model_id: str = "HuggingFaceTB/SmolLM2-135M-Instruct" 
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

    def __post_init__(self):
        """Validate fields after the dataclass is initialised."""
        if self.temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {self.temperature}")
        if not (0 < self.top_p <= 1):
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if self.max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be >= 1, got {self.max_new_tokens}")
        if self.repetition_penalty < 1.0:
            raise ValueError(f"repetition_penalty must be >= 1.0, got {self.repetition_penalty}")



def load_gen_config(file_path: str) -> GenConfig:
    """ Loads json config file and returns a GenConfig object. 
    Raises exceptions for file not found or invalid JSON."""
    try:
        with open(file_path, "r") as f:
            data = json.load(f)

    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {file_path}")
    
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config file: {e}")
    # Why: Using **data to unpack the dictionary into GenConfig fields, 
    # relying on GenConfig's __post_init__ for validation. 
    # This ensures that any config loaded from a file is subject to the same validation rules 
    # as configs created from CLI args or defaults.
    return GenConfig(**data)  
         


# Check config loading and validation
if __name__ == "__main__":
    cfg = load_gen_config("configs/default_config.json")
    print(cfg)