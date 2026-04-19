# Imports
import torch
from llm_cli.config import GenConfig#, SaveAndLoadConfig
from llm_cli.generate import GenerationEngine
#from typing import Optional, Dict, Any


def compare_models(model_id1: str, model_id2: str, prompt: str) -> dict: 
    """ Compares the generation outputs and tokens per second of two models given a prompt."""
    # Model 1
    engine_1 = GenerationEngine(GenConfig(model_id=model_id1))
    output_1 = engine_1.generate(prompt)
    # Clear memory to avoid OOM error
    del engine_1
    torch.cuda.empty_cache()
    
    # Model 2
    engine_2 = GenerationEngine(GenConfig(model_id=model_id2))
    output_2 = engine_2.generate(prompt)
    # Clear memory to avoid OOM error
    del engine_2
    torch.cuda.empty_cache()

    return {
        "model_id1_output": output_1,
        "model_id2_output": output_2,
        "prompt": prompt
        }

 