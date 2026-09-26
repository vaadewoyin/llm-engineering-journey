"""Compares the generation outputs of two models given a prompt."""

import torch
from llm_cli.config import GenConfig
from llm_cli.generate import GenerationEngine


def compare_models(model_id1: str, model_id2: str, prompt: str) -> dict: 
    """ Compares the generation outputs of two models given a prompt."""

    # Model 1:
    # Why: streamer not needed in compare, outputs will be properly displayed with rich
    engine_1 = GenerationEngine(GenConfig(model_id=model_id1, streamer=False)) 
    output_1 = engine_1.generate(prompt)

    # Why: Clear memory to avoid OOM error, to allow working with two models in limited memory environment
    del engine_1
    torch.cuda.empty_cache()
    
    # Model 2:
    engine_2 = GenerationEngine(GenConfig(model_id=model_id2, streamer=False))
    output_2 = engine_2.generate(prompt)
    del engine_2
    torch.cuda.empty_cache()

    return {
        "model_id1_output": output_1,
        "model_id2_output": output_2,
        "prompt": prompt
        }

 