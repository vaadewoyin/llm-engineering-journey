# Imports
from llm_cli.config import GenConfig, load_gen_config

# Test loading config from file
def test_load_gen_config():
    cfg = load_gen_config("configs/default_config.json")
    assert isinstance(cfg, GenConfig)
    assert cfg.model_id == "HuggingFaceTB/SmolLM2-135M-Instruct"
    assert cfg.system_prompt.startswith("You are a helpful assistant.")
    assert cfg.streamer is True
    assert cfg.do_sample is True
    assert cfg.temperature == 0.7
    assert cfg.top_p == 0.8
    assert cfg.max_new_tokens == 100
    assert cfg.repetition_penalty == 1.25