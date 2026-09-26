""" Test for the GenConfig and load_gen_config function. """
from llm_cli.config import GenConfig, load_gen_config

def test_gen_config_defaults():
    """Test that the GenConfig dataclass has the correct default values."""
    cfg = GenConfig()
    assert cfg.model_id == "HuggingFaceTB/SmolLM2-135M-Instruct"
    assert cfg.system_prompt.startswith("You are a helpful assistant.")
    assert cfg.streamer is True
    assert cfg.do_sample is True
    assert cfg.temperature == 0.7
    assert cfg.top_p == 0.8
    assert cfg.max_new_tokens == 100
    assert cfg.repetition_penalty == 1.25

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