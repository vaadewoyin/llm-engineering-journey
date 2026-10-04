"""Paths to the locked shared evaluation files."""

from pathlib import Path

SHARED_ROOT = Path(__file__).resolve().parents[2]

BASELINE_EVAL_CONFIG_PATH = (
    SHARED_ROOT / "configs" / "baseline_eval_config.json"
)

QUALITATIVE_RUBRIC_PATH = (
    SHARED_ROOT / "eval" / "qualitative_rubric.md"
)
