"""Load the locked shared evaluation configuration and rubric."""

import json
from dataclasses import dataclass

from shared.paths import (
    BASELINE_EVAL_CONFIG_PATH,
    QUALITATIVE_RUBRIC_PATH,
)


@dataclass(frozen=True)
class HardwareConfig:
    device: str
    gpu: str
    compute_dtype: str


@dataclass(frozen=True)
class BaselineEvalConfig:
    experiment_seed: int
    data_train_split_ratio: float
    eval_set_path: str
    hardware: HardwareConfig


def load_baseline_eval_config() -> BaselineEvalConfig:
    raw = json.loads(BASELINE_EVAL_CONFIG_PATH.read_text())

    return BaselineEvalConfig(
        experiment_seed=raw["experiment_seed"],
        data_train_split_ratio=raw["data_train_split_ratio"],
        eval_set_path=raw["eval_set_path"],
        hardware=HardwareConfig(
            device=raw["hardware"]["device"],
            gpu=raw["hardware"]["gpu"],
            compute_dtype=raw["hardware"]["compute_dtype"],
        ),
    )


def load_rubric_text() -> str:
    return QUALITATIVE_RUBRIC_PATH.read_text()
