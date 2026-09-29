"""Verify no paper appears in both train and val splits."""

import json
from config import SplitConfig


SPLIT_CFG = SplitConfig()


def load_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return [json.loads(line) for line in lines if line.strip()]


def run_pipeline(cfg=SPLIT_CFG):
    train = load_jsonl(cfg.splits_dir / "train.jsonl")
    val = load_jsonl(cfg.splits_dir / "val.jsonl")

    train_papers = {r[cfg.split_by] for r in train}
    val_papers = {r[cfg.split_by] for r in val}

    overlap = train_papers & val_papers
    if overlap:
        raise SystemExit(
            f"LEAKAGE: {len(overlap)} papers in both splits: "
            f"{sorted(overlap)[:5]}"
        )

    print(f"Train papers: {len(train_papers)} | Pairs: {len(train)}")
    print(f"Val papers:   {len(val_papers)} | Pairs: {len(val)}")
    print("No paper overlap.")


if __name__ == "__main__":
    run_pipeline()