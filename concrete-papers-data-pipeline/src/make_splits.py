"""Split cleaned QA pairs into train and val.

Splitting by paper, not by row, prevents the same paper appearing in
both splits. 
"""

import json
import random
from config import SplitConfig


SPLIT_CFG = SplitConfig()


def load_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return [json.loads(line) for line in lines if line.strip()]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# Grouping
def group_by_paper(rows, paper_field):
    groups = {}
    for row in rows:
        pid = row.get(paper_field)
        if not pid:
            raise ValueError(f"Row missing {paper_field}: {row.get('global_id')}")
        if pid not in groups:
            groups[pid] = []
        groups[pid].append(row)
    return groups


# Splitting
def split_papers(groups, cfg):
    paper_ids = sorted(groups.keys())
    rng = random.Random(cfg.split_seed)
    rng.shuffle(paper_ids)

    n = len(paper_ids)
    n_train = round(cfg.split_proportions["train"] * n)
    train_ids = set(paper_ids[:n_train])
    val_ids = set(paper_ids[n_train:])

    train_rows = [r for pid in train_ids for r in groups[pid]]
    val_rows = [r for pid in val_ids for r in groups[pid]]

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)

    return train_rows, val_rows


# Pipeline
def run_pipeline(cfg=SPLIT_CFG):
    rows = load_jsonl(cfg.clean_pairs_path)
    groups = group_by_paper(rows, cfg.split_by)

    train_rows, val_rows = split_papers(groups, cfg)

    write_jsonl(cfg.splits_dir / "train.jsonl", train_rows)
    write_jsonl(cfg.splits_dir / "val.jsonl", val_rows)

    train_papers = len({r[cfg.split_by] for r in train_rows})
    val_papers = len({r[cfg.split_by] for r in val_rows})

    print(f"Loaded: {len(rows)} pairs")
    print(f"Papers: {len(groups)} total | {train_papers} train | {val_papers} val")
    print(f"Pairs:  {len(train_rows)} train | {len(val_rows)} val")
    print(f"Saved train: {cfg.splits_dir / 'train.jsonl'}")
    print(f"Saved val:   {cfg.splits_dir / 'val.jsonl'}")


if __name__ == "__main__":
    run_pipeline()