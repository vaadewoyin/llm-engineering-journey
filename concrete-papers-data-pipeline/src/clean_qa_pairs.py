"""Clean judged QA pairs and write the kept pairs to a new JSONL file.

Keeps only rows where `decision` equals "keep" label, drops
duplicates on (paper_id, question), and writes the result to
`data/cleaned/pairs_clean.jsonl`.

Output is JSONL, one object per line, ready for make_splits.py.
"""

import json
import re

from config import SplitConfig


SPLIT_CFG = SplitConfig()


# Normalization
def normalize(text):
    """Lowercase, strip, and collapse internal whitespace."""
    return re.sub(r"\s+", " ", str(text).strip().lower())


def load_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return [json.loads(line) for line in lines if line.strip()]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# Cleaning
def clean_pairs(rows, cfg: SplitConfig):
    kept = []
    dropped_decision = 0
    dropped_dup = 0
    seen = set()

    for row in rows:
        if row.get("decision") != cfg.keep_decision:
            dropped_decision += 1
            continue

        key = tuple(normalize(row.get(k, "")) for k in cfg.dedup_on)
        if key in seen:
            dropped_dup += 1
            continue
        seen.add(key)

        kept.append(row)

    return kept, dropped_decision, dropped_dup


# Pipeline
def main(cfg: SplitConfig = SPLIT_CFG):
    rows = load_jsonl(cfg.judged_pairs_path)
    print(f"Loaded {len(rows)} raw judged pairs from {cfg.judged_pairs_path}")

    kept, dropped_decision, dropped_dup = clean_pairs(rows, cfg)

    write_jsonl(cfg.clean_pairs_path, kept)

    print(f"Kept: {len(kept)}")
    print(f"Dropped: {dropped_decision} decision, {dropped_dup} duplicates")
    print(f"Saved clean judged pairs: {cfg.clean_pairs_path}")


if __name__ == "__main__":
    main()