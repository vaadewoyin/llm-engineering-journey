"""Clean generated QA pairs using judge verdicts as a filter.

Keep generated pairs whose (paper_id, question) key is marked "keep" in
the judged pairs.
"""

import json
import re
from config import SplitConfig


SPLIT_CFG = SplitConfig()


# Normalization
def normalize(text):
    return re.sub(r"\s+", " ", str(text).strip().lower())


def make_key(row, fields):
    return tuple(normalize(row.get(f, "")) for f in fields)


# IO helpers
def load_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return [json.loads(line) for line in lines if line.strip()]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# Filtering
def keep_keys(judged_rows, cfg):
    keys = set()
    for row in judged_rows:
        if row.get("decision") != cfg.keep_decision:
            continue
        keys.add(make_key(row, cfg.dedup_on))
    return keys


def filter_generated(qa_rows, keys, cfg):
    kept = []
    dropped_no_keep = 0
    dropped_dup = 0
    seen = set()

    for row in qa_rows:
        key = make_key(row, cfg.dedup_on)

        if key not in keys:
            dropped_no_keep += 1
            continue

        if key in seen:
            dropped_dup += 1
            continue
        seen.add(key)

        kept.append(row)

    return kept, dropped_no_keep, dropped_dup


# Pipeline
def run_pipeline(cfg=SPLIT_CFG):
    qa_rows = load_jsonl(cfg.qa_pairs_path)
    judged_rows = load_jsonl(cfg.judged_pairs_path)
    print(f"Loaded: {len(qa_rows)} generated | {len(judged_rows)} judged")

    keys = keep_keys(judged_rows, cfg)
    print(f"Keep keys from judge: {len(keys)}")

    kept, dropped_no_keep, dropped_dup = filter_generated(qa_rows, keys, cfg)
    write_jsonl(cfg.clean_pairs_path, kept)

    print(f"Kept: {len(kept)}")
    print(f"Dropped: {dropped_no_keep} not in judge keep, {dropped_dup} duplicates")
    print(f"Saved clean QA pairs: {cfg.clean_pairs_path}")


if __name__ == "__main__":
    run_pipeline()