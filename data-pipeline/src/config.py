"""Project-wide configuration.

Defines QAConfig, a dataclass holding every tunable in the QA pipeline:
model name, sequence lengths, sampling parameters, file paths, and metadata keys.
"""
from pathlib import Path
from dataclasses import dataclass, field


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

@dataclass
class QAConfig:
    # project 
    project_name: str = "llm-engineering-journey-data-gen"
    model_name:   str = "Qwen/Qwen3.8-27B-FP8"
    workspace:    str = "vaadewoyin"

    # context / generation 
    max_seq_length:     int = 6144
    max_input_tokens:   int = 4096
    max_new_tokens:     int = 512
    batch_size:         int = 256       # checkpoint chunk size, not a hardware limit

    # filtering
    token_threshold: int = 150

    # Qwen3 non-thinking recommended sampling
    temperature: float = 0.7
    top_p:       float = 0.80
    top_k:       int   = 20
    min_p:       float = 0.0
    
    # paths
    chunks_path:       Path = DATA_DIR / "chunks.jsonl"
    filtered_path:     Path = DATA_DIR / "filtered_chunks.jsonl"
    final_chunks_path: Path = DATA_DIR / "filtered_chunks_final.jsonl"
    qa_pairs_path:     Path = DATA_DIR / "qa_pairs.jsonl"

    # metadata propagated into each QA record 
    metadata_keys: list[str] = field(default_factory=lambda: [
        "text", "global_id", "paper_id",
        "paper_title", "paper_year", "paper_url",
        "downloaded_paper_name", "section",
    ])


@dataclass
class JudgeConfig:
    project_name: str = "llm-engineering-journey-data-judge"
    model_name:   str = "google/gemma-4-26B-A4B-it"
    workspace:    str = "vaadewoyin"

    max_seq_length:   int = 6144
    max_input_tokens: int = 4096
    max_new_tokens:   int = 256
    batch_size:       int = 256 # checkpoint chunk size, not a hardware limit

    temperature: float = 0
    top_p:       float = 1
    top_k:       int   = -1

    qa_pairs_path:     Path = DATA_DIR / "qa_pairs.jsonl"
    judged_pairs_path: Path = DATA_DIR / "judged_qa_pairs.jsonl"

    metadata_keys: list[str] = field(default_factory=lambda: [
        "question", "answer", "global_id", "paper_id",
        "paper_title", "paper_year", "paper_url",
        "downloaded_paper_name", "chunk_id", "section",
    ])

