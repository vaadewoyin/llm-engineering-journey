"""
Clean arXiv papers for Q&A generation.

Reads raw papers from outputs/raw/papers.jsonl, removes duplicates by arXiv ID,
filters abstracts by configurable min/max word count, and writes cleaned papers
to outputs/cleaned/papers.jsonl.

The output is ready for the Q&A generation step.
"""

from pathlib import Path
import json


class PaperCleaner:
    """Prepares papers for Q&A by loading raw papers, removing duplicates,
    filtering papers by abstract length.
    """
    def __init__(self, config_path, raw_file_path):
        self.config_path = config_path
        self.file_path = raw_file_path

    def load_config(self):
        # Why: config loaded at runtime so missing file gives clear error message
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Config file not found at {self.config_path}. "
                "Create configs/config.json before running."
            )
        with open(self.config_path, "r") as f:
            return json.load(f)

        
    def load_jsonl(self) -> list:
        """Read a JSONL file and return a list of records."""
        records = []
        with open(self.file_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:  # skip empty lines
                    records.append(json.loads(line))
        return records
    
    def remove_duplicate_papers(self, raw_papers: list) -> list:
        """Removes duplicate papers by arXiv ID."""
        records = []
        unique_ids = set()
        for item in raw_papers:
            if item["id"] in unique_ids:
                continue
            unique_ids.add(item["id"])
            records.append(item)
        print(f"Count of raw papers - {len(raw_papers)}")
        print(f"Count of deduplicated papers - {len(records)}")
        return records
    

    def length_filter(self, config, deduplicated_papers:list)-> list:
        """Filters papers by abstract word count (min/max from config)."""
        # Why: For high quality Q&A generation, reasonable abstract length  is preferred
        records = []
        min_words = config["min_abstract_words"]
        max_words = config["max_abstract_words"]
        for paper in deduplicated_papers:
            word_count = len(paper['abstract'].split())
            if min_words <= word_count <= max_words:
                records.append(paper)
        print(f"Count of filtered papers - {len(records)}")
        return records
    
    def save_jsonl(self, filtered_papers:list, save_dir:Path)-> None:
        with open(save_dir, "w") as f:   
            for item in filtered_papers:
                f.write(json.dumps(item) + "\n")
        print(f"SAVED {len(filtered_papers)} papers to {save_dir}")
        


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).parent.parent
    CONFIG_PATH = PROJECT_ROOT / "configs" / "pipeline_config.json"
    FILE_PATH = PROJECT_ROOT / "outputs" / "raw" / "papers.jsonl"
    SAVE_DIR = PROJECT_ROOT / "outputs" / "cleaned" / "cleaned_papers.jsonl"

    cleaner = PaperCleaner(config_path=CONFIG_PATH, raw_file_path=FILE_PATH)

    config = cleaner.load_config()
    raw_papers = cleaner.load_jsonl()
    deduplicated = cleaner.remove_duplicate_papers(raw_papers)
    filtered_papers = cleaner.length_filter(config, deduplicated)
    
    # Save filtered_papers
    cleaner.save_jsonl(filtered_papers, SAVE_DIR)
