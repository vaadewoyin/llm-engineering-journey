"""ArXiv scraper for QA dataset """

import arxiv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
RAW_DIR = PROJECT_ROOT / "outputs" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_FILE_PATH = PROJECT_ROOT/ "configs" / "pipeline_config.json"
FILE_PATH = RAW_DIR / "papers.jsonl"   

# Configuration
def load_config() -> dict:
    # Why: config loaded at runtime so missing file gives clear error message
    if not CONFIG_FILE_PATH.exists():
        raise FileNotFoundError(
            f"Config file not found at {CONFIG_FILE_PATH}. "
            "Create configs/config.json before running."
        )
    with open(CONFIG_FILE_PATH, "r") as f:
        return json.load(f)

config = load_config()

# Core functions

def fetch_papers(query: str, n: int) -> list:
    """Fetch paper abstracts & metadata using arXiv API"""
    client = arxiv.Client(
                        # Why: page_size=500 to minimise API round trips within rate limits
                        # Why: delay_seconds=3 to respect ArXiv API rate limiting
                        page_size=config["page_size"],  
                        delay_seconds=config["delay_seconds"]
                        )
    
    search = arxiv.Search(
        query=query,
        max_results=n,
        # Why: sort by SubmittedDate to get recent ML papers
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    records = []

    for r in client.results(search):
        records.append({
            "id": r.entry_id.split("/abs/")[-1],
            "title": r.title,
            "abstract": r.summary,
        })
    return records


def load_existing_ids() -> set:
    """Load already-scraped paper IDs from checkpoint."""
    if not FILE_PATH.exists():
        return set()
    ids = set()
    with open(FILE_PATH, "r") as f:
        for line in f:
            ids.add(json.loads(line)["id"])
    return ids


def save_papers(query: str, n: int = 10) -> tuple:
    """Fetch new papers and append to JSONL, skipping duplicates."""
    existing_ids = load_existing_ids()
    print(f"Already have {len(existing_ids)} papers")
    # Fetch papers 
    papers = fetch_papers(query=query, n=n)
    # Filter out papers we already have
    new_papers = [p for p in papers if p["id"] not in existing_ids]
    print(f"Found {len(new_papers)} new papers")
    # Append to file 
    with open(FILE_PATH, "a") as f:   
        for item in new_papers:
            f.write(json.dumps(item) + "\n")
    print(f"Appended {len(new_papers)} papers to {FILE_PATH}")
    return len(existing_ids), len(new_papers)   

# Check
if __name__ == "__main__":
    save_papers(query=config["arxiv_query"], 
                n=config["max_papers"])