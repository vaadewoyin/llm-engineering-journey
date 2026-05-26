""" Arxiv scraper for qa dataset"""

import arxiv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
RAW_DIR = PROJECT_ROOT / "outputs" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

FILE_PATH = RAW_DIR / "papers.jsonl"   


def fetch_papers(query: str, n: int = 3000) -> list:
    """Fetch paper abstracts & metadata using arXiv API"""
    client = arxiv.Client(page_size=500, delay_seconds=3)
    search = arxiv.Search(
        query=query,
        max_results=n,
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

def save_papers(query: str, n: int = 10) -> None:
    """Fetch papers and save to JSONL"""
    papers = fetch_papers(query=query, n=n)
    with open(FILE_PATH, "w") as f:
        for item in papers:
            f.write(json.dumps(item) + "\n")
    print(f"Saved {len(papers)} papers to {FILE_PATH}")

# Run it
if __name__ == "__main__":
    save_papers(query="cat:cs.LG", n=10)