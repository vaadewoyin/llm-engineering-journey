import pytest
from pathlib import Path
from src.cleaner import PaperCleaner

@pytest.fixture
def sample_papers():
    """Returns a list of test papers with possible scenerios."""
    return [
        {
            "id": "001",
            "title": "Valid Paper",
            "abstract": "This abstract has a reasonable length. It contains enough words to pass the length filter easily. More words here to ensure it is well above the minimum threshold but below the maximum."
        },
        {
            "id": "001",
            "title": "Duplicate of Valid Paper",
            "abstract": "This is a duplicate ID but different abstract. Should be removed."
        },
        {
            "id": "002",
            "title": "Empty Abstract",
            "abstract": ""
        },
        {
            "id": "003",
            "title": "Short Abstract",
            "abstract": "Too short."
        },
        {
            "id": "004",
            "title": "Long Abstract",
            "abstract": "word " * 200  # 200 words, exceeds max threshold
        }
    ]

@pytest.fixture
def sample_config():
    """Config dictionary for length filter."""
    return {
        "min_abstract_words": 10,
        "max_abstract_words": 100
    }

@pytest.fixture
def cleaner():
    """PaperCleaner instance with dummy paths (I/O methods are not used)."""
    dummy_config = Path("dummy_config.json")
    dummy_raw = Path("dummy_raw.jsonl")
    return PaperCleaner(config_path=dummy_config, raw_file_path=dummy_raw)

# Tests functions
def test_deduplication(cleaner, sample_papers):
    # Expect 4 unique papers (one duplicate with id "001" is removed)
    deduped = cleaner.remove_duplicate_papers(sample_papers)
    assert len(deduped) == 4
    ids = [i["id"] for i in deduped]
    assert ids == ["001", "002", "003", "004"]
           
def test_length_filter(sample_config, cleaner, sample_papers):
    deduped = cleaner.remove_duplicate_papers(sample_papers)
    filtered = cleaner.length_filter(sample_config, deduped)
    # Only the valid paper (id 001) left
    # - "001" valid: passes length ( >10 and <100)
    # - "002" empty: fails min
    # - "003" short: fails min
    # - "004" long: fails max
    assert len(filtered)  == 1
    assert filtered[0]["id"] == "001"

