"""
Filter for high quality Q&A pairs 

Reads Q&A pairs from outputs/pairs/pairs.jsonl, validates ChatML format for each pair, 
validates that Q&A length for each pair, saves filtered pairs to outputs/final/qa_dataset.jsonl,
"""

import json
from pathlib import Path


def load_config(config_path):
    # Why: config loaded at runtime so missing file gives clear error message
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found at {config_path}. "
            "Create configs/config.json before running."
        )
    with open(config_path, "r") as f:
        return json.load(f)

def is_valid_chatml_pair(obj: dict, config:dict) -> bool:
    """
    Returns True if obj matches:
    {"messages": [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]}
    with question word count >= 10 and answer word count >= 20.
    """
    if not isinstance(obj, dict): # validates that the pair is first a dict as expected of json
        return False   
    msgs = obj.get("messages")
    if not isinstance(msgs, list) or len(msgs) != 2:
        return False
    # Check first message (user) and second message (assistant)
    user_msg, asst_msg = msgs[0], msgs[1]
    if not (isinstance(user_msg, dict) and isinstance(asst_msg, dict)):
        return False
    if user_msg.get("role") != "user" or asst_msg.get("role") != "assistant":
        return False
    # Word counts
    user_words = user_msg.get("content", "").split()
    asst_words = asst_msg.get("content", "").split()
    return len(user_words) >= config["min_question_length"] and len(asst_words) >= config["min_answer_length"]


# Save filtered data
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_FILE_PATH = PROJECT_ROOT/ "configs" / "pipeline_config.json"
QA_PAIRS_PATH = PROJECT_ROOT / "outputs" / "pairs" /"pairs.jsonl"
FILTERED_PAIRS_PATH = PROJECT_ROOT / "outputs" / "final" /"qa_pairs.jsonl"

# Load config
config = load_config(CONFIG_FILE_PATH)

# Why: Writing filtered pairs to output instantly is more efficient esp. when data is too large for memory
with open(QA_PAIRS_PATH, "r") as f:
    # Why: "w" in each run to reflect current run & prevent duplicates over multiple re-runs
    with open(FILTERED_PAIRS_PATH, "w") as out: 
        total = 0
        passed = 0
        for line in f:
            try:
                data = json.loads(line)
                if is_valid_chatml_pair(data, config=config):
                    out.write(line)
                    passed += 1 
            except json.JSONDecodeError:
                continue
            total+=1
print(f"Total Q&A pairs before filtering - {total}")
print(f"Total Q&A pairs after filtering - {passed}")
