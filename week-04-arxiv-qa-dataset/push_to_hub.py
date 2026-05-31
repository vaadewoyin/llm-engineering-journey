"""Upload Q&A pair dataset to Huggingface hub"""

from datasets import load_dataset

# Load the JSONL file
dataset = load_dataset(
    "json", 
    data_files="outputs/final/qa_pairs.jsonl",  
    split="train"
)

# Push to Hugging Face Hub (overwrite former with new correction)
dataset.push_to_hub("vaadewoyin/arxiv-ml-qa-dataset", private=True)