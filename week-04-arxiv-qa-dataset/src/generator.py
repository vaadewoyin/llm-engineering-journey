"""
Generate Q&A pairs from processed arXiv paper abstracts

Loads cleaned jsonl, feeds abstracts to LLM to generate Q&A pairs in ChatMl format,
generated pairs are saved to outputs/pairs/pairs.jsonl
"""

# Imports
import json
import re
import os
import warnings
from pathlib import Path
from dotenv import load_dotenv

import opik
from opik import track
from unsloth import FastLanguageModel
from transformers import logging as tf_logging
warnings.filterwarnings("ignore", category=FutureWarning)
tf_logging.set_verbosity_error()


# Load model & tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-Instruct-bnb-4bit",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
    device_map="auto"
)

# Load comet api key
load_dotenv()
comet_api_key = os.getenv("COMET_API_KEY")

# generation function 
opik.configure(
    api_key=comet_api_key,
    use_local=False   ,
    workspace="vaadewoyin",        # Comet workspace name
    automatic_approvals=True   # use Comet's cloud (free)
)

@track(project_name="kaggle-qa-generation")
def generate_qa(system_prompt, abstract):
    prompt = f"{system_prompt}\n\nAbstract: {abstract}"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
                **inputs,
                max_new_tokens=768, # Why: 768 tokens is sufficiently okay for generated outputs and keeps model's max seq len within 2048 limit
                do_sample=True,
                temperature=0.6,   # Why: high temperature values allows for more random response
                repetition_penalty=1.2) # Why: 1.2 is okay to discourage repetition of tokens
    input_length = inputs['input_ids'].shape[1]
    generated_tokens = outputs[0][input_length:]
    response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return response_text

def load_jsonl(file_path) -> list:
    """Read a JSONL file and return a list of records."""
    records = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:  # skip empty lines
                records.append(json.loads(line))
    return records

def process_generated_response(response):
    """
    Extract JSON array from model response, split into two Q&A pairs,
    and return each in ChatML format: {"messages": [user, assistant]}.
    On error, returns (None, None).
    """
    # Find the first '[' and its matching ']'
    # Why: llm generates some text before & after the json array, bracket matching to extract just the needed json array
    start = response.find('[')
    if start == -1:
        return None, None
    
    bracket_count = 0
    end = start
    for i, ch in enumerate(response[start:], start=start):
        if ch == '[':
            bracket_count += 1
        elif ch == ']':
            bracket_count -= 1
            if bracket_count == 0:
                end = i
                break
    if bracket_count != 0:
        return None, None
    
    json_str = response[start:end+1]
    
    # Attempt to parse JSON
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        # Clean common model errors and retry
        cleaned = json_str.replace("\\'", "'")
        cleaned = re.sub(r',\s*([\]}])', r'\1', cleaned)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            return None, None
    
    # Validate we have a list of 4 objects
    if not isinstance(data, list) or len(data) != 4:
        return None, None
    
    # Split into two ChatML messages
    first_pair = {"messages": data[:2]}
    second_pair = {"messages": data[2:]}
    
    return first_pair, second_pair


# Required paths
PROJECT_ROOT = Path(__file__).parent.parent
PROMPT_TXT_PATH = PROJECT_ROOT / "configs" / "prompt.txt"
CLEANED_PAPERS_PATH = PROJECT_ROOT / "outputs" / "cleaned" / "cleaned_papers.jsonl"
QA_PAIRS_PATH = PROJECT_ROOT / "outputs" / "pairs" /"pairs.jsonl"

# Load saved prompt
with open(PROMPT_TXT_PATH, "r") as f:
    prompt_text = f.read()

# Load cleaned papers 
cleaned_papers = load_jsonl(CLEANED_PAPERS_PATH)

# Why: To resume qa generation from papers already used.
# Why //2 : each paper produces 2 pairs, so divide pair count by 2 to get paper index.
existing_count = 0
if QA_PAIRS_PATH.exists():
    existing_count = sum(1 for _ in open(QA_PAIRS_PATH))
cleaned_papers_to_process = cleaned_papers[existing_count // 2:]
    
# Why: generate qa pairs per paper & write to jsonl immediately for faster execution
for i, paper in enumerate(cleaned_papers[:1000]): # Why: 1000 papers to keep within kaggle runtime limit
    paper_abstract = paper["abstract"]
    response = generate_qa(system_prompt=prompt_text, abstract=paper_abstract)
    first, second = process_generated_response(response)
    if first is None or second is None:
        continue
    pairs = [first, second]
    with open(QA_PAIRS_PATH, "a") as f:
         for item in pairs:
             f.write(json.dumps(item) + "\n")

