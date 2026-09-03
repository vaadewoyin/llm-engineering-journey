
"""
Judges generated QA pairs using Gemma 4
"""

PROJECT_NAME = "sustainable-conc-papers-qa-filtering"

# Necessary imports
import json
from pathlib import Path
import torch
from unsloth import FastLanguageModel
import opik
from opik import track

JUDGE_SYSTEM_PROMPT = """
You are a strict quality-control judge for a scientific question-answer dataset about concrete, cement, mortar, geopolymer materials, and sustainable/alternative concrete materials.

You will receive a source chunk and a generated question-answer pair.

Your task is to evaluate the generated question and answer ONLY against the supplied source chunk.

Evaluate the following five criteria:

1. Factual correctness (1-5)
- Is the answer factually consistent with the source chunk?
- Does it contain any incorrect claims?
- Numerical values, percentages, units, and experimental results must be represented correctly.
- If the answer contradicts the source, score 1.

2. Groundedness (1-5)
- Can the answer be directly supported by information in the source chunk?
- Penalize unsupported inference or information that cannot reasonably be derived from the source.
- If the answer introduces information that is not supported by the source, score 2 or lower.
- Do not penalize a reasonable interpretation that follows directly from the information in the source.

3. Question relevance (1-5)
- Does the question ask about information actually contained in the source chunk?
- If the question cannot be answered from the source chunk, score 1.
- If the question addresses a key finding, relationship, result, or interpretation from the source, score 5.
- If the question concerns a minor but valid detail from the source, score 3 or 4.
- The question must be answerable using the supplied source chunk.

4. Answer quality (1-5)
- Is the answer clear, precise, complete, and directly responsive to the question?
- Penalize vague, incomplete, confusing, or unnecessarily verbose answers.
- Accept minor paraphrasing as long as the meaning is preserved.
- The answer should contain enough information to properly answer the question without adding irrelevant information.

5. Technical accuracy (1-5)
- Are technical terms, materials, experimental results, units, percentages, values, and relationships represented correctly?
- Penalize incorrect units, misstated relationships, incorrect numerical values, or misinterpretation of experimental findings.
- Do not accept technically plausible information that is not supported by the source.

IMPORTANT RULES:
- Judge ONLY from the supplied source chunk.
- Do not use outside knowledge to fill missing information.
- If the source does not provide enough information to answer the question, score the QA accordingly.
- Do not reward an answer simply because it sounds scientifically plausible.
- A question may be rejected even if its answer is correct if the question itself is not sufficiently grounded in the source.
- Minor wording differences are acceptable if the meaning remains faithful to the source.
- Distinguish between a reasonable interpretation of the source and an unsupported inference.
- Pay particular attention to numerical values, percentages, units, material proportions, experimental conditions, and reported trends.

OVERALL SCORE:
The overall score should reflect the overall quality of the QA pair. Do not simply average the five scores.

Give particular importance to:
- factual correctness
- groundedness
- technical accuracy

A serious weakness in any of these core dimensions should lower the overall score.

Overall score:
5 = Excellent QA with no meaningful weaknesses
4 = Good QA with only minor weaknesses
3 = Acceptable but has a noticeable weakness
2 = Poor QA with a significant problem
1 = Unacceptable QA

DECISION RULES:
- "keep": All five criteria are ≥ 4 and the QA is substantively correct and well grounded.
- "borderline": No criterion is ≤ 2, but at least one criterion is 3. These require manual review.
- "reject": Any criterion is ≤ 2, or there is a substantive factual, grounding, relevance, or technical problem.

Return ONLY valid JSON in exactly this format:

{
  "factual_correctness": 1-5,
  "groundedness": 1-5,
  "question_relevance": 1-5,
  "answer_quality": 1-5,
  "technical_accuracy": 1-5,
  "overall_score": 1-5,
  "decision": "keep" or "borderline" or "reject",
  "reason": "Brief explanation of the main reason for the score and decision."
}

SOURCE CHUNK:
{chunk}

GENERATED QUESTION:
{question}

GENERATED ANSWER:
{answer}
"""


def build_user_prompt(qa):
    return f"""
    Evaluate the following generated question-answer pair against the source chunk.

    SOURCE CHUNK:
    <<<
    {qa["text"]}
    >>>

    GENERATED QUESTION:
    <<<
    {qa["question"]}
    >>>

    GENERATED ANSWER:
    <<<
    {qa["answer"]}
    >>>
    """

def load_jsonl(file_path):
    """Load all lines from a JSONL file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def save_jsonl(file_path, data, overwrite=True):
    """Write a list of dicts to a JSONL file"""
    mode = "w" if overwrite else "a"
    with open(file_path, mode, encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

# Create prompts
def create_qa_prompts(qa_pairs, tokenizer):
    qa_prompts = []
    for qa_pair in qa_pairs:
        qa = {
            "question" : qa_pair["question"],
            "answer" : qa_pair["answer"],
            "text" : qa_pair["text"]
        }
        user_prompt = build_user_prompt(qa)
        messages = [
        {"role": "system", "content": f"{JUDGE_SYSTEM_PROMPT}"},
        {"role": "user", "content": f"{user_prompt}"}
        ]
        qa_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize =False,
            add_generation_prompt = True,
            enable_thinking=True
        )
        qa_prompts.append(qa_prompt)
    return qa_prompts

@track(name=PROJECT_NAME)#, ignore_arguments=["batch_chunks"])
def generate_batch(batch_prompts, model, tokenizer, max_new_tokens):

    inputs = tokenizer(batch_prompts,
                       return_tensors="pt",
                       truncation=True,
                       max_length=None,
                       padding=True,
                       padding_side="left").to("cuda")

    outputs = model.generate(**inputs,
                             max_length=None,
                             max_new_tokens=max_new_tokens,
                             temperature=1.0,
                             top_p=0.95,
                             top_k=64,
                             # repetititon
                             do_sample=True)

    batch_input_length = inputs["input_ids"].shape[1]
    return [tokenizer.decode(outputs[i][batch_input_length:], skip_special_tokens=True)
            for i in range(len(batch_prompts))]

@track(capture_input=False, capture_output=False)
def batch_qa_generation(qa_pairs, prompts,
                        model, tokenizer,
                        batch_size, max_new_tokens,
                        qa_metadata_keys, filtered_qa_save_dir):

    filtered_qa_results = []

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        batch_qa_pairs = qa_pairs[i:i+batch_size]
        responses = generate_batch(batch_prompts=batch_prompts,
                                   model=model, tokenizer=tokenizer,
                                   max_new_tokens=max_new_tokens)
        for response, qa_pair in zip(responses, batch_qa_pairs):
            try:
                responses_json = json.loads(response)
            except json.JSONDecodeError:
                continue

            if isinstance(responses_json, list) and len(responses_json) >= 1:
                for qa in responses_json:
                    for key in qa_metadata_keys:
                        qa[key] = qa_pair[key]

                    filtered_qa_results.append(qa)

        if (i > 10) and (i < 20):
            break


    # Write qa_results to jsonl
    with open(filtered_qa_save_dir, "w") as f:
        for qa in filtered_qa_results:
            f.write(json.dumps(qa, ensure_ascii=False) + "\n")

def run_pipeline():
    from google.colab import userdata
    COMET_ML_KEY = userdata.get('COMET_API_KEY')

    # Config
    PROJECT_NAME = "sustainable-conc-papers-qa-filtering-demo" #"sustainable-conc-papers-qa-gen"
    MODEL_NAME = "unsloth/gemma-4-E2B-it"  #"unsloth/Qwen3.8-27B-unsloth-bnb-4bit"
    
    MAX_NEW_TOKENS = 512
    MAX_SEQ_LENGTH = 6144
    BATCH_SIZE = 4

    QA_PAIRS_PATH = "qa_pairs.jsonl"
    FILTERED_QA_PATH = "filtered_qa_pairs.jsonl"

    METADATA_KEYS = [
        "question", "answer" "global_id", "paper_id",
        "paper_title", "paper_year", "paper_url",
        "downloaded_paper_name", "chunk_id", "section"
    ]

    # Opik config
    opik.configure(
        api_key=COMET_ML_KEY,
        project_name=PROJECT_NAME,
        use_local=False,
        workspace="vaadewoyin"        # Comet workspace name
    )

    # Load model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = None,
        load_in_4bit = True,
        device_map="auto"
    )

    qa_pairs = load_jsonl(QA_PAIRS_PATH)
    qa_prompts = create_qa_prompts(qa_pairs, tokenizer)

    batch_qa_generation(qa_pairs=qa_pairs,
                        prompts=qa_prompts,
                        model=model,
                        tokenizer=tokenizer,
                        batch_size=BATCH_SIZE,
                        max_new_tokens = MAX_NEW_TOKENS,
                        qa_metadata_keys=METADATA_KEYS,
                        qa_save_dir=FILTERED_QA_PATH)

run_pipeline()

