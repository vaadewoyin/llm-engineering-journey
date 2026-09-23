"""Judge generated QA pairs against their source chunks.

Scores each pair on factual correctness, groundedness, question relevance,
answer quality, and technical accuracy, and routes the result to keep,
borderline, or reject. Output is written as JSONL.
"""

import json
import os
import re
from itertools import islice

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import opik
from opik import track
from dotenv import load_dotenv

from config import JudgeConfig
from prompts import QA_JUDGE_SYSTEM_PROMPT as JUDGE_SYSTEM_PROMPT


load_dotenv()
COMET_ML_KEY = os.getenv("COMET_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
JUDGE_CFG = JudgeConfig()


# Batching helper
def chunked(iterable, size):
    """Yield successive chunks of `size` items from `iterable`."""
    it = iter(iterable)
    while batch := list(islice(it, size)):
        yield batch


# JSON helpers (strict)
def extract_json_object(text):
    if not text:
        return None

    text = text.strip()

    fence = re.search(r"```(?:json)?\s*(.*?)```", text,
                       flags=re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, dict):
        return None

    return parsed


# Prompt building
def build_user_prompt(qa):
    return f"""
Evaluate the following generated question-answer pair against the source chunk.

SOURCE CHUNK:
<
{qa["text"]}
>>>

GENERATED QUESTION:
<
{qa["question"]}
>>>

GENERATED ANSWER:
<
{qa["answer"]}
>>>
"""


# IO helpers
def load_jsonl(file_path, limit=None):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if limit is not None:
        lines = lines[:limit]
    return [json.loads(line) for line in lines]


# Prompt creation
def create_qa_prompts(qa_pairs, tokenizer):
    prompts = []
    for qa_pair in qa_pairs:
        qa = {
            "question": qa_pair["question"],
            "answer":   qa_pair["answer"],
            "text":     qa_pair["text"],
        }
        user_prompt = build_user_prompt(qa)
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            add_special_tokens=True,
            enable_thinking=False, 
        )
        prompts.append(prompt)
    return prompts


# Generation
@track(name=f"{JUDGE_CFG.project_name}-generate")
def generate_batch(batch_prompts, llm, sampling_params):
    outputs = llm.generate(batch_prompts, sampling_params, use_tqdm=False)
    return [o.outputs[0].text for o in outputs]


@track(capture_input=False, capture_output=False)
def batch_qa_judging(qa_pairs, prompts, llm, cfg: JudgeConfig,
                      checkpoint_every=5):

    bad_path = str(cfg.judged_pairs_path).replace(".jsonl", "_bad.jsonl")
    judged_count = 0

    sampling_params = SamplingParams(
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        max_tokens=cfg.max_new_tokens,
    )

    num_batches = (len(prompts) + cfg.batch_size - 1) // cfg.batch_size

    with open(cfg.judged_pairs_path, "w", encoding="utf-8") as f_out, \
         open(bad_path, "w", encoding="utf-8") as f_bad:

        prompt_batches = chunked(prompts, cfg.batch_size)
        pair_batches = chunked(qa_pairs, cfg.batch_size)

        for batch_idx, (batch_prompts, batch_pairs) in enumerate(
                zip(prompt_batches, pair_batches)):

            try:
                responses = generate_batch(
                    batch_prompts=batch_prompts,
                    llm=llm,
                    sampling_params=sampling_params,
                )
            except Exception as e:
                print(f"[batch {batch_idx}] generation failed: {e}")
                continue

            for response_text, qa_pair in zip(responses, batch_pairs):
                parsed = extract_json_object(response_text)
                if not parsed:
                    f_bad.write(json.dumps({
                        "global_id": qa_pair.get("global_id"),
                        "chunk_id": qa_pair.get("chunk_id"),
                        "raw_output": response_text,
                    }, ensure_ascii=False) + "\n")
                    continue

                for key in cfg.metadata_keys:
                    parsed[key] = qa_pair.get(key)

                f_out.write(json.dumps(parsed, ensure_ascii=False) + "\n")
                judged_count += 1

            if (batch_idx + 1) % checkpoint_every == 0:
                f_out.flush()
                os.fsync(f_out.fileno())
                print(f"[checkpoint] {batch_idx + 1}/{num_batches} batches, "
                      f"{judged_count} judged")

    print(f"Done. {judged_count} judgments written to {cfg.judged_pairs_path}")
    if os.path.exists(bad_path) and os.path.getsize(bad_path) > 0:
        print(f"Malformed outputs written to {bad_path}")
    return judged_count


# Pipeline
def run_pipeline(cfg: JudgeConfig = JUDGE_CFG):
    api_key = os.environ.get("COMET_API_KEY")
    if not api_key:
        raise RuntimeError("COMET_API_KEY not found in environment / .env")

    opik.configure(
        api_key=api_key,
        project_name=cfg.project_name,
        use_local=False,
        workspace=cfg.workspace,
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, token=HF_TOKEN)

    llm = LLM(
        model=cfg.model_name,
        dtype="auto",
        max_model_len=cfg.max_seq_length,
        gpu_memory_utilization=0.85,
        limit_mm_per_prompt={"image": 0},  
        trust_remote_code=True,
    )

    qa_pairs = load_jsonl(cfg.qa_pairs_path)
    prompts = create_qa_prompts(qa_pairs, tokenizer)

    batch_qa_judging(
        qa_pairs=qa_pairs,
        prompts=prompts,
        llm=llm,
        cfg=cfg,
    )


if __name__ == "__main__":
    run_pipeline()