"""Judge generated QA pairs against their source chunks.

Scores each pair on factual correctness, groundedness, question relevance,
answer quality, and technical accuracy, and routes the result to keep,
borderline, or reject. Output is written as JSONL.
"""

import json
import os
import re

import torch
from unsloth import FastLanguageModel
import opik
from opik import track
from dotenv import load_dotenv

from config import JudgeConfig
from prompts import QA_JUDGE_SYSTEM_PROMPT as JUDGE_SYSTEM_PROMPT


load_dotenv()
COMET_ML_KEY = os.getenv("COMET_API_KEY")
JUDGE_CFG = JudgeConfig()


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
            {"role": "system", "content": [{"type": "text", "text": JUDGE_SYSTEM_PROMPT}]},
            {"role": "user",   "content": [{"type": "text", "text": user_prompt}]},
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            add_special_tokens=True,
            enable_thinking=False,   # strict JSON output — no thinking block
        )
        prompts.append(prompt)
    return prompts


# Generation
@track(name=f"{JUDGE_CFG.project_name}-generate")
def generate_batch(batch_prompts, model, tokenizer, cfg: JudgeConfig):
    inputs = tokenizer(
        batch_prompts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=cfg.max_input_tokens,
    ).to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        do_sample=True,
    )

    input_len = inputs["input_ids"].shape[1]
    return [
        tokenizer.decode(outputs[i][input_len:], skip_special_tokens=True)
        for i in range(len(batch_prompts))
    ]


@track(capture_input=False, capture_output=False)
def batch_qa_judging(qa_pairs, prompts, model, tokenizer, cfg: JudgeConfig,
                     checkpoint_every=5):
    bad_path = str(cfg.judged_pairs_path).replace(".jsonl", "_bad.jsonl")
    judged_count = 0

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    num_batches = (len(prompts) + cfg.batch_size - 1) // cfg.batch_size

    with open(cfg.judged_pairs_path, "w", encoding="utf-8") as f_out, \
         open(bad_path,              "w", encoding="utf-8") as f_bad, \
         torch.inference_mode():

        for batch_idx, i in enumerate(range(0, len(prompts), cfg.batch_size)):
            batch_prompts = prompts[i:i + cfg.batch_size]
            batch_pairs   = qa_pairs[i:i + cfg.batch_size]

            try:
                responses = generate_batch(
                    batch_prompts=batch_prompts,
                    model=model,
                    tokenizer=tokenizer,
                    cfg=cfg,
                )
            except Exception as e:
                print(f"[batch {batch_idx}] generation failed: {e}")
                continue

            for response_text, qa_pair in zip(responses, batch_pairs):
                parsed = extract_json_object(response_text)
                if not parsed:
                    f_bad.write(json.dumps({
                        "global_id":  qa_pair.get("global_id"),
                        "chunk_id":   qa_pair.get("chunk_id"),
                        "raw_output": response_text,
                    }, ensure_ascii=False) + "\n")
                    continue

                # Attach metadata from the original QA record
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

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model_name,
        max_seq_length=cfg.max_seq_length,
        dtype=None,
        load_in_4bit=True,
        device_map="auto",
    )

    qa_pairs = load_jsonl(cfg.qa_pairs_path)
    prompts = create_qa_prompts(qa_pairs, tokenizer)

    batch_qa_judging(
        qa_pairs=qa_pairs,
        prompts=prompts,
        model=model,
        tokenizer=tokenizer,
        cfg=cfg,
    )


if __name__ == "__main__":
    run_pipeline()