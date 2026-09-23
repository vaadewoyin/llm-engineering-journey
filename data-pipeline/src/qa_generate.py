"""Generate question-answer pairs from scientific chunks.

Loads a quantized Qwen3 model via vLLM, builds chat-templated prompts for
each chunk, and generates one grounded QA pair per chunk.
"""

# Imports
import json
import os
import re
from itertools import islice

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import opik
from opik import track
from dotenv import load_dotenv

from config import QAConfig
from prompts import QA_GENERATION_SYSTEM_PROMPT as SYSTEM_PROMPT

# Config & secrets
load_dotenv()
COMET_ML_KEY = os.getenv("COMET_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
CFG = QAConfig()


# Batching helper
def chunked(iterable, size):
    """Yield successive chunks of `size` items from `iterable`."""
    it = iter(iterable)
    while batch := list(islice(it, size)):
        yield batch


# JSON parsing helpers
def extract_json_array(text):
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

    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        return None

    return parsed


# Prompt building
def build_user_prompt(chunk_text):
    return f"""
Analyze the following scientific chunk according to your instructions.

<
{chunk_text}
>>>
"""


# IO helpers
def load_jsonl(file_path, limit=None):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if limit is not None:
        lines = lines[:limit]
    return [json.loads(line) for line in lines]


def save_jsonl(file_path, data, overwrite=True):
    mode = "w" if overwrite else "a"
    with open(file_path, mode, encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# Chunk processing
def filter_chunks(chunks_path, filtered_path, token_threshold=150):
    """Keep chunks with token_count >= threshold AND non-empty text."""
    chunks = load_jsonl(chunks_path)
    kept, removed = [], 0
    for chunk in chunks:
        has_text = bool(chunk.get("text"))
        long_enough = chunk.get("token_count", 0) >= token_threshold
        if has_text and long_enough:
            kept.append(chunk)
        else:
            removed += 1
    save_jsonl(filtered_path, kept)
    print(f"Kept: {len(kept)} chunks, Removed: {removed} chunks "
          f"(threshold {token_threshold})")
    return len(kept), removed


def add_global_id(input_path, output_path, id_format="{:04d}"):
    chunks = load_jsonl(input_path)
    for idx, chunk in enumerate(chunks, start=1):
        chunk["global_id"] = f"chunk_{id_format.format(idx)}"
    save_jsonl(output_path, chunks)
    print(f"Added global IDs to {len(chunks)} chunks")
    return chunks


def process_chunks(chunks_path, filtered_path, final_path, token_threshold=150):
    filter_chunks(chunks_path, filtered_path, token_threshold)
    add_global_id(filtered_path, final_path)
    print(f"Final file: {final_path}")


# Prompt creation
def create_prompts(chunks, tokenizer):
    prompts = []
    for chunk in chunks:
        chunk_text = chunk["text"]
        user_prompt = build_user_prompt(chunk_text)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
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


def sort_prompts(chunks, prompts):
    combined = sorted(
        zip(chunks, prompts),
        key=lambda pair: pair[0].get("token_count", 0),
    )
    chunks_sorted = [c for c, _ in combined]
    prompts_sorted = [p for _, p in combined]
    return chunks_sorted, prompts_sorted


# Generation
@track(name=f"{CFG.project_name}-generate")
def generate_batch(batch_prompts, llm, sampling_params):
    """One call to vLLM — it handles internal batching/scheduling itself."""
    outputs = llm.generate(batch_prompts, sampling_params, use_tqdm=False)
    return [o.outputs[0].text for o in outputs]


@track(capture_input=False, capture_output=False)
def batch_qa_generation(chunks, prompts, llm, cfg: QAConfig,
                         checkpoint_every=5):

    bad_path = str(cfg.qa_pairs_path).replace(".jsonl", "_bad.jsonl")
    qa_written = 0

    sampling_params = SamplingParams(
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        min_p=cfg.min_p,
        max_tokens=cfg.max_new_tokens,
    )

    num_batches = (len(prompts) + cfg.batch_size - 1) // cfg.batch_size

    with open(cfg.qa_pairs_path, "w", encoding="utf-8") as f_out, \
         open(bad_path, "w", encoding="utf-8") as f_bad:

        prompt_batches = chunked(prompts, cfg.batch_size)
        chunk_batches = chunked(chunks, cfg.batch_size)

        for batch_idx, (batch_prompts, batch_chunks) in enumerate(
                zip(prompt_batches, chunk_batches)):

            try:
                responses = generate_batch(
                    batch_prompts=batch_prompts,
                    llm=llm,
                    sampling_params=sampling_params,
                )
            except Exception as e:
                print(f"[batch {batch_idx}] generation failed: {e}")
                continue

            for response_text, chunk in zip(responses, batch_chunks):
                parsed = extract_json_array(response_text)
                if not parsed:
                    f_bad.write(json.dumps({
                        "global_id": chunk.get("global_id"),
                        "chunk_id": chunk.get("chunk_id"),
                        "raw_output": response_text,
                    }, ensure_ascii=False) + "\n")
                    continue

                qa = parsed[0]
                if not isinstance(qa, dict):
                    continue
                if "question" not in qa or "answer" not in qa:
                    continue

                for key in cfg.metadata_keys:
                    qa[key] = chunk.get(key)

                f_out.write(json.dumps(qa, ensure_ascii=False) + "\n")
                qa_written += 1

            if (batch_idx + 1) % checkpoint_every == 0:
                f_out.flush()
                os.fsync(f_out.fileno())
                print(f"[checkpoint] {batch_idx + 1}/{num_batches} batches, "
                      f"{qa_written} QA pairs")

    print(f"Done. {qa_written} QA pairs written to {cfg.qa_pairs_path}")
    if os.path.exists(bad_path) and os.path.getsize(bad_path) > 0:
        print(f"Malformed outputs written to {bad_path}")
    return qa_written


# Pipeline
def run_pipeline(cfg: QAConfig = CFG):
    if not COMET_ML_KEY:
        raise RuntimeError("COMET_API_KEY not found in environment / .env")

    opik.configure(
        api_key=COMET_ML_KEY,
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
        limit_mm_per_prompt={"image": 0, "video": 0},  
        max_num_seqs=64,
        trust_remote_code=True,
    )

    process_chunks(cfg.chunks_path, cfg.filtered_path,
                    cfg.final_chunks_path, cfg.token_threshold)

    chunks = load_jsonl(cfg.final_chunks_path)
    prompts = create_prompts(chunks, tokenizer)
    chunks_sorted, prompts_sorted = sort_prompts(chunks, prompts)

    batch_qa_generation(
        chunks=chunks_sorted,
        prompts=prompts_sorted,
        llm=llm,
        cfg=cfg,
    )


if __name__ == "__main__":
    run_pipeline()