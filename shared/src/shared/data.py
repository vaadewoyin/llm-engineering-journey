"""Dataset loading + chat formatting — shared across training scripts."""

from datasets import load_dataset


def prepare_dataset(hf_repo: str):
    """Load train and validation splits from the HF dataset repo."""
    train = load_dataset(hf_repo, split="train")
    val = load_dataset(hf_repo, split="validation")
    return train, val


def build_message(row):
    return [
        {"role": "user", "content": row["question"]},
        {"role": "assistant", "content": row["answer"]},
    ]


def format_chat(tokenizer, dataset):
    """Apply the model's chat template for every row."""

    def build_prompt(row):
        messages = build_message(row)
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": prompt}

    return dataset.map(build_prompt)