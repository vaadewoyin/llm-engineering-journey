"""Dataset loading + chat formatting — shared across training scripts."""

from datasets import load_dataset


def prepare_dataset(hf_train_link: str, hf_eval_link: str):
    """Load the train and eval HF datasets."""
    train = load_dataset(hf_train_link, split="train")
    eval = load_dataset(hf_eval_link, split="train")
    return train, eval


def build_message(row):
    """Convert row's question/answer columns into a chat-format message list."""
    return [
        {"role": "user", "content": row["question"]},
        {"role": "assistant", "content": row["answer"]},
    ]


def format_chat(tokenizer, dataset):
    """Apply the model's chat template to every row."""

    def build_prompt(row):
        messages = build_message(row)
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": prompt}

    return dataset.map(build_prompt)