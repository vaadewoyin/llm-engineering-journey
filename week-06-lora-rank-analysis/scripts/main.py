"""LoRA SFT on ArXiv Q&A dataset"""

import os
import json
import math
from pathlib import Path
from dotenv import load_dotenv
import torch
import comet_ml
from unsloth import FastLanguageModel
from datasets import load_dataset, Dataset
from trl import SFTConfig, SFTTrainer


def setup_comet():
    load_dotenv()
    COMET_API_KEY = os.getenv("COMET_API_KEY")
    if COMET_API_KEY:
        os.environ["COMET_API_KEY"] = COMET_API_KEY
    os.environ["COMET_PROJECT_NAME"] = "week6-lora-rank"


def load_baseline_config():
    config_path = Path("configs/baseline_eval_config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def prepare_dataset(config):
    # Load original dataset
    old_dataset = load_dataset(config["data_hf_dataset_path"], split="train")

    # Filter words
    filter_words = ["authors", "researchers", "study", "paper", 
                    "the authors", "the researchers", "the paper"]
    filtered_data = []
    for data in old_dataset:
        content = data["messages"][0]["content"] + " " + data["messages"][1]["content"]
        if not any(w in content for w in filter_words):
            filtered_data.append(data)

    # Convert to HF Dataset
    dataset = Dataset.from_list(filtered_data)
    print(f"Filtered dataset size: {len(dataset)}")

    # Train/val split
    split_dataset = dataset.train_test_split(
        test_size=config["data_eval_size"],
        seed=config["experiment_seed"]
    )
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    return train_dataset , eval_dataset


def compute_perplexity(model, tokenizer, eval_dataset, max_length=2048):
    model.eval()
    total_loss = 0.0
    for example in eval_dataset:
        text = example["text"]
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False
        ).to("cuda")
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item()
    avg_loss = total_loss / len(eval_dataset)
    return math.exp(avg_loss)

def train_rank(rank, config, train_dataset, eval_dataset):
    print(f"\n{'='*60}")
    print(f" Training LoRA rank = {rank}")
    print(f"{'='*60}")

    # Load base model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["model_name"],
        max_seq_length=config["data_max_seq_length"],
        dtype=None,
        load_in_4bit=config["model_load_in_4bit"],
    )

    # Apply LoRA with rank
    model = FastLanguageModel.get_peft_model(
        model,
        r=rank,
        target_modules=config["lora_target_modules"],
        lora_alpha=rank,                     # alpha = rank
        lora_dropout=config["lora_dropout"],
        bias=config["lora_bias"],
        use_gradient_checkpointing=config["lora_use_gradient_checkpointing"],
        random_state=config["experiment_seed"],
    )

    # Format dataset with chat template
    def build_prompt(row):
        prompt = tokenizer.apply_chat_template(
            row["messages"],
            tokenize=False,
            add_generation_prompt=False
        )
        return {"text": prompt}

    train_ds = train_dataset.map(build_prompt)
    eval_ds = eval_dataset.map(build_prompt)

    # Training args 
    training_args = SFTConfig(
        output_dir=f"outputs/rank-{rank}",
        report_to=["comet_ml"],
        run_name=f"lora-rank-{rank}",
        dataset_text_field="text",
        packing=config["data_packing"],
        max_seq_length=config["data_max_seq_length"],
        per_device_train_batch_size=config["training_batch_size"],
        gradient_accumulation_steps=config["training_gradient_accumulation_steps"],
        warmup_steps=config["training_warmup_steps"],
        num_train_epochs=config["training_num_epochs"],                 
        learning_rate= 2e-4, #config["training_learning_rates"],                  
        optim=config["training_optimizer"],
        weight_decay=config["training_weight_decay"],
        lr_scheduler_type=config["training_lr_scheduler"],
        max_grad_norm=config["training_max_grad_norm"],
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=config["logging_logging_steps"],
        save_steps=config["eval_steps"],                     
        eval_strategy=config["eval_strategy"],
        eval_steps=config["eval_steps"],
        load_best_model_at_end=config["load_best_model_at_end"],
        metric_for_best_model=config["metric_for_best_model"],
        greater_is_better=config["greater_is_better"],
        save_total_limit=config["save_total_limit"],
        seed=config["experiment_seed"],
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=training_args,
    )

    # Reset peak memory stats
    torch.cuda.reset_peak_memory_stats()
    trainer.train()
    peak_memory = torch.cuda.max_memory_allocated() / 1024**3

    # Best eval loss (from the loaded best model)
    best_eval_loss = trainer.state.best_metric

    # Compute perplexity on the best model
    ppl = compute_perplexity(model, tokenizer, eval_ds)

    # Save final model 
    model.save_pretrained(f"outputs/rank-{rank}/final")
    tokenizer.save_pretrained(f"outputs/rank-{rank}/final")

    # Log to Comet
    exp = comet_ml.get_running_experiment()
    if exp:
        exp.log_metric("best_eval_loss", best_eval_loss)
        exp.log_metric("perplexity", ppl)
        exp.log_metric("peak_memory_gb", peak_memory)
        exp.log_metric("rank", rank)
        exp.end()

    return {
        "rank": rank,
        "best_eval_loss": best_eval_loss,
        "perplexity": ppl,
        "peak_memory_gb": peak_memory,
        "training_time": trainer.state.log_history[-1].get("train_runtime", None) if trainer.state.log_history else None,
    }


if __name__ == "__main__":

    setup_comet()
    config = load_baseline_config()
    train_dataset, eval_dataset =  prepare_dataset(config)

    # Run sweep 
    ranks = [8, 16, 32, 64]
    results = []
    for rank in ranks:
        result = train_rank(rank, config, train_dataset, eval_dataset)
        results.append(result)

    # print quick summary
    print("\n" + "="*60)
    print("RANK SWEEP COMPLETE")
    print("="*60)
    for r in results:
        print(f"r={r['rank']}: perplexity={r['perplexity']:.2f}, best_eval_loss={r['best_eval_loss']:.4f}, peak_memory={r['peak_memory_gb']:.2f}GB")
    print("="*60)