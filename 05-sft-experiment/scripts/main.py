"""LoRA SFT on ArXiv Q&A dataset"""

import os
import json
import math
import random
from pathlib import Path
from dotenv import load_dotenv

import torch
import comet_ml
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer


def setup_comet():
    load_dotenv()
    COMET_API_KEY = os.getenv("COMET_API_KEY")
    if COMET_API_KEY:
        os.environ["COMET_API_KEY"] = COMET_API_KEY
    os.environ["COMET_PROJECT_NAME"] = "week5-sft"


def load_baseline_config():
    config_path = Path("configs/baseline_eval_config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def load_model_and_lora(config):
    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["model_name"],
        max_seq_length=config["data_max_seq_length"],
        dtype=None,
        load_in_4bit=config["model_load_in_4bit"],
    )
    # LoRA adapter
    # Why : For efficiency & speed
    print(f"Adding LoRA (r={config['lora_r']}, alpha={config['lora_alpha']})")
    model = FastLanguageModel.get_peft_model(
        model,
        r=config["lora_r"],
        target_modules=config["lora_target_modules"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        bias=config["lora_bias"],
        use_gradient_checkpointing=config["lora_use_gradient_checkpointing"],
        random_state=config["experiment_seed"],
    )
    return model, tokenizer


def prepare_datasets(tokenizer, config):
    # Dataset prep.
    dataset = load_dataset(config["data_hf_dataset_path"], split="train")
    # Dataset split
    split_dataset = dataset.train_test_split(
        test_size=config["data_eval_size"], seed=config["experiment_seed"]
    )
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    # Build prompt
    def build_prompt(row):
        prompt = tokenizer.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=False,
        )
        return {"text": prompt}

    train_dataset = train_dataset.map(build_prompt)
    eval_dataset = eval_dataset.map(build_prompt)
    return train_dataset, eval_dataset


def create_trainer(model, tokenizer, train_dataset, eval_dataset, config):
    # SFTConfig – Training Configuration
    training_args = SFTConfig(
        # Output & Logging 
        output_dir=config["logging_output_dir"],
        report_to=[config["logging_report_to"]] if config["logging_report_to"] != "none" else [],
        run_name=config["logging_run_name"],
        # Data Handling 
        dataset_text_field="text",
        packing=config["data_packing"],
        max_seq_length=config["data_max_seq_length"],
        # Batch & Steps 
        per_device_train_batch_size=config["training_batch_size"],
        gradient_accumulation_steps=config["training_gradient_accumulation_steps"],
        warmup_steps=config["training_warmup_steps"],
        num_train_epochs=config["training_num_epochs"],
        # Optimization
        learning_rate=config["training_learning_rates"][0],   #lr =1e-4
        optim=config["training_optimizer"],
        weight_decay=config["training_weight_decay"],
        lr_scheduler_type=config["training_lr_scheduler"],
        max_grad_norm=config["training_max_grad_norm"],
        # Mixed Precision
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        # Logging & Saving Frequency 
        logging_steps=config["logging_logging_steps"],
        save_steps=config["logging_save_steps"],
        # Evaluation & Checkpointing (from config)
        eval_strategy=config["eval_strategy"],
        eval_steps=config["eval_steps"],
        load_best_model_at_end=config["load_best_model_at_end"],
        metric_for_best_model=config["metric_for_best_model"],
        greater_is_better=config["greater_is_better"],
        save_total_limit=config["save_total_limit"],
        # Hugging Face Hub
        push_to_hub=config["hub_push_to_hub"],
        hub_model_id=config["hub_hub_model_id"] if config["hub_push_to_hub"] else None,
        # Reproducibility
        seed=config["experiment_seed"],
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )
    return trainer


def compute_perplexity(model, tokenizer, eval_dataset, max_length=2048):
    """Compute full‑sequence perplexity on the evaluation dataset."""
    model.eval()
    total_loss = 0.0
    for example in eval_dataset:
        text = example["text"]
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False,
        ).to("cuda")
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item()
    avg_loss = total_loss / len(eval_dataset)
    # Why: perplexity computed over over entire seq.
    # for simplication 
    perplexity = math.exp(avg_loss)
    return perplexity


def generate_answers_for_scoring(model, tokenizer, eval_dataset, num_samples=10, seed=42):
    # Why: scoring answers based on eval_rubic gives robust model performance estimation
    """Generate answers for qualitative scoring. Prints Q and A."""
    random.seed(seed)
    indices = random.sample(range(len(eval_dataset)), num_samples)
    for idx in indices:
        example = eval_dataset[idx]
        user_msg = example["messages"][0]["content"]
        # Format prompt (add generation prompt, no assistant message)
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7, do_sample=False)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):]
        print(f"\nQ{idx}: {user_msg}")
        print(f"A: {answer}")
        print("-" * 60)


def main():
    setup_comet()
    config = load_baseline_config()
    model, tokenizer = load_model_and_lora(config)
    train_dataset, eval_dataset = prepare_datasets(tokenizer, config)
    trainer = create_trainer(model, tokenizer, train_dataset, eval_dataset, config)

    # Train
    print("Starting training...")
    trainer.train()

    # Save model
    print(f"Saving to {config['logging_output_dir']}")
    model.save_pretrained(config["logging_output_dir"])
    tokenizer.save_pretrained(config["logging_output_dir"])

    # Perplexity computed on eval dataset
    ppl = compute_perplexity(model, tokenizer, eval_dataset)
    print(f"Full‑sequence perplexity (50 eval samples): {ppl:.2f}")

    # Log value in comet ml
    exp = comet_ml.get_running_experiment()
    if exp is not None:
        exp.log_metric("perplexity", ppl)
        exp.end()
    else:
        print("No Comet experiment running; perplexity not logged.")

    # Run generate answer func.
    generate_answers_for_scoring(
        model, tokenizer, eval_dataset, num_samples=10, seed=config["experiment_seed"]
    )

    print("Training complete!")


if __name__ == "__main__":
    main()