"""LoRA r=16 on Qwen3-1.7B."""

import os
from pathlib import Path
import unsloth
import comet_ml
import torch
from dotenv import load_dotenv
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel

from shared.config import load_baseline_eval_config
from shared.data import format_chat, prepare_dataset
from shared.memory import peak_snapshot, profile_memory, reset_peak

OUTPUT_DIR = Path(__file__).parent / "outputs" / "lora_r16"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIG = {
    "model_name": "unsloth/Qwen3-1.7B",
    "load_in_4bit": False,
    "max_seq_length": 512,
    "lora_r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.0,
    "lora_target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    "data_hf_repo": "vaadewoyin/llm-engineering-journey-data",
    "batch_size": 2,
    "grad_accum": 4,
    "epochs": 3,
    "lr": 1e-4,
    "warmup_steps": 50,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "optim": "adamw_8bit",
    "lr_scheduler": "linear",
    "eval_steps": 50,
    "save_steps": 100,
    "logging_steps": 5,
    "run_name": "lora-r16-qwen3-1.7b",
    "comet_project": "concrete-ft-phase",
}


def setup_comet(project_name: str) -> None:
    load_dotenv()
    api_key = os.getenv("COMET_API_KEY")
    if api_key:
        os.environ["COMET_API_KEY"] = api_key
    os.environ["COMET_PROJECT_NAME"] = project_name


def main():
    setup_comet(CONFIG["comet_project"])
    eval_cfg = load_baseline_eval_config()

    train_ds, eval_ds = prepare_dataset(CONFIG["data_hf_repo"])

    with profile_memory("base model load"):
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=CONFIG["model_name"],
            max_seq_length=CONFIG["max_seq_length"],
            dtype=torch.bfloat16,
            load_in_4bit=CONFIG["load_in_4bit"],
        )

    with profile_memory("apply LoRA"):
        model = FastLanguageModel.get_peft_model(
            model,
            r=CONFIG["lora_r"],
            lora_alpha=CONFIG["lora_alpha"],
            lora_dropout=CONFIG["lora_dropout"],
            target_modules=CONFIG["lora_target_modules"],
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=eval_cfg.experiment_seed,
        )

    train_ds = format_chat(tokenizer, train_ds)
    eval_ds = format_chat(tokenizer, eval_ds)

    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        run_name=CONFIG["run_name"],
        report_to=["comet_ml"],
        dataset_text_field="text",
        max_length=CONFIG["max_seq_length"],
        per_device_train_batch_size=CONFIG["batch_size"],
        gradient_accumulation_steps=CONFIG["grad_accum"],
        num_train_epochs=CONFIG["epochs"],
        learning_rate=CONFIG["lr"],
        warmup_steps=CONFIG["warmup_steps"],
        weight_decay=CONFIG["weight_decay"],
        max_grad_norm=CONFIG["max_grad_norm"],
        optim=CONFIG["optim"],
        lr_scheduler_type=CONFIG["lr_scheduler"],
        fp16=False,
        bf16=True,
        logging_steps=CONFIG["logging_steps"],
        save_steps=CONFIG["save_steps"],
        eval_strategy="steps",
        eval_steps=CONFIG["eval_steps"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        seed=eval_cfg.experiment_seed,
    )

    with profile_memory("trainer init"):
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            args=training_args,
        )

    reset_peak()
    trainer.train()
    peaks = peak_snapshot()

    history = trainer.state.log_history
    eval_losses = [h["eval_loss"] for h in history if "eval_loss" in h]
    final_eval_loss = eval_losses[-1] if eval_losses else None
    final_train_loss = history[-1].get("train_loss")
    train_runtime = history[-1].get("train_runtime")

    model.save_pretrained(str(OUTPUT_DIR / "final"))
    tokenizer.save_pretrained(str(OUTPUT_DIR / "final"))

    adapter_file = OUTPUT_DIR / "final" / "adapter_model.safetensors"
    adapter_size_mb = (
        adapter_file.stat().st_size / 1024**2 if adapter_file.exists() else None
    )

    exp = comet_ml.get_running_experiment()
    if exp:
        exp.log_metric("peak_allocated_gb", peaks["peak_allocated_gb"])
        exp.log_metric("peak_reserved_gb", peaks["peak_reserved_gb"])
        exp.log_metric("final_train_loss", final_train_loss or 0)
        exp.log_metric("final_eval_loss", final_eval_loss or 0)
        exp.log_metric("best_eval_loss", trainer.state.best_metric or 0)
        exp.log_metric("train_runtime_sec", train_runtime or 0)
        exp.log_metric("adapter_size_mb", adapter_size_mb or 0)
        exp.end()

    print(f"\n{'='*60}")
    print(f"LORA r=16 COMPLETE — {CONFIG['run_name']}")
    print(f"  peak allocated: {peaks['peak_allocated_gb']:.2f} GB")
    print(f"  peak reserved:  {peaks['peak_reserved_gb']:.2f} GB")
    print(f"  adapter size:   {adapter_size_mb:.1f} MB")
    print(f"  final train loss: {final_train_loss:.4f}")
    print(f"  final eval loss:  {final_eval_loss:.4f}")
    print(f"  wall-clock: {train_runtime / 60:.1f} min")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()