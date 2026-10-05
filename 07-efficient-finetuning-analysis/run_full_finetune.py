"""Full fine-tune of Qwen3-1.7B."""

import os
from pathlib import Path

import comet_ml
import torch
from dotenv import load_dotenv
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel

from shared.config import load_baseline_eval_config
from shared.data import format_chat, prepare_dataset
from shared.memory import peak_snapshot, profile_memory, reset_peak

OUTPUT_DIR = Path(__file__).parent / "outputs" / "sft"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIG = {
    "model_name": "unsloth/Qwen3-1.7B",
    "load_in_4bit": False,
    "max_seq_length": 512,
    "data_hf_repo": "vaadewoyin/llm-engineering-journey-data",
    "batch_size": 2,
    "grad_accum": 4,
    "epochs": 3,
    "lr": 2e-5,
    "warmup_steps": 50,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "optim": "adamw_8bit",
    "lr_scheduler": "linear",
    "eval_steps": 50,
    "save_steps": 100,
    "logging_steps": 5,
    "run_name": "sft-full-ft-qwen3-1.7b",
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

    with profile_memory("model load"):
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=CONFIG["model_name"],
            max_seq_length=CONFIG["max_seq_length"],
            dtype=None,
            load_in_4bit=CONFIG["load_in_4bit"],
            use_gradient_checkpointing=True,
        )

    train_ds = format_chat(tokenizer, train_ds)
    eval_ds = format_chat(tokenizer, eval_ds)

    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        run_name=CONFIG["run_name"],
        report_to=["comet_ml"],
        dataset_text_field="text",
        max_seq_length=CONFIG["max_seq_length"],
        per_device_train_batch_size=CONFIG["batch_size"],
        gradient_accumulation_steps=CONFIG["grad_accum"],
        num_train_epochs=CONFIG["epochs"],
        learning_rate=CONFIG["lr"],
        warmup_steps=CONFIG["warmup_steps"],
        weight_decay=CONFIG["weight_decay"],
        max_grad_norm=CONFIG["max_grad_norm"],
        optim=CONFIG["optim"],
        lr_scheduler_type=CONFIG["lr_scheduler"],
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
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

    exp = comet_ml.get_running_experiment()
    if exp:
        exp.log_metric("peak_allocated_gb", peaks["peak_allocated_gb"])
        exp.log_metric("peak_reserved_gb", peaks["peak_reserved_gb"])
        exp.log_metric("final_train_loss", final_train_loss or 0)
        exp.log_metric("final_eval_loss", final_eval_loss or 0)
        exp.log_metric("best_eval_loss", trainer.state.best_metric or 0)
        exp.log_metric("train_runtime_sec", train_runtime or 0)
        exp.end()

    print(f"\n{'='*60}")
    print(f"FULL FINE-TUNE COMPLETE — {CONFIG['run_name']}")
    print(f"  peak allocated: {peaks['peak_allocated_gb']:.2f} GB")
    print(f"  peak reserved:  {peaks['peak_reserved_gb']:.2f} GB")
    print(f"  final train loss: {final_train_loss:.4f}")
    print(f"  final eval loss:  {final_eval_loss:.4f}")
    print(f"  wall-clock: {train_runtime / 60:.1f} min")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()