# DESIGN.md

## 1. Problem

Fine-tuning has three common methods with different memory and quality trade-offs — full fine-tuning, LoRA, and QLoRA. This project sets out to determine which of these methods allows me to reliably train a small instruction-finetuned model on free hardware (Kaggle T4).

**Decision this project feeds:** which training method the DPO phase uses for the served Qwen3-1.7B.

## 2. Components

- `run_full_finetune.py` — Full fine-tune of Qwen3-1.7B. Produces `outputs/sft/`.
- `run_lora.py` — LoRA r=16 on Qwen3-1.7B. Same data. Produces `outputs/lora_r16/`.
- `run_qlora.py` — QLoRA-NF4 on Qwen3-8B. Produces `outputs/qlora_nf4/`.
- `baseline_eval_config.json` — locked eval config such as seed 45, train/val split 0.95, eval size, etc.

## 3. Component Communication

The fine-tuning scripts (`run_full_finetune.py`, `run_lora.py`, `run_qlora.py`) are all run independently, and each loads the eval config `baseline_eval_config.json` to evaluate the finetuned model. For each script, the training config is placed at the top of the script in a `CONFIG` dict right after the imports. All three write to the same Comet project (`concrete-ft-phase`) with distinct run names, so the curves are directly comparable. Outputs go to separate subdirectories under `outputs/`.

## 4. Failure Modes

- **NaN loss within a few steps.** Cause: the code silently defaults to bf16, or a bad LR. Mitigation: force `fp16=True`, `bf16=False`, log loss at step 10, abort if NaN.
- **OOM.** Cause: full FT of 1.7B is borderline on a 16 GB T4. Mitigation: batch size 2, using gradient accumulation and gradient checkpointing.
- **Chat template mismatch.** Cause: training succeeds but outputs are garbage because the data wasn't formatted in Qwen3's chat format. Mitigation: assert the formatted example follows Qwen ChatML format.

## 5. Definition of Done

- Three checkpoints exist: `outputs/sft/`, `outputs/lora_r16/`, `outputs/qlora_nf4/`
- Comet shows all three runs in project `concrete-ft-phase`, each with loss and LR curves
- A 3-way table exists in the README with real values for: trainable params, peak VRAM, wall-clock, final train loss, final val loss
- Detailed README giving a full overview of the entire project with metrics, findings, and instructions for replication

## 6. Production Boundaries

### What is deterministic (LLM never decides)

The training pipeline is fully deterministic Python.

### Human inspection point

Comet dashboard (loss, LR, VRAM, wall-clock per run) plus the `outputs/` checkpoints.

### State representation

JSONL training log + Comet + a checkpoint every 100 steps. No hidden state. Everything is inspectable as plain files.

### Serial vs parallel

The three runs execute in parallel across separate Kaggle/Modal sessions. Justified because they share no state — different models, different output directories.

## 7. Pre-Build Questions

**Q: Why only 1.7B carries SFT → LoRA → DPO? What fails on a 16 GB T4 if you try 8B? (Rule 37)**

**A:** Full fine-tuning of 8B needs fp16 weights, which is around 16 GB already, not accounting for gradients, activations, and optimizer states (which can be 2–3× model weight size). All these show that a T4 is not sufficient for such a model, but a 1.7B can fit a T4 using small batch size and gradient checkpointing.

**Q: Write your prediction for NF4's memory and quality profile on Qwen3-8B *before* you run it. What number would surprise you?**

**A:** Prediction: 4-bit weights ≈ 4 GB, LoRA adapters fp16 < 100 MB, LoRA optimizer tiny, activations ~2–3 GB with gradient checkpointing. Peak ~7–8 GB. Quality: the val loss should be within 1–2% of LoRA r=16 loss. A peak value of 14 GB would be surprising.

**Q: T4 has no native bf16 — what does fp16 compute risk, and what's the mitigation?**

**A:** fp16 has a narrow exponent range compared to bf16, which has the same precision as fp32. The small range of fp16 means gradients can overflow or underflow during training, which can lead to NaN loss. Mitigation: on T4, we use mixed precision through the Unsloth training stack, and also with gradient clipping (`max_grad_norm=1.0`) as an additional stability safeguard.

**Q: What does `@profile_memory` measure at each stage, and where can it mislead?**

**A:** It measures VRAM at different points during each run. It can be misleading because PyTorch keeps both allocated and reserved memory, and recording just one of these can be misleading because it doesn't give a full overview of memory usage.

**Q: Why is FP4 not run? What does that cost you, and where is the substitute documented?**

**A:** FP4 requires Hopper/Blackwell tensor cores; T4 doesn't have them. Cost: I can't claim "NF4 beats FP4 for this workload" — only "NF4 works on free hardware." Substitute: the NF4 run itself documents the memory/quality profile.

## 8. Known Limitations

### What this system does not handle

1. **Models larger than 8B.** Every config is sized to fit the Kaggle T4's 16 GB, and the 8B QLoRA run only fits because of quantization.
2. **Sequences longer than the training default.** Examples with length greater than max length get truncated, behaviour on longer inputs is untested.

### What would break it, that I already know about

**Kaggle session timeout mid-run.** Kaggle sesssion time out mid-run may happen before the last checkpoint is saved, and it won't restart on its own.
