# DESIGN.md

## 1. Problem
To find the inflection point where increasing LoRA rank parameter ceases to improve fine-tuning performance on the filtered 703‑pair dataset.

## 2. Components
- `main.py` — Runs LoRA rank analysis for r=8, 16, 32, 64 and logs results.
- `baseline_eval_config.json` — Contains the config used in fine-tuning the model, such as data split, random seed, etc. (Locked - same as Week 5).

## 3. Component Communication
`main.py` loads config from `configs/baseline_eval_config.json`; data is loaded from the Hugging Face dataset using the path in the same config file. The filtered 703‑pair dataset is used. The script then runs training for each rank, logs metrics to Comet ML, and saves the final model for each rank.

## 4. Failure Modes
1. OOM error can occur if the model is too large for available GPU memory — triggered when batch size × sequence length exceeds VRAM. Recovery: reduce batch size, enable gradient checkpointing.
2. Logging errors due to improper Comet ML setup — triggered when API key is missing or network is blocked. Recovery: set `report_to="none"` and log locally.
3. Model fine-tuning gets interrupted during the process — triggered by Kaggle session timeout or manual stop. Recovery: because each run is short (~45 min) and the total sweep (~3 hours) fits within a Kaggle session, checkpointing is omitted to keep the script simple. If interrupted, only the affected rank needs to be restarted.

## 5. Definition of Done
1. Rank vs perplexity curve plotted.
2. Comet ML charts and logs for the four runs in the same project. Logs include loss curves, learning rate curve, gradient norms, and GPU memory per run.
3. Full comparison table with all metrics filled (rank, eval loss, perplexity, training time, peak GPU memory, adapter file size).
4. Qualitative evaluation (10 examples) for the best‑performing rank using the locked rubric from Week 5.

## 5a. Tests I Will Write
- `test_environment.py`: To test that the environment works with all needed functionality (GPU, libraries, etc.) operational.
- `test_rank.py`: To test that each rank runs without error and produces a valid adapter file.

## 6. Production Boundaries

### What is deterministic (LLM never decides)
1. Loading the data and config is done using Python code.
2. Logging to Comet ML is also done using Python.
3. Evaluation.
4. The learning rate value (fixed at 3e‑4).
5. The rank value (fixed per run: 8, 16, 32, 64). 
6. The train/validation split (fixed by seed=42).

### Human inspection point
To check what the system did, the user can check the Comet ML dashboard for all logging info, which includes charts for loss & training curve, and all other metrics. The final models and adapter files are saved in `outputs/rank-{r}/final_model/` for inspection.

### State representation
The dataset is stored as a JSONL file (`filtered_703.jsonl`), the configuration is stored in `baseline_eval_config.json`. Training metrics (loss curves, gradient norms, GPU memory) are logged to Comet ML. The final model and adapter for each rank are saved as files in `outputs/rank-{r}/final_model/`. The results table (rank vs perplexity) is written as a Markdown file for easy viewing.

### Serial vs parallel
The entire rank sweep is done serially. Each rank (r=8, 16, 32, 64) is trained one after the other, not in parallel. This avoids memory contention and makes the logs easier to follow. 

## 7. Pre-Build Questions

**Before running: which rank do you predict will win, and why? Write the prediction with a reason.**

I predict r=8 will have the lowest perplexity because higher ranks (r=32, r=64) are more likely to overfit on our small dataset of 703 Q&A pairs derived from short abstracts. 

**What is the mathematical relationship between rank r, alpha, and the effective learning rate?**

The update is approximately: old weights (frozen) + (alpha / r) * BA. So the effective learning rate is scaled by alpha / r. With alpha = rank, the scaling factor is 1.

**You get r=64 perplexity = r=32 perplexity. What does that tell you about the rank-quality curve for this task?**

It tells me that the rank-quality curve has flattened. Increasing rank beyond 32 does not improve perplexity – the model has reached its capacity for this dataset. The inflection point is at or before r=32.

**LoRA adds adapter matrices. Where exactly do they attach in a transformer? Why those layers?**

In this implementation, LoRA adapters are attached to seven modules: `q_proj`, `k_proj`, `v_proj`, `o_proj` in the attention layers, and `gate_proj`, `up_proj`, `down_proj` in the MLP layers. These are chosen because they cover the key projection and feed‑forward transformations that are most influential for task adaptation. Targeting both attention and MLP gives more capacity for learning domain‑specific patterns, while still being efficient.

**Your comparison table has 4 runs. A recruiter asks: what is the engineering conclusion? One sentence.**

The optimal rank for this dataset and model is r=8, as higher ranks show no significant improvement and increase memory usage.

