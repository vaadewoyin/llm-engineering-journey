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
3. Model fine-tuning gets interrupted during the process — triggered by Kaggle session timeout or manual stop. Recovery: restart the run for that rank from scratch (since rank sweep runs are short, checkpoint recovery is not needed).

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
The entire rank sweep is done serially. Each rank (r=8, 16, 32, 64) is trained one after the other, not in parallel. This avoids memory contention and makes the logs easier to follow. Since each run takes ~45 minutes, the total time is about 3 hours, which is acceptable for a single Kaggle session.

## 7. Pre-Build Questions

**Before running: which rank do you predict will win, and why? Write the prediction with a reason.**

r=8 should have better performance than the remaining ranks (r=16, r=32, r=64). Reason: rank in LoRA generally represents adapter capacity. The higher the rank, the more the capacity of the adapters and the better the fine-tuning result technically. However, there is also more risk of overfitting. In our dataset with less than 1000 Q&A pairs that are obtained from relatively short abstracts, higher values of rank are more likely to overfit owing to the simple nature of the dataset. r=8, being smaller than the other rank values, should have less overfit than the remaining.

**What is the mathematical relationship between rank r, alpha, and the effective learning rate?**

The update is approximately: old weights (frozen) + (alpha / r) * BA. So the effective learning rate is scaled by alpha / r. With alpha = rank, the scaling factor is 1.

**You get r=64 perplexity = r=32 perplexity. What does that tell you about the rank-quality curve for this task?**

It tells me that the rank-quality curve has flattened. Increasing rank beyond 32 does not improve perplexity – the model has reached its capacity for this dataset. The inflection point is at or before r=32.

**LoRA adds adapter matrices. Where exactly do they attach in a transformer? Why those layers?**

Adapters attach in the attention layer and MLP layer. Attention layers because that is where the model gets contextual understanding of each token, so we typically attach it there. We also attach in the MLP layer because that is where information about what the token means is used to do the actual prediction of the next token.

**Your comparison table has 4 runs. A recruiter asks: what is the engineering conclusion? One sentence.**

The optimal rank for this dataset and model is r=8, as higher ranks show no significant improvement and increase memory usage.

## 8. Known Limitations

**What does this system not handle?**
1. The system does not handle fine-tuning of a large model.
2. It does not generate reasoning traces (Thinking SFT) or agentic tool-use patterns.

**What would break it that you are aware of right now?**
1. Loading a large model will break the system because it will lead to an out-of-memory error.
2. Missing or incorrect chat template in the tokenizer will prevent the model from learning proper turn boundaries.