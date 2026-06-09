# DESIGN.md 

## 1. Problem
Fine-tuning a small language model on Q&A pairs generated from ArXiv ML abstracts to teach the model how to respond to ML questions.

## 2. Components
- `train_sft.py` — For SFT on a small language model.
- `eval.py` — For evaluating the fine-tuned model.
- `baseline_eval_config.json` — Contains the config used in fine-tuning the model, such as data split, random seed, etc.

## 3. Component Communication
`train_sft.py` loads data & config from `data/qa_pairs.jsonl` and `configs/baseline_eval_config.json`, uses it to fine-tune the model, and saves the fine-tuned model to `output/fine_tuned_model`. `eval.py` loads the fine-tuned model and does evaluation.

## 4. Failure Modes
1. OOM error can occur if the model is too large for available GPU memory — triggered when batch size × sequence length exceeds VRAM. Recovery: reduce batch size, enable gradient checkpointing.
2. Logging errors due to improper Comet ML setup — triggered when API key is missing or network is blocked. Recovery: set `report_to="none"` and log locally.
3. Model fine-tuning gets interrupted during the process — triggered by Kaggle session timeout or manual stop. Recovery: reload checkpoint from last save and resume.

## 5. Definition of Done
1. Fine-tuned model stored in `output/`.
2. Comet ML charts and logs for the two LR runs, including loss curves, learning rate curve, gradient norms, and GPU memory per run.
3. Perplexity computed on 50 held-out samples, with the actual number reported.
4. 10 qualitative examples scored against `eval/qualitative_rubric.md`, with scores (1–4 per dimension).
5. Generators used in data pipeline (lazy loading).
6. pytest fixtures with parametrised eval suite passing.

## 5a. Tests I Will Write
- `test_environment.py`: To test that the environment works with all needed functionality (GPU, libraries, etc.) operational.
- `test_checkpoint_recovery.py`: Kills training at step 200, resumes from checkpoint 100, checks loss continuity.
- `test_evaluation.py`: Parametrised eval suite using pytest fixtures.

## 6. Production Boundaries

### What is deterministic
1. Loading the data and config is done using Python code.
2. Logging to Comet ML is also done using Python.
3. Evaluation.
4. The learning rate value (hard-coded per run)
5. When to save a checkpoint (`save_steps=100`)
6. The train/validation split (fixed by seed=42)

### Human inspection point
To check what the system did, the user can check the Comet ML dashboard for all logging info, which includes charts for loss & training curve, and all other metrics. **For checkpoint inspection, open `output/checkpoint-{step}/trainer_state.json` to see the exact step, loss, and learning rate.**

### State representation
Apart from Comet ML logs, model fine-tuning is checkpointed every 100 steps and logged in a JSON file that can be easily accessed to get log details.

### Serial vs parallel
The entire SFT pipeline is done serially, with data & config loaded, and model fine-tuned then saved for further evaluation. As each step depends on the previous, serial processing is most appropriate.

## 7. Unified Assumption + Hypothesis Blocks

### LR = 1e-4
**Assumption:** A lower learning rate will update weights slowly, preserving pretrained knowledge.  
**Risk if wrong:** Training may converge too slowly or get stuck.  
**Hypothesis:** The loss curve will descend smoothly without sharp spikes. Final validation loss ~2.8, perplexity ~25.  


### LR = 3e-4
**Assumption:** A higher learning rate will cause faster initial loss reduction but may destabilise pretrained weights.  
**Risk if wrong:** The model may overfit or produce gradient norm spikes.  
**Hypothesis:** Loss will drop faster in the first 200 steps but may show oscillations. Final validation loss similar or slightly higher than 1e-4, perplexity ~27.  


## 8. Pre-Build Questions
**You are running two learning rate experiments. What do you expect each curve to look like — before running?** 
For the larger LR, I expect the loss curve to decrease faster than the smaller learning rate. The smaller LR curve should show a more steady decrease, while the larger learning rate curve may have more undulations.

**What is the difference between perplexity dropping and the model actually learning to answer questions well?**  
Perplexity dropping shows the model is less confused about choosing the appropriate next token, but this does not necessarily mean the model has learned to answer questions well because the model may have just memorised the answers and is giving them back to us.

**Your training run dies at step 800. You have a checkpoint at step 700. What is your exact recovery procedure?**  
The exact recovery is to load the model checkpoint at step 700 and continue training from this checkpoint.

**What does gradient norm tell you about training stability? What value should alarm you?**  
A smaller gradient norm shows the gradient is more stable, which means training is more stable. A value that is very large (e.g., above 10) or suddenly spikes should alarm you because it indicates unstable training. Values close to 1 are normal and not a concern.

**You evaluate on 50 held-out samples. Is that enough? What would make you trust the evaluation more?** 
50 held‑out samples are a reasonable starting point for evaluation. To trust the evaluation more, I would use a larger eval set and combine perplexity with human scoring using the `qualitative rubric.`

## 9. Known Limitations

**What does this system not handle?**
1. The system does not handle fine-tuning of a large model.
2. It does not generate reasoning traces (Thinking SFT) or agentic tool‑use patterns.

**What would break it that you are aware of right now?**
1. Loading a large model will break the system because it will lead to an out‑of‑memory error.
2. Missing or incorrect chat template in the tokenizer will prevent the model from learning proper turn boundaries.