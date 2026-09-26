# POSTMORTEM — Week 6 LoRA Rank Analysis

## 1. What I built
Built LoRA rank analysis pipeline for ArXiv ML abstract QA dataset (filtered 703 pairs) to find inflection point where increasing rank stops improving performance. Tested ranks 8, 16, 32, 64 on Qwen2.5‑1.5B with fixed hyperparameters (LR=2e‑4, 3 epochs).

## 2. What worked
Model training completed without errors for all four ranks. Metrics were properly logged to Comet ML (loss curves, gradient norms, GPU memory). From the result of the analysis, r=16 is the optimal rank with perplexity 6.17 and best eval loss 1.8085.

## 3. What broke
Nothing major broke during this project. Loading config function failed but was fixed using proper config path.

## 4. What I learned
- LoRA rank matters but only up to a point. r=16 gave the best perplexity. r=32 gave the same perplexity but used more memory. With respect to the inflection point, the curve dropped from r=8 (6.21) to r=16 (6.17), flattened at r=32 (6.17) and went up at r=64 (6.30).
- My hypothesis was wrong. I predicted r=8 would win because the dataset was relatively small, but the analysis showed that r=16 performed better than r=8, which shows r=8 couldn't fully capture the patterns/structures in the dataset.
- Data quality is extremely important. Even though r=16 gave the best quantitative performance, the qualitative answers were still generic (scored ~6.2/16). This confirms that abstract‑only data is the real bottleneck, not LoRA rank.
- Memory scales with rank. r=64 used 3.71 GB vs r=16 using 2.42 GB – 53% more memory for no quality gain.

## 5. What I would do differently
- Use better/enriched dataset (full paper sections + reasoning traces) to see if the inflection point changes with better data.

## 6. What carries forward to Week 7
- Optimal rank (r=16) will be used for QLoRA on a 7B model in Week 7.
- Locked baseline config – same seed, split, and hyperparameters.
- Locked qualitative rubric – will be reused for Week 7 evaluation.