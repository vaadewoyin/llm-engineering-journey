# POSTMORTEM — Week 5 SFT Experiment

## 1. What I built
Built lora based sft pipeline for arxiv ml abstract qa dataset and also compared performance of the model (Qwen2.5-1.5B) trained with two different learning rate.

## 2. What worked
Model was finetuned and saved without error. The training results, metrics were properly logged to comet ml, charts and other training details can be viewed on comet dashboard. The two models performance on qualitative evaluation rubric was also done and full result is displayed in readme.

## 3. What broke
Packing = True caused wrong dataset size — 40 examples instead of 901. For unsloth to report to comet ml, comet ml api key must be available as environment variable. Also, different runs under same project need different experiment key.

## 4. What I learned
Packing=True affected model performance due to cross contamination but i thought unsloth handles this well, will have to further check. Also learned that data quality is everything – my abstract‑only dataset produced generic answers that scored low on factual correctness and groundedness. No amount of tuning learning rate can fix shallow data.

## 5. What I would do differently
- Use full paper sections (introduction, methodology, conclusion) instead of just abstracts.
- Add reasoning traces (<think> tags) to teach the model step‑by‑step thinking.
- Use a larger model and higher LoRA rank (r=32) for better capacity.

## 6. What carries forward to Week 6
Baseline config locked, rubric locked, Comet ML project established.
Finding about data quality: abstract‑only data is not enough for real reasoning.