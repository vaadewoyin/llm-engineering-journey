# Forest Cover Classifier

### PyTorch MLP · CoverType Dataset · 91.9% Accuracy

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](...)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](...)

> Configurable PyTorch MLP trainer for tabular multiclass classification.
> Clean CLI, reproducible experiments.
> **94.2% Accuracy on 581k real-world samples**

```

## The problem

Predict forest cover type using cartographic variables (elevation, slope, soil type, etc.). This is a real USDA dataset, not a toy.

The goal of this project was not to squeeze out max accuracy but to build the trainer correctly : clean CLI, configurable architecture, reproducible result and proper plots.

## Results

| Config              | Hidden sizes | Test Acc | Notes                                            |
| ------------------- | ------------ | -------- | ------------------------------------------------ |
| Baseline            | [128]        | 82%      | Single layer                                     |
| Best (2 layers)     | [128, 96]    | 91.9%    | Two layers, ReLU                                 |
| Best + lr_scheduler | [128, 96]    | 94.2%    | + ReduceLROnPlateau (factor=0.5, patience=3)     |
```

**Key finding:** Adding a second hidden layer and using a ReduceLROnPlateau scheduler improved accuracy by 12% (from 82% to 94.2%). On this dataset, which mixes both continuous variable with 40 binary soil type indicator, a single layer is not sufficient to capture the interaction between the features.

## Architecture

```
Input(54) → Linear(128) → ReLU → Linear(96) → ReLU → Linear(7)
```

A two‑layer MLP with decreasing hidden sizes (128 → 96), a common pattern - was used for training. The first layer learns complex interactions between the features, while the second layer helps compress those learned interactions into class‑relevant representations.

**Why this two sizes:** The sizes [128, 96] was chosen by manual search over [128], [128, 96], [256], [128, 32]. The chosen sizes gave best validation and test accuracy

**Why ReLU:** The ReLU activation function, a good default for most problems, was chosen for its simplicity and efficiency

## Key Insights

**1. Depth matters more than width on this dataset.**

**2. Stratified split ensures reliable evaluation and helps model generalisation**

**3. Standard scaling was essential for proper convergence as features were on different scales**

## Usage

```bash
# 1.Clone and enter the project
git clone https://github.com/vaadewoyin/llm-engineering-journey.git
cd llm-engineering-journey/week-01-mlp-trainer

# 2. Install dependencies & sync
uv sync

# 3.Run training
uv run mlp-trainer # for default parameters
uv run mlp-trainer  --epochs 20 --hidden-dim 128 -- hidden-dim 96 # custom run example ( 20 epochs, hidden sizes - [128, 96])
```

## What I'd do next

- **Improve CLI output** – use Rich for better‑looking tables and progress bars.
- **Handle class imbalance** - use class weighted loss
- **Per class analysis** - Analyse performance per class and also generate confusion matrix
