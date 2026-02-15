# EQCCT Installation Instructions

This repository contains:
- `eqcctone` (original EQCCT package)
- `eqcctpro` (parallel framework with TensorFlow and SeisBench/PyTorch workflows)

The previous `eqcctpro/environment.yml` mixes TensorFlow and PyTorch CUDA dependencies in one environment, which can produce resolver/runtime conflicts. Use the split environments below instead.

## 1) Clone Repository

```bash
git clone https://github.com/ut-beg-texnet/eqcct.git
cd eqcct
```

## 2) Choose an EQCCTPro Environment

### Option A: TensorFlow workflow (`model_type='eqcct'`)

```bash
cd eqcctpro
conda env create -f environment-tf.yml
conda activate eqcctpro-tf
python -c "import tensorflow as tf; print(tf.__version__)"
```

### Option B: SeisBench/PyTorch workflow (`model_type='seisbench'`)

```bash
cd eqcctpro
conda env create -f environment-torch.yml
conda activate eqcctpro-torch
python -c "import torch, seisbench; print(torch.__version__)"
```

## 3) Run EQCCTPro Code

From the `eqcctpro` directory, run scripts as documented in `eqcctpro/README.md`, for example:

```bash
python experiments/main/run.py
```

## 4) Optional: Install EQCCTOne (Separate Legacy Environment)

`eqcctone` is pinned to an older stack (Python 3.7 + TensorFlow 2.8), so keep it in a dedicated environment.

```bash
cd eqcctone
conda env create -f environment-eqcctone.yml
conda activate eqcctone
pip install -e .
```

## Notes

- Do not install TensorFlow and PyTorch GPU stacks into the same conda environment for this project.
- Do not install `eqcctone` into `eqcctpro-tf` or `eqcctpro-torch`; use the separate `eqcctone` environment.
- If you need all workflows, create and switch between `eqcctpro-tf`, `eqcctpro-torch`, and `eqcctone`.
