# Chest X-ray Classification Assignment

This repository contains a polished version of the Deep Learning assignment for chest X-ray classification. It includes a reusable training script, model definitions, and utilities for evaluating performance on a 64x64 chest X-ray dataset.

## Files

- `train.py` - main training entry point with CLI support for model choice and regularization strategy
- `models.py` - model definitions for an MLP and a small convolutional neural network
- `utils.py` - data loading, training loop, evaluation, and plotting utilities
- `requirements.txt` - minimal Python dependencies for running the project
- `README.md` - this file
- `.gitignore` - recommended ignores for dataset and temporary files

## Setup

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

2. Place the dataset in the assignment directory or point the script to the dataset root:

```bash
Assignment1\chest_xray_64
```

The script supports a dataset that contains `train/` and `val/` subfolders. If those folders are not present, it will split the data automatically.

## Run training

Train the CNN with baseline settings:

```bash
python train.py --data-dir chest_xray_64 --model cnn --strategy baseline --epochs 30 --batch-size 32 --plot
```

Train the MLP instead:

```bash
python train.py --data-dir chest_xray_64 --model mlp --strategy baseline --epochs 30 --batch-size 32
```

Train with L2 weight decay:

```bash
python train.py --data-dir chest_xray_64 --model cnn --strategy weight_decay --weight-decay 0.0075
```

Train with early stopping:

```bash
python train.py --data-dir chest_xray_64 --model cnn --strategy early_stopping --epochs 30
```

## Output

Results are saved under the `output/` directory by default:

- `*.pth` - trained model weights
- `metrics.txt` - final loss and accuracy
- `loss_curve.png` / `accuracy_curve.png` - saved plots when using `--plot`

## Notes

- The dataset folder is intentionally ignored in version control to keep the GitHub repository lightweight.
- For proper reproducibility, run experiments with the same seed and Torch configuration.
