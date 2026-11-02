# From Pixels to Language - CSE 485/488 Deep Learning

Environment and dataset setup: see [SETUP.md](SETUP.md).

## Task 1: Classification

Train (hyperparameters live in `configs/`, any of them can be overridden from the CLI):

    python -m src.cli train --task classification --config configs/cls_scratch.yaml

Configs:

| Config | Model | What trains |
|---|---|---|
| `configs/cls_scratch.yaml` | 5-block CNN, random init (baseline) | everything |
| `configs/cls_transfer_frozen.yaml` | ResNet-18, ImageNet weights | only the 200-class head (feature extraction) |
| `configs/cls_transfer.yaml` | ResNet-18, ImageNet weights | everything; encoder at 10x lower lr (fine-tuning) |

Quick smoke test (3 batches per epoch, 2 epochs):

    python -m src.cli train --task classification --config configs/cls_scratch.yaml --epochs 2 --limit-batches 3 --run-name smoke

The best checkpoint (by validation accuracy) is saved to `outputs/checkpoints/<run_name>.pt`
and the training curves to `outputs/plots/`.

Final evaluation on the test set (metrics JSON + confusion matrix in `outputs/eval/`):

    python -m src.cli evaluate --task classification --checkpoint outputs/checkpoints/<run_name>.pt

Inference on one image (top-5 species):

    python -m src.cli predict --task classification --image path/to/bird.jpg --checkpoint outputs/checkpoints/<run_name>.pt

## Experiment tracking

Every run logs its config, per-epoch metrics, curves and checkpoint to MLflow. Browse them with:

    mlflow ui --backend-store-uri sqlite:///mlflow.db
