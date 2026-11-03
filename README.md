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

One table of every evaluated model (config, best val accuracy, test metrics) in
`outputs/eval/model_comparison.csv`:

    python -m src.cli compare --task classification

Inference on one image (top-5 species):

    python -m src.cli predict --task classification --image path/to/bird.jpg --checkpoint outputs/checkpoints/<run_name>.pt

Grad-CAM heatmaps (correct vs wrong predictions per model; with 2+ checkpoints also a
side-by-side comparison on the same images), saved to `outputs/gradcam/`:

    python -m src.cli explain --task classification --checkpoint outputs/checkpoints/scratch.pt outputs/checkpoints/resnet18_finetune.pt

## Hyperparameter tuning

Grid search and random search over the scratch CNN, same budget and bounds
(search space in `configs/tune_scratch.yaml`):

    python -m src.cli tune --task classification --config configs/tune_scratch.yaml --method both

Per-trial results go to `outputs/tuning/{grid,random}_results.csv`, the comparison
(best score, trials, time to best) to `outputs/tuning/comparison.csv`, plus
`best_so_far.png` and `grid_heatmap.png`. Rerunning the command skips finished
trials, so an interrupted search resumes where it stopped.

## Experiment tracking

Every run logs its config, per-epoch metrics, curves and checkpoint to MLflow. Browse them with:

    mlflow ui --backend-store-uri sqlite:///mlflow.db

## Report

The report is `report/report.docx` (a LaTeX version is in `report.tex`). Rebuild the
figures from `outputs/` and MLflow, then the Word file (needs the `docx` npm package;
open it in Word once to fill in the table of contents):

    python report/make_figures.py
    node report/make_docx.js
