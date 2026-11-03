# Collect the report's figures into report/figures/ (so the folder compiles on its
# own, e.g. on Overleaf) and draw the combined validation-curve plot from MLflow.
# Run from the project root:  python report/make_figures.py
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
from PIL import Image

FIG = Path("report/figures")

COPY = {
    "outputs/eda/samples_train.png": "samples_train.jpg",
    "outputs/eda/class_dist_train.png": "class_dist_train.png",
    "outputs/eda/class_dist_test.png": "class_dist_test.png",
    "outputs/eda/image_stats_train.png": "image_stats_train.png",
    "outputs/eda/quality_train.png": "quality_train.png",
    "outputs/eda/blurriest_train.png": "blurriest_train.jpg",
    "outputs/eda/caption_len_train.png": "caption_len_train.png",
    "outputs/eda/top_words_train.png": "top_words_train.png",
    "outputs/eda/augmentation_examples.png": "augmentation_examples.jpg",
    "outputs/plots/scratch_tuned_curves.png": "scratch_tuned_curves.png",
    "outputs/plots/resnet18_frozen_curves.png": "resnet18_frozen_curves.png",
    "outputs/plots/resnet18_finetune_curves.png": "resnet18_finetune_curves.png",
    "outputs/tuning/grid_heatmap.png": "grid_heatmap.png",
    "outputs/tuning/best_so_far.png": "tuning_best_so_far.png",
    "outputs/eval/scratch_tuned_confusion.png": "scratch_tuned_confusion.png",
    "outputs/eval/resnet18_finetune_confusion.png": "resnet18_finetune_confusion.png",
    "outputs/gradcam/comparison_val.png": "gradcam_comparison.jpg",
    "outputs/gradcam/scratch_tuned_val_correct_wrong.png": "gradcam_scratch_tuned.jpg",
    "outputs/gradcam/resnet18_finetune_val_correct_wrong.png": "gradcam_resnet18_finetune.jpg",
}

RUNS = {
    "scratch": "Scratch CNN (default)",
    "scratch_tuned": "Scratch CNN (tuned, 60 ep.)",
    "resnet18_frozen": "ResNet-18 frozen",
    "resnet18_finetune": "ResNet-18 fine-tuned",
}


def plot_val_curves(path):
    """Validation accuracy per epoch of the final classification runs, one figure."""
    client = mlflow.MlflowClient()
    runs = mlflow.search_runs(experiment_names=["classification"])
    plt.figure(figsize=(7, 4))
    for name, label in RUNS.items():
        run_id = runs.loc[runs["tags.mlflow.runName"] == name, "run_id"].iloc[0]
        history = sorted((m.step, m.value) for m in client.get_metric_history(run_id, "val_accuracy"))
        plt.plot([s for s, _ in history], [v for _, v in history], label=label)
    plt.xlabel("epoch")
    plt.ylabel("validation accuracy")
    plt.title("Validation accuracy per epoch")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    for src, dst in COPY.items():
        if dst.endswith(".jpg"):  # photo grids: JPEG is ~10x smaller than PNG
            Image.open(src).convert("RGB").save(FIG / dst, quality=88)
        else:
            shutil.copy(src, FIG / dst)
    plot_val_curves(FIG / "val_curves_all.png")
    print(f"{len(COPY) + 1} figures in {FIG}")


if __name__ == "__main__":
    main()
