# training curves + confusion matrix
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_curves(history, path):
    """Train vs val loss and accuracy per epoch."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a, key, title in [(ax[0], "loss", "Loss"), (ax[1], "acc", "Accuracy")]:
        a.plot(epochs, history[f"train_{key}"], label="train")
        a.plot(epochs, history[f"val_{key}"], label="val")
        a.set_title(title)
        a.set_xlabel("epoch")
        a.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_confusion(cm, path, title="Confusion matrix"):
    """200x200 is too big for labels: show the normalized heatmap only."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    plt.figure(figsize=(8, 7))
    plt.imshow(cm, cmap="viridis")
    plt.colorbar(label="fraction of true class")
    plt.title(title)
    plt.xlabel("predicted class")
    plt.ylabel("true class")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path
