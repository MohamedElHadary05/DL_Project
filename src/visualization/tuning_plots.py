# grid vs random search plots
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_best_so_far(results, path):
    """Best val accuracy found so far vs. compute spent, one line per method."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    for method, df in results.items():
        df = df.sort_values("trial")
        plt.step(df.minutes.cumsum(), df.best_val_accuracy.cummax(), where="post",
                 marker="o", label=f"{method} ({len(df)} trials)")
    plt.xlabel("cumulative training time (minutes)")
    plt.ylabel("best val accuracy so far")
    plt.title("Grid search vs random search")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_grid_heatmap(df, path, x="lr", y="weight_decay"):
    """Val accuracy for every (x, y) pair of the grid."""
    table = df.pivot_table(index=y, columns=x, values="best_val_accuracy")
    plt.figure(figsize=(6, 4))
    plt.imshow(table.values, cmap="viridis", origin="lower")
    plt.colorbar(label="best val accuracy")
    plt.xticks(range(len(table.columns)), [f"{v:g}" for v in table.columns])
    plt.yticks(range(len(table.index)), [f"{v:g}" for v in table.index])
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            plt.text(j, i, f"{table.values[i, j]:.3f}", ha="center", va="center", color="w")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title("Grid search results")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path
