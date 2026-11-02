from pathlib import Path

import matplotlib.pyplot as plt

from src.data.dataset import load_cub
from src.data.transforms import get_train_transform

OUT = Path("outputs/eda")


def unnormalize(t):
    import torch
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (t * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train, _ = load_cub()
    tf = get_train_transform(128)
    rows = [0, 1500, 3000, 4500]
    fig, axes = plt.subplots(len(rows), 5, figsize=(12, 10))
    for r, idx in enumerate(rows):
        img = train[idx]["image"].convert("RGB")
        axes[r, 0].imshow(img)
        axes[r, 0].set_title("original", fontsize=8)
        for c in range(1, 5):
            axes[r, c].imshow(unnormalize(tf(img)))
            axes[r, c].set_title(f"aug {c}", fontsize=8)
        for ax in axes[r]:
            ax.axis("off")
    plt.tight_layout()
    plt.savefig(OUT / "augmentation_examples.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()