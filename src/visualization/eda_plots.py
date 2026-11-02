from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from src.data.dataset import load_cub, get_class_names

OUT = Path("outputs/eda")


def plot_class_distribution(ds, class_names, name):
    labels = ds["label"]
    counts = Counter(labels)
    values = sorted(counts.values())
    print(f"[{name}] classes={len(counts)} min={min(values)} max={max(values)} "
          f"mean={sum(values)/len(values):.1f}")

    plt.figure(figsize=(8, 4))
    plt.hist(values, bins=20, edgecolor="black")
    plt.title(f"Images per class ({name})")
    plt.xlabel("images in class")
    plt.ylabel("number of classes")
    plt.tight_layout()
    plt.savefig(OUT / f"class_dist_{name}.png", dpi=150)
    plt.close()


def collect_image_stats(ds, name):
    rows = []
    for i in tqdm(range(len(ds)), desc=f"image stats {name}"):
        try:
            img = ds[i]["image"]
            w, h = img.size
            rows.append({"idx": i, "w": w, "h": h, "mode": img.mode, "ok": True})
        except Exception as e:
            rows.append({"idx": i, "w": None, "h": None, "mode": None, "ok": False})
            print(f"corrupted image at {i}: {e}")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"image_stats_{name}.csv", index=False)
    print(f"[{name}] modes:\n{df['mode'].value_counts()}")
    print(f"[{name}] size describe:\n{df[['w', 'h']].describe()}")
    return df


def plot_image_stats(df, name):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].scatter(df["w"], df["h"], s=4, alpha=0.4)
    ax[0].set_title(f"Width vs height ({name})")
    ax[0].set_xlabel("width")
    ax[0].set_ylabel("height")
    ax[1].hist(df["w"] / df["h"], bins=40, edgecolor="black")
    ax[1].set_title("Aspect ratio (w/h)")
    plt.tight_layout()
    plt.savefig(OUT / f"image_stats_{name}.png", dpi=150)
    plt.close()


def plot_samples(ds, class_names, name, n=8):
    fig, axes = plt.subplots(2, n // 2, figsize=(14, 7))
    for ax, i in zip(axes.ravel(), range(0, len(ds), len(ds) // n)):
        item = ds[i]
        ax.imshow(item["image"])
        label = item["label"]
        ax.set_title(class_names[label] if isinstance(label, int) else label, fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(OUT / f"samples_{name}.png", dpi=150)
    plt.close()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train, test = load_cub()
    class_names = get_class_names(train)
    print(f"train={len(train)} test={len(test)} classes={len(class_names)}")

    for name, ds in [("train", train), ("test", test)]:
        plot_class_distribution(ds, class_names, name)
        df = collect_image_stats(ds, name)
        plot_image_stats(df[df["ok"]], name)
        plot_samples(ds, class_names, name)


if __name__ == "__main__":
    main()