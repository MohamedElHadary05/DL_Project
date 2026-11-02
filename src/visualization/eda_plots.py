from collections import Counter
from pathlib import Path

import re
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

def caption_eda(ds, name):
    from src.data.dataset import split_captions

    caps_per_image, all_caps = [], []
    for desc in tqdm(ds["description"], desc=f"captions {name}"):
        caps = split_captions(desc)
        caps_per_image.append(len(caps))
        all_caps.extend(caps)

    lengths = [len(c.split()) for c in all_caps]
    words = [w for c in all_caps for w in re.findall(r"[a-z']+", c.lower())]
    vocab = Counter(words)
    rare = [w for w, n in vocab.items() if n == 1]

    print(f"[{name}] images={len(ds)} captions={len(all_caps)}")
    print(f"[{name}] captions/image: {Counter(caps_per_image).most_common(5)}")
    print(f"[{name}] length (words): min={min(lengths)} max={max(lengths)} "
          f"mean={sum(lengths)/len(lengths):.1f}")
    print(f"[{name}] vocab={len(vocab)} words appearing once={len(rare)}")
    print(f"[{name}] top 15 words: {vocab.most_common(15)}")
    print(f"[{name}] unique captions={len(set(all_caps))} "
          f"duplicates={len(all_caps) - len(set(all_caps))}")
    nonascii = [c for c in all_caps if re.search(r"[^\x00-\x7F]", c)]
    print(f"[{name}] captions with non-ASCII chars={len(nonascii)}")
    print(f"[{name}] 20 rare words (possible typos): {rare[:20]}")

    plt.figure(figsize=(8, 4))
    plt.hist(lengths, bins=40, edgecolor="black")
    plt.title(f"Caption length in words ({name})")
    plt.xlabel("words")
    plt.ylabel("captions")
    plt.tight_layout()
    plt.savefig(OUT / f"caption_len_{name}.png", dpi=150)
    plt.close()

    top = vocab.most_common(25)
    plt.figure(figsize=(10, 4))
    plt.bar([w for w, _ in top], [n for _, n in top])
    plt.xticks(rotation=70)
    plt.title(f"Top 25 words ({name})")
    plt.tight_layout()
    plt.savefig(OUT / f"top_words_{name}.png", dpi=150)
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
        caption_eda(ds, name)


if __name__ == "__main__":
    main()