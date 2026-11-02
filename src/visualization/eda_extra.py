import difflib
import hashlib
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.dataset import load_cub, split_captions

OUT = Path("outputs/eda")


def dhash(img, size=8):
    """64-bit perceptual hash: survives resizing and re-compression."""
    g = img.convert("L").resize((size + 1, size))
    a = np.asarray(g, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def scan_images(ds, name):
    rows, s, s2, npix = [], np.zeros(3), np.zeros(3), 0
    for i in tqdm(range(len(ds)), desc=f"quality {name}"):
        img = ds[i]["image"].convert("RGB")
        small = img.copy()
        small.thumbnail((256, 256))
        a = np.asarray(small, dtype=np.float32) / 255.0
        g = a.mean(axis=2) * 255
        lap = (-4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1]
               + g[1:-1, :-2] + g[1:-1, 2:])
        s += a.sum(axis=(0, 1))
        s2 += (a ** 2).sum(axis=(0, 1))
        npix += a.shape[0] * a.shape[1]
        rows.append({
            "idx": i, "brightness": g.mean(), "contrast": g.std(),
            "blur": lap.var(), "dhash": dhash(img),
            "md5": hashlib.md5(img.tobytes()).hexdigest(),
        })
    mean = s / npix
    std = np.sqrt(s2 / npix - mean ** 2)
    print(f"[{name}] RGB mean={mean.round(3)} std={std.round(3)} "
          f"(ImageNet: [0.485 0.456 0.406] / [0.229 0.224 0.225])")
    return pd.DataFrame(rows)


def report_quality(df, ds, name):
    print(f"[{name}] brightness mean={df.brightness.mean():.1f} "
          f"contrast mean={df.contrast.mean():.1f}")
    thr = df.blur.quantile(0.02)
    print(f"[{name}] blur (Laplacian var): median={df.blur.median():.0f} "
          f"2nd percentile={thr:.0f}")
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.5))
    for a, col in zip(ax, ["brightness", "contrast", "blur"]):
        a.hist(df[col].clip(upper=df[col].quantile(0.99)), bins=40, edgecolor="black")
        a.set_title(f"{col} ({name})")
    plt.tight_layout()
    plt.savefig(OUT / f"quality_{name}.png", dpi=150)
    plt.close()

    worst = df.nsmallest(8, "blur").idx.tolist()
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for a, i in zip(axes.ravel(), worst):
        a.imshow(ds[i]["image"])
        a.set_title(f"idx {i}", fontsize=8)
        a.axis("off")
    plt.tight_layout()
    plt.savefig(OUT / f"blurriest_{name}.png", dpi=150)
    plt.close()


def report_duplicates(dfs):
    for name, df in dfs.items():
        for col in ["md5", "dhash"]:
            groups = defaultdict(list)
            for i, h in zip(df.idx, df[col]):
                groups[h].append(i)
            dup = [g for g in groups.values() if len(g) > 1]
            print(f"[{name}] duplicate groups by {col}: {len(dup)} "
                  f"(images involved: {sum(len(g) for g in dup)})")
    for col in ["md5", "dhash"]:
        leak = set(dfs["train"][col]) & set(dfs["test"][col])
        print(f"[train/test leakage] shared {col}: {len(leak)}")


def typo_candidates(ds, name, rare_max=2, common_min=20):
    words = Counter(w for d in ds["description"] for c in split_captions(d)
                    for w in re.findall(r"[a-z']+", c.lower()))
    common = [w for w, n in words.items() if n >= common_min]
    rows = []
    for w, n in words.items():
        if n <= rare_max:
            m = difflib.get_close_matches(w, common, n=1, cutoff=0.8)
            if m:
                rows.append({"word": w, "count": n, "closest_common_word": m[0]})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"typo_candidates_{name}.csv", index=False)
    rare_total = sum(1 for n in words.values() if n <= rare_max)
    print(f"[{name}] rare words={rare_total}, with a close common word "
          f"(likely typos)={len(df)}")
    print(df.head(25).to_string(index=False))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train, test = load_cub()
    dfs = {}
    for name, ds in [("train", train), ("test", test)]:
        dfs[name] = scan_images(ds, name)
        report_quality(dfs[name], ds, name)
        typo_candidates(ds, name)
    report_duplicates(dfs)


if __name__ == "__main__":
    main()