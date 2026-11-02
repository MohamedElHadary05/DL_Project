# Grad-CAM heatmaps: correct vs wrong predictions, and model-vs-model comparison
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.dataset import CubClassificationDataset, get_class_names, load_cub
from src.data.preprocessing import clean_test_indices, stratified_split
from src.data.transforms import get_eval_transform
from src.training.trainer import load_checkpoint
from src.visualization.aug_plots import unnormalize

OUT = Path("outputs/gradcam")


class GradCAM:
    """Grad-CAM (Selvaraju et al., 2017) on the model's `gradcam_layer()`.

    Each channel of the layer's feature map is weighted by the mean gradient
    of the class score w.r.t. that channel; the ReLU of the weighted sum shows
    which regions pushed the prediction towards that class.
    """

    def __init__(self, model, layer=None):
        self.model = model
        layer = layer or model.gradcam_layer()
        self.handle = layer.register_forward_hook(self._save)

    def _save(self, module, inputs, output):
        self.acts = output
        output.register_hook(lambda grad: setattr(self, "grads", grad))

    def __call__(self, x, class_idx=None):
        """x: normalized batch. Returns (cams in [0, 1] at input size, logits)."""
        self.model.eval()
        # Input needs grad so the graph reaches the layer even if the encoder is frozen.
        x = x.clone().requires_grad_(True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(1)
        self.model.zero_grad(set_to_none=True)
        logits.gather(1, class_idx.view(-1, 1)).sum().backward()

        weights = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        cam = cam - cam.amin(dim=(1, 2), keepdim=True)
        cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-8)
        return cam.detach().cpu().numpy(), logits.detach()

    def remove(self):
        self.handle.remove()


def load_split(split):
    """(hf dataset, indices) for 'val' (stratified slice of train) or 'test'."""
    train, test = load_cub()
    if split == "val":
        return train, stratified_split(train["label"])[1]
    return test, clean_test_indices(len(test))


@torch.no_grad()
def predict(model, ds, indices, image_size, batch_size=64):
    """Softmax probabilities and labels for every image of the split."""
    device = next(model.parameters()).device
    data = CubClassificationDataset(ds, indices, get_class_names(ds),
                                    get_eval_transform(image_size))
    probs, labels = [], []
    for x, y in DataLoader(data, batch_size=batch_size):
        probs.append(model(x.to(device)).softmax(1).cpu())
        labels.append(y)
    return torch.cat(probs), torch.cat(labels)


def explain_one(cam, ds, hf_idx, image_size):
    """(image as HxWx3 array, heatmap, predicted class, confidence) for one image."""
    device = next(cam.model.parameters()).device
    x = get_eval_transform(image_size)(ds[hf_idx]["image"].convert("RGB"))
    heat, logits = cam(x.unsqueeze(0).to(device))
    p = logits.softmax(1)[0]
    return unnormalize(x), heat[0], int(p.argmax()), float(p.max())


def _short(name, width=18):
    return name if len(name) <= width else name[:width - 1] + "."


def plot_correct_wrong(ckpt_path, split="val", n=6, seed=0):
    """Two rows of correct and two of wrong predictions: image, then heatmap overlay."""
    model, cfg, class_names = load_checkpoint(ckpt_path)
    ds, indices = load_split(split)
    probs, labels = predict(model, ds, indices, cfg["image_size"])
    preds = probs.argmax(1)
    rng = np.random.default_rng(seed)
    groups = {
        "correct": rng.permutation(np.where(preds == labels)[0])[:n],
        "wrong": rng.permutation(np.where(preds != labels)[0])[:n],
    }

    cam = GradCAM(model)
    fig, axes = plt.subplots(4, n, figsize=(2.3 * n, 10.5), squeeze=False)
    for g, (kind, positions) in enumerate(groups.items()):
        for j in range(n):
            top, bottom = axes[2 * g, j], axes[2 * g + 1, j]
            top.axis("off")
            bottom.axis("off")
            if j >= len(positions):
                continue
            pos = int(positions[j])
            img, heat, pred, conf = explain_one(cam, ds, indices[pos], cfg["image_size"])
            top.imshow(img)
            top.set_title(f"true: {_short(class_names[int(labels[pos])])}", fontsize=7)
            bottom.imshow(img)
            bottom.imshow(heat, cmap="jet", alpha=0.45)
            bottom.set_title(f"pred: {_short(class_names[pred])} ({conf:.0%})", fontsize=7,
                             color="green" if kind == "correct" else "red")
        axes[2 * g, 0].text(-0.15, 0, kind.upper(), transform=axes[2 * g, 0].transAxes,
                            rotation=90, va="top", fontsize=11, weight="bold")
    cam.remove()

    name = Path(ckpt_path).stem
    acc = float((preds == labels).float().mean())
    fig.suptitle(f"Grad-CAM: {name} ({split} accuracy {acc:.1%})", fontsize=12)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}_{split}_correct_wrong.png"
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_comparison(ckpt_paths, split="val", n=6, seed=0):
    """Same images for every model: rows are images, columns are models."""
    ds, indices = load_split(split)
    class_names = get_class_names(ds)
    picks = np.random.default_rng(seed).choice(len(indices), n, replace=False)

    fig, axes = plt.subplots(n, len(ckpt_paths) + 1,
                             figsize=(2.6 * (len(ckpt_paths) + 1), 2.6 * n), squeeze=False)
    for c, ckpt in enumerate(ckpt_paths, start=1):
        model, cfg, _ = load_checkpoint(ckpt)
        cam = GradCAM(model)
        axes[0, c].set_title(Path(ckpt).stem, fontsize=10, weight="bold", pad=18)
        for r, pos in enumerate(picks):
            img, heat, pred, conf = explain_one(cam, ds, indices[int(pos)], cfg["image_size"])
            label = ds[indices[int(pos)]]["label"]
            ax = axes[r, c]
            ax.imshow(img)
            ax.imshow(heat, cmap="jet", alpha=0.45)
            ax.text(0.5, -0.06, f"{_short(class_names[pred])} ({conf:.0%})",
                    transform=ax.transAxes, ha="center", va="top", fontsize=7,
                    color="green" if pred == label else "red")
            ax.axis("off")
            if c == 1:
                axes[r, 0].imshow(img)
                axes[r, 0].set_title(f"true: {_short(class_names[label], 24)}", fontsize=8)
                axes[r, 0].axis("off")
        cam.remove()
    axes[0, 0].set_title(axes[0, 0].get_title(), fontsize=8, pad=18)

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"comparison_{split}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path
