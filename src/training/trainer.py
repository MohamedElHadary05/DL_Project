# generic training / validation loop + MLflow logging
import random
import time
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.data.loaders import build_classification_loaders
from src.evaluation.metrics import classification_metrics, top_k_accuracy
from src.models.factory import build_model
from src.visualization.training_plots import plot_curves

CHECKPOINT_DIR = Path("outputs/checkpoints")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None,
              limit_batches=None):
    """One pass over `loader`. Trains if an optimizer is given, else evaluates.

    Returns (mean loss, metrics dict, all logits, all labels).
    """
    training = optimizer is not None
    model.train(training)
    total_loss, all_logits, all_labels = 0.0, [], []
    use_amp = device.type == "cuda"

    with torch.set_grad_enabled(training):
        for i, (x, y) in enumerate(tqdm(loader, leave=False,
                                        desc="train" if training else "eval")):
            if limit_batches and i >= limit_batches:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.item() * len(y)
            all_logits.append(logits.detach().float().cpu())
            all_labels.append(y.cpu())

    logits, labels = torch.cat(all_logits), torch.cat(all_labels)
    metrics = classification_metrics(labels.numpy(), logits.argmax(1).numpy())
    metrics["top5_accuracy"] = top_k_accuracy(logits, labels, k=5)
    return total_loss / len(labels), metrics, logits, labels


def train_classifier(cfg, run_name=None):
    """Train the model described by `cfg`; best val checkpoint goes to outputs/."""
    set_seed(cfg.get("seed", 42))
    device = get_device()
    train_loader, val_loader, _, class_names = build_classification_loaders(
        image_size=cfg["image_size"], batch_size=cfg["batch_size"],
        num_workers=cfg.get("num_workers", 2), seed=cfg.get("seed", 42),
    )
    model = build_model(cfg, len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.get("label_smoothing", 0.0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                  weight_decay=cfg.get("weight_decay", 0.0))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    limit = cfg.get("limit_batches")

    run_name = run_name or f"{cfg['model']}_{time.strftime('%Y%m%d_%H%M%S')}"
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / f"{run_name}.pt"
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_acc = -1.0

    mlflow.set_experiment(cfg.get("experiment", "classification"))
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({k: v for k, v in cfg.items() if k != "limit_batches"})
        mlflow.log_params({
            "device": device.type,
            "num_params": sum(p.numel() for p in model.parameters()),
        })

        for epoch in range(1, cfg["epochs"] + 1):
            start = time.time()
            train_loss, train_m, _, _ = run_epoch(model, train_loader, criterion, device,
                                                  optimizer, scaler, limit)
            val_loss, val_m, _, _ = run_epoch(model, val_loader, criterion, device,
                                              limit_batches=limit)
            lr = optimizer.param_groups[0]["lr"]
            scheduler.step()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_m["accuracy"])
            history["val_acc"].append(val_m["accuracy"])
            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss, "lr": lr,
                                **{f"train_{k}": v for k, v in train_m.items()},
                                **{f"val_{k}": v for k, v in val_m.items()}}, step=epoch)

            improved = val_m["accuracy"] > best_acc
            if improved:
                best_acc = val_m["accuracy"]
                torch.save({"cfg": cfg, "class_names": class_names, "epoch": epoch,
                            "val_accuracy": best_acc, "state_dict": model.state_dict()},
                           ckpt_path)
            print(f"epoch {epoch:3d}/{cfg['epochs']}  "
                  f"train loss {train_loss:.3f} acc {train_m['accuracy']:.3f}  |  "
                  f"val loss {val_loss:.3f} acc {val_m['accuracy']:.3f} "
                  f"f1 {val_m['f1_macro']:.3f}  ({time.time() - start:.0f}s)"
                  f"{'  *best' if improved else ''}")

        curves = plot_curves(history, Path("outputs/plots") / f"{run_name}_curves.png")
        mlflow.log_metric("best_val_accuracy", best_acc)
        mlflow.log_artifact(str(curves))
        mlflow.log_artifact(str(ckpt_path))

    print(f"best val accuracy {best_acc:.3f} -> {ckpt_path}")
    return ckpt_path, best_acc


def load_checkpoint(path, device=None):
    """Rebuild the model from a checkpoint. Returns (model, cfg, class_names)."""
    device = device or get_device()
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["cfg"], len(ckpt["class_names"])).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["cfg"], ckpt["class_names"]


def evaluate_classifier(ckpt_path, limit_batches=None):
    """Final test-set evaluation. Run this once per final model, not for tuning."""
    device = get_device()
    model, cfg, class_names = load_checkpoint(ckpt_path, device)
    _, _, test_loader, _ = build_classification_loaders(
        image_size=cfg["image_size"], batch_size=cfg["batch_size"],
        num_workers=cfg.get("num_workers", 2),
    )
    loss, metrics, logits, labels = run_epoch(model, test_loader, nn.CrossEntropyLoss(),
                                              device, limit_batches=limit_batches)
    return {"test_loss": loss, **{f"test_{k}": v for k, v in metrics.items()}}, \
        logits.argmax(1).numpy(), labels.numpy(), class_names
