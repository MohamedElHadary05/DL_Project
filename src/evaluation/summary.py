# one table for the report: config + val (MLflow) + test (evaluate) per final model
import json
from pathlib import Path

import mlflow
import pandas as pd

EVAL_DIR = Path("outputs/eval")


def model_comparison(experiment="classification", out_path=EVAL_DIR / "model_comparison.csv"):
    """Join every outputs/eval/<run>_test_metrics.json with that run's MLflow params."""
    runs = mlflow.search_runs(experiment_names=[experiment])
    rows = []
    for path in sorted(EVAL_DIR.glob("*_test_metrics.json")):
        name = path.name.removesuffix("_test_metrics.json")
        if name.startswith("smoke"):
            continue
        match = runs[runs["tags.mlflow.runName"] == name]
        run = match.iloc[0] if len(match) else pd.Series(dtype=object)

        def param(key, default=None):
            value = run.get(f"params.{key}")
            return default if value is None or pd.isna(value) else value

        test = json.loads(path.read_text())
        rows.append({
            "run": name,
            "model": param("model"),
            "pretrained": param("pretrained", "False"),
            "frozen_encoder": param("freeze_encoder", "False"),
            "image_size": param("image_size"),
            "lr": param("lr"),
            "encoder_lr": param("encoder_lr", param("lr")),
            "weight_decay": param("weight_decay"),
            "epochs": param("epochs"),
            "trainable_params": param("num_trainable_params"),
            "best_val_accuracy": run.get("metrics.best_val_accuracy"),
            "test_accuracy": test["test_accuracy"],
            "test_precision_macro": test["test_precision_macro"],
            "test_recall_macro": test["test_recall_macro"],
            "test_f1_macro": test["test_f1_macro"],
            "test_top5_accuracy": test["test_top5_accuracy"],
        })
    df = pd.DataFrame(rows).sort_values("test_accuracy")
    df.to_csv(out_path, index=False)
    return df, out_path
