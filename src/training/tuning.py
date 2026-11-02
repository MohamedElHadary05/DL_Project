# grid search + random search
import itertools
import math
import random
import time
from pathlib import Path

import pandas as pd
import yaml

from src.training.trainer import train_classifier

OUT = Path("outputs/tuning")


def grid_trials(space):
    """Every combination of the listed values, e.g. {lr: [a, b], wd: [c, d]} -> 4 trials."""
    keys = list(space)
    return [dict(zip(keys, values)) for values in itertools.product(*space.values())]


def sample(dist, rng):
    """A list means "pick one"; {low, high, log} means a continuous range."""
    if isinstance(dist, list):
        return rng.choice(dist)
    low, high = dist["low"], dist["high"]
    if dist.get("log"):
        return math.exp(rng.uniform(math.log(low), math.log(high)))
    return rng.uniform(low, high)


def random_trials(space, n_trials, seed=0):
    """Fixed seed -> the same trials every time, so an interrupted search can resume."""
    rng = random.Random(seed)
    return [{k: sample(d, rng) for k, d in space.items()} for _ in range(n_trials)]


def run_search(method, tune_cfg, overrides=None):
    """Train one model per trial; results go to outputs/tuning/<method>_results.csv.

    Trials already in the CSV are skipped, so a search killed mid-way
    (e.g. a Kaggle session timeout) continues where it stopped.
    """
    with open(tune_cfg["base"]) as f:
        base = yaml.safe_load(f)
    base.update(overrides or {})

    if method == "grid":
        trials = grid_trials(tune_cfg["grid"])
    else:
        spec = tune_cfg["random"]
        trials = random_trials(spec["space"], spec["n_trials"], spec.get("seed", 0))

    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / f"{method}_results.csv"
    rows = pd.read_csv(csv_path).to_dict("records") if csv_path.exists() else []
    done = {r["trial"] for r in rows}

    for i, params in enumerate(trials):
        if i in done:
            continue
        run_name = f"{method}_{i:02d}"
        print(f"\n=== {run_name} ({i + 1}/{len(trials)}): {params}")
        cfg = {**base, **params, "experiment": tune_cfg.get("experiment", "tuning"),
               "tuning_method": method, "log_checkpoint": False}
        start = time.time()
        _, best_acc = train_classifier(cfg, run_name=run_name)
        rows.append({"method": method, "trial": i, **params,
                     "best_val_accuracy": best_acc, "minutes": (time.time() - start) / 60})
        pd.DataFrame(rows).to_csv(csv_path, index=False)

    df = pd.DataFrame(rows)
    best = df.loc[df.best_val_accuracy.idxmax()]
    print(f"\n[{method}] best val accuracy {best.best_val_accuracy:.3f} with "
          f"{ {k: float(best[k]) for k in trials[0]} }")
    return df


def compare_searches():
    """Grid vs random: best score, trials, total time, and time to reach the best."""
    from src.visualization.tuning_plots import plot_best_so_far, plot_grid_heatmap

    results = {m: pd.read_csv(OUT / f"{m}_results.csv")
               for m in ("grid", "random") if (OUT / f"{m}_results.csv").exists()}
    rows = []
    for method, df in results.items():
        df = df.sort_values("trial")
        best_i = df.best_val_accuracy.idxmax()
        rows.append({
            "method": method,
            "trials": len(df),
            "best_val_accuracy": df.best_val_accuracy.max(),
            "best_trial": int(df.loc[best_i, "trial"]),
            "minutes_to_best": df.minutes.cumsum().loc[best_i],
            "total_minutes": df.minutes.sum(),
            "mean_val_accuracy": df.best_val_accuracy.mean(),
        })
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "comparison.csv", index=False)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    plot_best_so_far(results, OUT / "best_so_far.png")
    if "grid" in results:
        plot_grid_heatmap(results["grid"], OUT / "grid_heatmap.png")
    print(f"saved {OUT / 'comparison.csv'} and plots")
    return table
