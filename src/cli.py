import argparse
import json
from pathlib import Path

import yaml


def load_config(path, overrides):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


def cmd_train(args):
    from src.training.trainer import train_classifier

    cfg = load_config(args.config, {"epochs": args.epochs, "lr": args.lr,
                                    "batch_size": args.batch_size,
                                    "num_workers": args.num_workers,
                                    "limit_batches": args.limit_batches})
    train_classifier(cfg, run_name=args.run_name)


def cmd_tune(args):
    from src.training.tuning import compare_searches, run_search

    with open(args.config) as f:
        tune_cfg = yaml.safe_load(f)
    overrides = {k: v for k, v in {"epochs": args.epochs, "num_workers": args.num_workers,
                                   "limit_batches": args.limit_batches}.items()
                 if v is not None}
    methods = ["grid", "random"] if args.method == "both" else [args.method]
    for method in methods:
        run_search(method, tune_cfg, overrides)
    compare_searches(tune_cfg)


def cmd_evaluate(args):
    from src.evaluation.metrics import confusion
    from src.training.trainer import evaluate_classifier
    from src.visualization.training_plots import plot_confusion

    metrics, y_pred, y_true, class_names = evaluate_classifier(
        args.checkpoint, limit_batches=args.limit_batches)
    for k, v in metrics.items():
        print(f"{k:22s} {v:.4f}")
    stem = Path(args.checkpoint).stem
    out = Path("outputs/eval")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{stem}_test_metrics.json").write_text(json.dumps(metrics, indent=2))
    cm = confusion(y_true, y_pred, len(class_names))
    print(f"saved {plot_confusion(cm, out / f'{stem}_confusion.png', f'Test confusion: {stem}')}")


def cmd_explain(args):
    from src.visualization.gradcam import plot_comparison, plot_correct_wrong

    for ckpt in args.checkpoint:
        print(f"saved {plot_correct_wrong(ckpt, args.split, args.n, args.seed)}")
    if len(args.checkpoint) > 1:
        print(f"saved {plot_comparison(args.checkpoint, args.split, args.n, args.seed)}")


def cmd_predict(args):
    import torch
    from PIL import Image

    from src.data.transforms import get_eval_transform
    from src.training.trainer import load_checkpoint

    model, cfg, class_names = load_checkpoint(args.checkpoint)
    device = next(model.parameters()).device
    img = Image.open(args.image).convert("RGB")
    x = get_eval_transform(cfg["image_size"])(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = model(x).softmax(1)[0]
    top = probs.topk(args.top_k)
    for p, i in zip(top.values.tolist(), top.indices.tolist()):
        print(f"{p:6.1%}  {class_names[i]}")


def main():
    parser = argparse.ArgumentParser(description='From Pixels to Language')
    sub = parser.add_subparsers(dest='command', required=True)

    train = sub.add_parser('train', help='train a model from a YAML config')
    train.add_argument('--task', required=True, choices=['classification'])
    train.add_argument('--config', required=True)
    train.add_argument('--run-name', help='MLflow run / checkpoint name')
    train.add_argument('--epochs', type=int, help='override config epochs')
    train.add_argument('--lr', type=float, help='override config lr')
    train.add_argument('--batch-size', type=int, help='override config batch_size')
    train.add_argument('--num-workers', type=int, help='override config num_workers')
    train.add_argument('--limit-batches', type=int,
                       help='only run N batches per epoch (quick smoke test)')
    train.set_defaults(func=cmd_train)

    tune = sub.add_parser('tune', help='grid / random hyperparameter search')
    tune.add_argument('--task', required=True, choices=['classification'])
    tune.add_argument('--config', required=True, help='e.g. configs/tune_scratch.yaml')
    tune.add_argument('--method', default='both', choices=['grid', 'random', 'both'])
    tune.add_argument('--epochs', type=int, help='override epochs for every trial')
    tune.add_argument('--num-workers', type=int)
    tune.add_argument('--limit-batches', type=int,
                      help='only run N batches per epoch (quick smoke test)')
    tune.set_defaults(func=cmd_tune)

    evaluate = sub.add_parser('evaluate', help='final metrics on the test set')
    evaluate.add_argument('--task', required=True, choices=['classification'])
    evaluate.add_argument('--checkpoint', required=True)
    evaluate.add_argument('--limit-batches', type=int)
    evaluate.set_defaults(func=cmd_evaluate)

    explain = sub.add_parser('explain', help='Grad-CAM heatmaps (correct vs wrong, model comparison)')
    explain.add_argument('--task', required=True, choices=['classification'])
    explain.add_argument('--checkpoint', required=True, nargs='+',
                         help='one or more checkpoints; 2+ also makes a side-by-side comparison')
    explain.add_argument('--split', default='val', choices=['val', 'test'])
    explain.add_argument('--n', type=int, default=6, help='images per group')
    explain.add_argument('--seed', type=int, default=0, help='which images get picked')
    explain.set_defaults(func=cmd_explain)

    predict = sub.add_parser('predict', help='run a trained model on one image')
    predict.add_argument('--task', required=True, choices=['classification'])
    predict.add_argument('--image', required=True)
    predict.add_argument('--checkpoint', required=True)
    predict.add_argument('--top-k', type=int, default=5)
    predict.set_defaults(func=cmd_predict)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
