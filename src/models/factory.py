# build_model(name, cfg) -> swap encoders without touching the trainer
from src.models.scratch_cnn import Classifier, ScratchEncoder


def _scratch_cnn(cfg, num_classes):
    encoder = ScratchEncoder(widths=tuple(cfg.get("widths", (32, 64, 128, 256, 512))))
    return Classifier(encoder, num_classes, dropout=cfg.get("dropout", 0.3))


# Add transfer-learning models here (e.g. "resnet18") once pretrained.py exists.
MODELS = {
    "scratch_cnn": _scratch_cnn,
}


def build_model(cfg, num_classes):
    name = cfg["model"]
    if name not in MODELS:
        raise ValueError(f"unknown model '{name}', choose from {sorted(MODELS)}")
    return MODELS[name](cfg, num_classes)
