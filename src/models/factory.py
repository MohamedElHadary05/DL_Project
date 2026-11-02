# build_model(name, cfg) -> swap encoders without touching the trainer
from src.models.pretrained import ResNet18Encoder
from src.models.scratch_cnn import Classifier, ScratchEncoder


def _scratch_cnn(cfg, num_classes):
    encoder = ScratchEncoder(widths=tuple(cfg.get("widths", (32, 64, 128, 256, 512))))
    return Classifier(encoder, num_classes, dropout=cfg.get("dropout", 0.3))


def _resnet18(cfg, num_classes):
    encoder = ResNet18Encoder(pretrained=cfg.get("pretrained", True),
                              freeze=cfg.get("freeze_encoder", False))
    return Classifier(encoder, num_classes, dropout=cfg.get("dropout", 0.3))


MODELS = {
    "scratch_cnn": _scratch_cnn,
    "resnet18": _resnet18,
}


def build_model(cfg, num_classes):
    name = cfg["model"]
    if name not in MODELS:
        raise ValueError(f"unknown model '{name}', choose from {sorted(MODELS)}")
    return MODELS[name](cfg, num_classes)
