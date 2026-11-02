# transfer learning encoders (ResNet / EfficientNet / ...)
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class ResNet18Encoder(nn.Module):
    """ResNet-18 without its avgpool / 1000-class fc: image -> 512 x H/32 x W/32.

    Same interface as ScratchEncoder (`blocks`, `out_channels`), so Classifier,
    the trainer and Grad-CAM work with either encoder unchanged.
    """

    def __init__(self, pretrained=True, freeze=False):
        super().__init__()
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.blocks = nn.Sequential(stem, net.layer1, net.layer2, net.layer3, net.layer4)
        self.out_channels = 512
        self.frozen = freeze
        if freeze:
            for p in self.parameters():
                p.requires_grad = False

    def train(self, mode=True):
        # A frozen encoder must also keep its BatchNorm running stats fixed.
        super().train(mode and not self.frozen)
        return self

    def forward(self, x):
        return self.blocks(x)
