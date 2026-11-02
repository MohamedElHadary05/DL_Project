# CNN encoder trained from scratch (baseline)
import torch.nn as nn


def conv_block(c_in, c_out):
    """Two 3x3 conv-BN-ReLU layers, then halve the resolution."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class ScratchEncoder(nn.Module):
    """Image -> spatial feature map. Reused later by colorization and captioning.

    128x128 input -> 5 blocks -> 512 x 4 x 4 feature map.
    """

    def __init__(self, widths=(32, 64, 128, 256, 512)):
        super().__init__()
        blocks, c_in = [], 3
        for c_out in widths:
            blocks.append(conv_block(c_in, c_out))
            c_in = c_out
        self.blocks = nn.Sequential(*blocks)
        self.out_channels = c_in

    def forward(self, x):
        return self.blocks(x)


class Classifier(nn.Module):
    """Any encoder exposing `out_channels` + global average pooling + linear head."""

    def __init__(self, encoder, num_classes, dropout=0.3):
        super().__init__()
        self.encoder = encoder
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout),
                                  nn.Linear(encoder.out_channels, num_classes))

    def forward(self, x):
        return self.head(self.pool(self.encoder(x)))

    def gradcam_layer(self):
        """Last conv stage: highest-level features that still have spatial layout."""
        return self.encoder.blocks[-1]
