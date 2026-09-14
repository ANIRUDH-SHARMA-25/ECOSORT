"""Custom and transfer-learning classifiers."""
from __future__ import annotations

import torch.nn as nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class CustomCNN(nn.Module):
    """Compact CNN baseline designed for 224px RGB images."""
    def __init__(self, num_classes: int):
        super().__init__()
        def block(in_channels: int, out_channels: int) -> nn.Sequential:
            return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels),
                                 nn.ReLU(inplace=True), nn.MaxPool2d(2), nn.Dropout2d(0.1))
        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128), block(128, 256), nn.AdaptiveAvgPool2d(1))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.35), nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(0.25), nn.Linear(128, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Build the selected model. `custom_cnn` and `mobilenet_v3_small` are supported."""
    if name == "custom_cnn":
        return CustomCNN(num_classes)
    if name == "mobilenet_v3_small":
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = mobilenet_v3_small(weights=weights)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        return model
    raise ValueError(f"Unsupported model '{name}'. Choose custom_cnn or mobilenet_v3_small.")


def freeze_backbone(model: nn.Module, name: str, freeze: bool) -> None:
    """Freeze/unfreeze transfer backbone while preserving its classifier gradients."""
    if name != "mobilenet_v3_small":
        return
    for parameter in model.features.parameters():
        parameter.requires_grad = not freeze
