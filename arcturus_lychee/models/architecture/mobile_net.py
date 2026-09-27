"""MobileNet models from torchvision, with a new classifier head.

The backbones use the ImageNet weights of torchvision. The first use
downloads these weights.
"""

import torch
import torch.nn as nn
import torchvision as tv


class BasicMobileNetV3(nn.Module):
    """MobileNetV3-Large with a new head: 960 -> 1280 -> output_classes, with Hardswish and dropout 0.25."""

    def __init__(self, output_classes : int) -> None:
        super().__init__()

        self.base = tv.models.mobilenet_v3_large(weights = tv.models.MobileNet_V3_Large_Weights.DEFAULT)
        self.base.classifier = nn.Sequential(
            nn.Linear(960, 1280),
            nn.Hardswish(),
            nn.Dropout(0.25),
            nn.Linear(1280, output_classes),
        )

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.base(x)


class BasicMobileNetV2(nn.Module):
    """MobileNetV2 with a new head: dropout 0.2, then 1280 -> output_classes."""

    def __init__(self, output_classes : int) -> None:
        super().__init__()

        self.base = tv.models.mobilenet_v2(weights = tv.models.MobileNet_V2_Weights.IMAGENET1K_V2)
        self.base.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, output_classes),
        )

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.base(x)


if __name__ == "__main__":
    # Demo: one forward pass. The first start downloads the ImageNet weights.
    model = BasicMobileNetV2(100)
    x     = torch.rand(1, 3, 224, 224)
    y     = model(x)
    print(f"Output shape: {tuple(y.shape)}")
