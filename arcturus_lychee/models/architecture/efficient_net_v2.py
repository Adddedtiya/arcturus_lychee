"""EfficientNetV2 models from torchvision, with a new classifier head.

The backbones use the ImageNet weights of torchvision. The first use
downloads these weights. The head has dropout and one linear layer, with
output_classes outputs.
"""

import torch
import torch.nn as nn
import torchvision as tv


class EfficientNetV2_Large(nn.Module):
    """EfficientNetV2-L with a new head. The dropout before the last layer is 0.4."""

    def __init__(self, output_classes : int) -> None:
        super().__init__()

        self.base = tv.models.efficientnet_v2_l(weights = tv.models.EfficientNet_V2_L_Weights.IMAGENET1K_V1)
        self.base.classifier = nn.Sequential(
            nn.Dropout(0.4, inplace = True),
            nn.Linear(1280, output_classes),
        )

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.base(x)


class EfficientNetV2_Medium(nn.Module):
    """EfficientNetV2-M with a new head. The dropout before the last layer is 0.3."""

    def __init__(self, output_classes : int) -> None:
        super().__init__()

        self.base = tv.models.efficientnet_v2_m(weights = tv.models.EfficientNet_V2_M_Weights.IMAGENET1K_V1)
        self.base.classifier = nn.Sequential(
            nn.Dropout(0.3, inplace = True),
            nn.Linear(1280, output_classes),
        )

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.base(x)


class EfficientNetV2_Small(nn.Module):
    """EfficientNetV2-S with a new head. The dropout before the last layer is 0.2."""

    def __init__(self, output_classes : int) -> None:
        super().__init__()

        self.base = tv.models.efficientnet_v2_s(weights = tv.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        self.base.classifier = nn.Sequential(
            nn.Dropout(0.2, inplace = True),
            nn.Linear(1280, output_classes),
        )

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.base(x)


if __name__ == "__main__":
    # Demo: one forward pass. The first start downloads the ImageNet weights.
    model = EfficientNetV2_Small(100)
    x     = torch.rand(1, 3, 224, 224)
    y     = model(x)
    print(f"Output shape: {tuple(y.shape)}")
    print(model)
