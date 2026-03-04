import torch
import torch.nn as nn
import torchvision as tv

class BasicMobileNetV3(nn.Module):
    def __init__(self, output_classes : int) -> None:
        super().__init__()

        self.base = tv.models.mobilenet_v3_large(weights = tv.models.MobileNet_V3_Large_Weights.DEFAULT)
        self.base.classifier = nn.Sequential(
            nn.Linear(960, 1280),
            nn.Hardswish(),
            nn.Dropout(0.25),
            nn.Linear(1280, output_classes)
        )
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = self.base(x)
        return x

class BasicModuleNetV2(nn.Module):
    def __init__(self, output_classes : int) -> None:
        super().__init__()

        self.base = tv.models.mobilenet_v2(weights = tv.models.MobileNet_V2_Weights.IMAGENET1K_V2)
        self.base.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, output_classes)
        )
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = self.base(x)
        return x

if __name__ == "__main__":
    print("MobileNet Arcitectures")

    model = BasicModuleNetV2(100)
    x = torch.rand(1, 3, 224, 224)
    y = model(x)
    print(y.shape)