import torch
import torch.nn as nn
import torchvision as tv

class EfficientNetV2_Large(nn.Module):
    def __init__(self, output_classes : int) -> None:
        super().__init__()

        self.base = tv.models.efficientnet_v2_l(weights = tv.models.EfficientNet_V2_L_Weights.IMAGENET1K_V1)
        self.base.classifier = nn.Sequential(
            nn.Dropout(0.4, inplace = True),
            nn.Linear(1280, output_classes)
        )
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = self.base(x)
        return x

class EfficientNetV2_Medium(nn.Module):
    def __init__(self, output_classes : int) -> None:
        super().__init__()

        self.base = tv.models.efficientnet_v2_m(weights = tv.models.EfficientNet_V2_M_Weights.IMAGENET1K_V1)
        self.base.classifier = nn.Sequential(
            nn.Dropout(0.3, inplace = True),
            nn.Linear(1280, output_classes)
        )
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = self.base(x)
        return x

class EfficientNetV2_Small(nn.Module):
    def __init__(self, output_classes : int) -> None:
        super().__init__()

        self.base = tv.models.efficientnet_v2_s(weights = tv.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        self.base.classifier = nn.Sequential(
            nn.Dropout(0.2, inplace = True),
            nn.Linear(1280, output_classes)
        )
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = self.base(x)
        return x

if __name__ == "__main__":
    print("MobileNet Arcitectures")

    model = EfficientNetV2_Small(100)
    x = torch.rand(1, 3, 224, 224)
    y = model(x)
    print(y.shape)
    print(model)
    # tv.models.EfficientNet_V2_S_Weights