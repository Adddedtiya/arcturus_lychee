# Models. These are examples of project code.
# For a new project, add a model file in architecture/, or build blocks in block/.

# EfficientNetV2 with a new classifier head (pretrained on ImageNet)
from arcturus_lychee.models.architecture.efficient_net_v2 import (
    EfficientNetV2_Large,
    EfficientNetV2_Medium,
    EfficientNetV2_Small,
)

# MobileNet with a new classifier head (pretrained on ImageNet)
from arcturus_lychee.models.architecture.mobile_net import (
    BasicMobileNetV3,
    BasicModuleNetV2,
)
