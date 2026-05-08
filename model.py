"""Single-task classifier with swappable pretrained backbone (via timm)."""
import timm
import torch.nn as nn


def build_model(num_classes: int, backbone: str = "efficientnet_b0", pretrained: bool = True,
                drop_rate: float = 0.2) -> nn.Module:
    """Backbone options ranked by speed / accuracy:
        - mobilenetv3_small_100  (fastest, ~2.5M params)
        - mobilenetv3_large_100  (fast, ~5.5M)
        - efficientnet_b0        (default, strong, ~5.3M)
        - efficientnetv2_rw_t    (stronger, ~13M)
        - convnext_tiny          (max accuracy, ~28M)
    """
    model = timm.create_model(
        backbone,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
    )
    return model
