import torch
import torch.nn as nn
import torchvision.models as models
from tensorflow import Tensor


class CheXNetBackbone(nn.Module):
    """
    CheXNet model adapted for dual-view X-ray images (PA + LAT).
    """

    def __init__(self, weights_path: str | None = None):
        super().__init__()
        self.backbone = _build_chexnet(1, weights_path)

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)

def _build_chexnet(input_channels: int, weights_path: str | None):
    """
    Loads DenseNet-121 and adapts it for binary classification with dual-view X-rays.

    Args:
        input_channels (int): Number of input channels (2 for PA + LAT views)
        weights_path (str): Path to the CheXNet .pth.tar checkpoint

    Returns:
        torch.nn.Module: Adapted DenseNet-121 model
    """

    assert 1 <= input_channels <= 3, f'invalid number of input channels ({input_channels})'

    # 1. Load base model without ImageNet weights
    model = models.densenet121(weights=None)

    # 2. Modify first conv layer for 2-channel input
    original_conv = model.features.conv0
    model.features.conv0 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

    if input_channels > 1:
        with torch.no_grad():
            for i in range(input_channels):
                model.features.conv0.weight[:, i:i+1] = original_conv.weight[:, 0:1]

    # Remove classifier
    model.classifier = nn.Identity()

    if weights_path is not None:
        #  Load pretrained weights
        checkpoint = torch.load(weights_path, weights_only=False)
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(clean_state_dict, strict=False)

    return model
