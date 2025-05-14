from huggingface_hub import hf_hub_download
import torch
import torch.nn as nn
from torch.nn import Linear

from x2ct_mme3d.lib.resnet import resnet18, ResNet


class Med3DBackbone(nn.Module):
    """
    Med3D Feature extractor based on resnet 18
    """
    def __init__(self):
        super().__init__()
        model = _load_med3d()

        # only include backbone layers
        self.backbone = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = x.flatten(start_dim=1)

        return x


class X2CTMed3D(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = Med3DBackbone()
        self.classifier = Linear(1048576, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features_3d = self.backbone(x)
        x = self.classifier(features_3d)

        return x


def _load_med3d() -> ResNet:
    model = resnet18(
        sample_input_D=64,
        sample_input_H=128,
        sample_input_W=128,
        num_seg_classes=1,
        shortcut_type='A')

    ckpt_path = hf_hub_download(
        repo_id='TencentMedicalNet/MedicalNet-Resnet18',
        filename='resnet_18_23dataset.pth',
        cache_dir='models/checkpoints'
    )

    state = torch.load(ckpt_path)
    # keys are prefixed with `.module`
    state_dict = {k.replace('module.', ''): v for k, v in state['state_dict'].items()}
    # checkpoint does not contain decoder (segmentation) weights,
    # but we are only interested in backbone, so `strict=False`
    # to suppress errors about missing segmentation weights
    model.load_state_dict(state_dict, strict=False)

    return model