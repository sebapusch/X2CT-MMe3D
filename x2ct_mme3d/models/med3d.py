from huggingface_hub import hf_hub_download
import torch
import torch.nn as nn
from torch.nn import Linear

from x2ct_mme3d.lib.resnet import resnet18, ResNet, resnet34


class Med3DBackbone(nn.Module):
    """
    Med3D Feature extractor based on resnet 18
    """
    def __init__(self, arch: str = 'resnet18', pretrained: bool = False):
        super().__init__()
        model = _load_med3d(arch, pretrained)

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

        self.flatten = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.flatten(x)

        return x


class X2CTMed3D(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = Med3DBackbone()
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.classifier(x)

        return x


def _load_med3d(arch: str, pretrained: bool) -> ResNet:
    ckpt_path = ''
    if arch == 'resnet18':

        model = resnet18(
            sample_input_D=64,
            sample_input_H=128,
            sample_input_W=128,
            num_seg_classes=1,
            shortcut_type='A')

        if pretrained:
            ckpt_path = hf_hub_download(
                repo_id='TencentMedicalNet/MedicalNet-Resnet18',
                filename='resnet_18_23dataset.pth',
                cache_dir='models/checkpoints'
            )
    elif arch == 'resnet34':
        model = resnet34(
            sample_input_D=64,
            sample_input_H=128,
            sample_input_W=128,
            num_seg_classes=1,
            shortcut_type='A')

        if pretrained:
            ckpt_path = hf_hub_download(
                repo_id='TencentMedicalNet/MedicalNet-Resnet34',
                filename='resnet_34_23dataset.pth',
                cache_dir='models/checkpoints'
            )
    else: raise ValueError(f'Unknown architecture {arch}')

    if pretrained:
        state = torch.load(ckpt_path)
        # keys are prefixed with `.module`
        state_dict = {k.replace('module.', ''): v for k, v in state['state_dict'].items()}
        # checkpoint does not contain decoder (segmentation) weights,
        # but we are only interested in backbone, so `strict=False`
        # to suppress errors about missing segmentation weights
        model.load_state_dict(state_dict, strict=False)

    return model