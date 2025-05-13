from huggingface_hub import hf_hub_download
import torch
import torch.nn as nn
from torch.nn import Linear

from lib.resnet import resnet18


class Med3DBackbone(nn.Module):
    """
    Med3D Feature extractor based on resnet 18
    """
    def __init__(self):
        super(Med3DBackbone, self).__init__()
        model = resnet18(sample_input_D=64, shortcut_type='A')
        state = torch.load(_load_med3d_checkpoint())
        self.model.load_state_dict(state)

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
        super(X2CTMed3D, self).__init__()

        self.backbone = Med3DBackbone()
        self.classifier = Linear(512 * 4 * 4 * 8, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features_3d = self.backbone(x)
        x = self.classifier(features_3d)

        return x


def _load_med3d_checkpoint() -> str:
    return hf_hub_download(
        repo_id='TencentMedicalNet/MedicalNet-Resnet18',
        filename='resnet_18_23dataset.pth',
        cache_dir='models/checkpoints'
    )