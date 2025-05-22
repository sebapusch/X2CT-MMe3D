from os import path

import torch

from x2ct_mme3d.models.chexnet import CheXNetBackbone
from x2ct_mme3d.models.med3d import Med3DBackbone

from torch import nn, Tensor


CHEX_PATH = path.abspath(path.join('..', '..', 'models', 'checkpoints_', 'chexnet.pth.tar'))


class BiplanarCheXNet(nn.Module):
    """
    Classifier for biplanar x-rays based on CheXNet backbones
    """
    def __init__(self, pretrained: bool = False):
        super().__init__()
        self.frontal_backbone = CheXNetBackbone(CHEX_PATH if pretrained else None)
        self.lateral_backbone = CheXNetBackbone(CHEX_PATH if pretrained else None)
        self.classifier = _make_classifier(1024 * 2)

    def forward(self, x: dict[str, Tensor]) -> Tensor:
        front = self.frontal_backbone(x['frontal'])
        back = self.lateral_backbone(x['lateral'])

        x = torch.cat((front, back), dim=1)
        y = self.classifier(x)

        return y

class X2CTMMe3D(nn.Module):
    """
    Classifier for biplanar x-rays and CT scans based on
    CheXNet + Med3D backbones
    """
    def __init__(self, pretrained = False):
        super().__init__()
        self.frontal_backbone = CheXNetBackbone(CHEX_PATH if pretrained else None)
        self.lateral_backbone = CheXNetBackbone(CHEX_PATH if pretrained else None)
        self.ct_backbone = Med3DBackbone('resnet18', pretrained)
        self.classifier = _make_classifier(512 + 1024 * 2)

    def forward(self, x: dict[str, Tensor]) -> Tensor:
        front = self.frontal_backbone(x['frontal'])
        back = self.lateral_backbone(x['lateral'])
        ct = self.ct_backbone(x['ct'])

        x = torch.cat((front, back, ct), dim=1)
        y = self.classifier(x)

        return y

def _make_classifier(input_size: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_size, 128),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(128, 1)
    )