import torch

from x2ct_mme3d.models.chexnet import CheXNetBackbone
from x2ct_mme3d.models.med3d import Med3DBackbone

from torch import nn, Tensor


class X2CTMMe3D(nn.Module):
    def __init__(self, med3d: bool = False, chestx_path: str | None = None):
        super().__init__()
        self.frontal_backbone = CheXNetBackbone(chestx_path)   # (Bx1x512x512  -> 1024)
        self.lateral_backbone = CheXNetBackbone(chestx_path)   # (Bx1x512x512  -> 1024)
        self.ct_backbone = Med3DBackbone('resnet18', med3d)          # (Bx64x128x128 -> 512)
        self.classifier = nn.Sequential(
            nn.Linear(512 + 1024 * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )

    def forward(self, x: dict[str, Tensor]) -> Tensor:
        front = self.frontal_backbone(x['frontal'])
        back = self.lateral_backbone(x['lateral'])
        ct = self.ct_backbone(x['ct'])

        x = torch.cat((front, back, ct), dim=1)
        y = self.classifier(x)

        return y
