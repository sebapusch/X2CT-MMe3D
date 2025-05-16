import torch

from x2ct_mme3d.models.chexnet import CheXNetBackbone
from x2ct_mme3d.models.med3d import Med3DBackbone

from torch import nn, Tensor


class X2CTMMe3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.xray_backbone = CheXNetBackbone()
        self.ct_backbone = Med3DBackbone()
        self.classifier = nn.Sequential(
            nn.Linear(1536, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x: dict[str, Tensor]) -> Tensor:
        xray_features = self.xray_backbone(x['xrays'])
        ct_features = self.ct_backbone(x['ct'])

        x = torch.cat((xray_features, ct_features), dim=1)
        y = self.classifier(x)

        return y
