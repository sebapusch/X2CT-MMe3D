from x2ct_mme3d.models.chexnet_model import CheXNetBackbone
from x2ct_mme3d.models.med3d import Med3DBackbone

from torch import nn, Tensor


class X2CTMMe3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.xray_backbone = CheXNetBackbone()
        self.ct_backbone = Med3DBackbone()

    def forward(self, x: dict) -> Tensor:
        xray_features = self.xray_backbone(x['xrays'])
        ct_features = self.ct_backbone(x['ct'])

        print(xray_features.shape)
        print(ct_features.shape)

        return Tensor([])

