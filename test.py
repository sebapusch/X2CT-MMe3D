from x2ct_mme3d.data.dataset import XRayCTDataset
from x2ct_mme3d.models.classifier import X2CTMMe3D

model = X2CTMMe3D()

dataset = XRayCTDataset(
    './data/test/indiana_reports.csv',
    './data/test/indiana_projections.csv',
    './data/processed/xrays',
    './data/processed/volumes',
)

model(dataset[0][0])