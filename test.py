from torch.utils.data import DataLoader

from x2ct_mme3d.data.dataset import X2CTDataset
from x2ct_mme3d.models.x2ct_mme3d import X2CTMMe3D

model = X2CTMMe3D()

dataset = X2CTDataset(
    './data/test/indiana_reports.csv',
    './data/test/indiana_projections.csv',
    './data/processed/xrays',
    './data/processed/volumes',
)

data = DataLoader(dataset, batch_size=2)

batch, _ = next(iter(data))

print(model(batch))