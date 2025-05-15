import os.path

from PIL import Image
from torch import Tensor
from torchvision import transforms
from torch.utils.data import Dataset
import torch
import torchio
import pandas as pd
import numpy as np
import h5py as h5


XRAY_TARGET_SIZE = (512, 512)


class XRayDataset(Dataset):
    def __init__(self,
                 reports_csv_path: str,
                 projections_csv_path: str,
                 xray_dir: str):
        self.xray_dir = xray_dir
        self.reports = pd.read_csv(reports_csv_path)
        self.projections = pd.read_csv(projections_csv_path)
        self.preprocess = transforms.Compose([
                transforms.CenterCrop(2048),
                transforms.Resize(XRAY_TARGET_SIZE),
                transforms.Grayscale(),
                transforms.ToTensor(),
            ])

    def __len__(self) -> int:
        return len(self.reports)

    def __getitem__(self, ix: int) -> (dict[str, Tensor], Tensor):
        report = self.reports.iloc[ix]

        imgs = []
        for proj in ['Frontal', 'Lateral']:
            projection = self.projections[(self.projections['uid'] == report['uid']) &
                                   (self.projections['projection'] == proj)].iloc[0]
            imgs.append(Image.open(os.path.join(self.xray_dir, projection['filename'])))

        xrays = {
            'frontal': self.preprocess(imgs[0]),
            'lateral': self.preprocess(imgs[1])
        }

        return xrays, torch.tensor(report['disease'], dtype=torch.long)


class XRayCTDataset(XRayDataset):
    def __init__(self,
                 reports_csv_path: str,
                 projections_csv_path: str,
                 xray_dir: str,
                 ct_dir: str):
        super().__init__(reports_csv_path, projections_csv_path, xray_dir)
        self.ct_dir = ct_dir
        self.ct_transform = torchio.Compose([
            torchio.RescaleIntensity(out_min_max=(0, 1), percentiles=(0.5, 99.5)),
            torchio.ZNormalization(),
            torchio.Resample((1.0, 1.0, 1.0)),
            torchio.CropOrPad((64, 128, 128)),
        ])


    def __getitem__(self, ix: int) -> (dict[str, Tensor], Tensor):
        data, label = super().__getitem__(ix)

        report = self.reports.iloc[ix]
        projection = self.projections[(self.projections['uid'] == report['uid']) &
                                   (self.projections['projection'] == 'Volume')].iloc[0]

        with h5.File(os.path.join(self.ct_dir, projection['filename']), 'r') as volume:
            volume = torch.tensor(np.array(volume['ct']))
            volume = self.ct_transform(volume)
            data['ct'] = volume.unsqueeze(0)

        return data, label