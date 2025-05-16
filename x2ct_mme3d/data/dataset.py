import os.path
from abc import ABC

from PIL import Image
from torch import Tensor
from torchvision import transforms
from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
import h5py as h5


XRAY_TARGET_SIZE = (512, 512)


class BaseDataset(Dataset, ABC):
    def __init__(self,
                 reports_csv_path: str,
                 projections_csv_path: str):
        self.reports = pd.read_csv(reports_csv_path)
        self.projections = pd.read_csv(projections_csv_path)

    def __len__(self) -> int:
        return len(self.reports)


class CtDataset(BaseDataset):
    def __init__(self, ct_dir: str, **kwargs):
        self.ct_dir = ct_dir
        super().__init__(**kwargs)

    def __getitem__(self, ix: int) -> (dict[str, Tensor], Tensor):
        report = self.reports.iloc[ix]
        projection = self.projections[(self.projections['uid'] == report['uid']) &
                                   (self.projections['projection'] == 'Volume')].iloc[0]

        data = {'ct': None}
        with h5.File(os.path.join(self.ct_dir, projection['filename']), 'r') as volume:
            data['ct'] = torch.tensor(np.array(volume['ct']))

        return data, torch.tensor(report['disease'], dtype=torch.long)



class XRayDataset(BaseDataset):
    def __init__(self, xray_dir: str, **kwargs):
        super().__init__(**kwargs)
        self.xray_dir = xray_dir
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
            projection = self.projections[
                (self.projections['uid'] == report['uid']) &
                (self.projections['projection'] == proj)
                ].iloc[0]

            img = Image.open(os.path.join(self.xray_dir, projection['filename']))
            imgs.append(self.preprocess(img))

        xrays = {
            'xrays': torch.stack(imgs, dim=0),
        }

        return xrays, torch.tensor(report['disease'], dtype=torch.long)


class X2CTDataset(XRayDataset, CtDataset):
    def __init__(self,
                 reports_csv_path: str,
                 projections_csv_path: str,
                 xray_dir: str,
                 ct_dir: str):
        super().__init__(
            reports_csv_path=reports_csv_path,
            projections_csv_path=projections_csv_path,
            xray_dir=xray_dir,
            ct_dir=ct_dir
        )

    def __getitem__(self, ix: int) -> (dict[str, Tensor], Tensor):
        xrays, _  = XRayDataset.__getitem__(self, ix)
        ct, label = CtDataset.__getitem__(self, ix)

        out = {
            'xrays': xrays,
            'ct': ct,
        }

        return out, label

