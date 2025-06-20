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
from x2ct_mme3d.data.preprocess import xray_pipeline


XRAY_TARGET_SIZE = (512, 512)


class BaseDataset(Dataset, ABC):
    def __init__(self,
                 reports_csv_path: str,
                 projections_csv_path: str,
                 include_uid: bool = False):
        self.reports = pd.read_csv(reports_csv_path)
        self.projections = pd.read_csv(projections_csv_path)
        self.include_uid = include_uid

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

        with h5.File(os.path.join(self.ct_dir, projection['filename']), 'r') as volume:
            data = torch.tensor(np.array(volume['ct']))

        return data, torch.tensor(report['disease'], dtype=torch.long)



class XRayDataset(BaseDataset):
    def __init__(self, xray_dir: str, **kwargs):
        super().__init__(**kwargs)
        self.xray_dir = xray_dir
        self.preprocess = xray_pipeline()
        

    def __len__(self) -> int:
        return len(self.reports)

    def __getitem__(self, ix: int) -> (dict[str, Tensor], Tensor):
        report = self.reports.iloc[ix]

        data = {}
        for proj in ['Frontal', 'Lateral']:
            projection = self.projections[
                (self.projections['uid'] == report['uid']) &
                (self.projections['projection'] == proj)
                ].iloc[0]

            img = Image.open(os.path.join(self.xray_dir, projection['filename']))
            data[proj.lower()] = self.preprocess(img)

        if self.include_uid:
            data['uid'] = report['uid']

        return data, torch.tensor(report['disease'], dtype=torch.long)


class X2CTDataset(XRayDataset, CtDataset):
    def __init__(self,
                 reports_csv_path: str,
                 projections_csv_path: str,
                 xray_dir: str,
                 ct_dir: str,
                 include_uid: bool = False):
        super().__init__(
            reports_csv_path=reports_csv_path,
            projections_csv_path=projections_csv_path,
            xray_dir=xray_dir,
            ct_dir=ct_dir,
            include_uid=include_uid
        )

    def __getitem__(self, ix: int) -> (dict[str, Tensor], Tensor):
        data, _  = XRayDataset.__getitem__(self, ix)
        ct, label = CtDataset.__getitem__(self, ix)

        data['ct'] = ct
        if self.include_uid:
            data['uid'] = label

        return data, label

