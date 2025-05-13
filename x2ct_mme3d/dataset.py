import os.path

from PIL import Image
from torch import Tensor
from torchvision import transforms
from torch.utils.data import Dataset
import torch
import pandas as pd
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

    def __getitem__(self, ix: int) -> (dict[str, Tensor], bool):
        report = self.reports.iloc[ix]

        imgs = []
        for proj in ['Frontal', 'Lateral']:
            filename = self.projections[(self.projections['uid'] == report['uid']) &
                                   (self.projections['projection'] == proj)].iloc[0]
            imgs.append(Image.open(os.path.join(self.xray_dir, filename)))

        xrays = {
            'frontal': self.preprocess(imgs[0]),
            'lateral': self.preprocess(imgs[1])
        }

        return xrays, report['normal'].boolval()


class XRayCTDataset(XRayDataset):
    def __init__(self,
                 reports_csv_path: str,
                 projections_csv_path: str,
                 xray_dir: str,
                 ct_dir: str):
        super().__init__(reports_csv_path, projections_csv_path, xray_dir)
        self.ct_dir = ct_dir


    def __getitem__(self, ix: int) -> (dict[str, Tensor], bool):
        data, label = super().__getitem__(ix)

        report = self.reports.iloc[ix]
        filename = self.projections[(self.projections['uid'] == report['uid']) &
                                   (self.projections['projection'] == 'Volume')].iloc[0]

        with h5.File(os.path.join(self.ct_dir, filename), 'r') as volume:
            data['ct'] = torch.tensor(volume['ct'], dtype=torch.float32).unsqueeze(0)

        return data, label