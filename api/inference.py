import io
from multiprocessing.connection import Client
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from api.perx2ct_client import PerX2CTClient
from x2ct_mme3d.data.preprocess import xray_pipeline, ct_pipeline
from x2ct_mme3d.models.x2ct_mme3d import X2CTMMe3D

class Inference:
    def __init__(self,
                 perx2ct: PerX2CTClient,
                 checkpoint_path: str,
                 device: torch.device,
                 threshold: float = 0.5):
        self.device = device
        self.perx2ct = perx2ct
        self.threshold = threshold
        self.model = self._load_model(checkpoint_path)
        self.model.eval()

        # Pipelines
        self.xray_pre_pipe = xray_pipeline()
        self.ct_pre_pipe = ct_pipeline()

    def _load_model(self, checkpoint_path: str) -> X2CTMMe3D:
        model = X2CTMMe3D()
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        return model

    def __call__(self, frontal_img: Image.Image, lateral_img: Image.Image) -> dict:
        # Send X-rays to PerX2CT process
        self.perx2ct.send(np.array(frontal_img))
        self.perx2ct.send(np.array(lateral_img))

        try:
            volume_np = self.perx2ct.receive()
        except ValueError:
            raise RuntimeError('Unable to generate CT scan')

        volume = torch.from_numpy(volume_np).unsqueeze(0)

        data = {
            'frontal': self.xray_pre_pipe(frontal_img).unsqueeze(0).to(self.device),
            'lateral': self.xray_pre_pipe(lateral_img).unsqueeze(0).to(self.device),
            'ct': self.ct_pre_pipe(volume).unsqueeze(0).to(self.device),
        }

        with torch.no_grad():
            pred = self.model(data)

        prob = float(torch.sigmoid(pred).flatten()[0])

        return {
            'probability': prob,
            'diagnosis':  'healthy' if prob < self.threshold else 'disease',
        }
