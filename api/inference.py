import io
from multiprocessing.connection import Client
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from x2ct_mme3d.data.preprocess import xray_pipeline, ct_pipeline
from x2ct_mme3d.models.x2ct_mme3d import X2CTMMe3D


class Inference:
    def __init__(self, checkpoint_path: str, device: torch.device):
        self.device = device
        self.conn = None
        self.model = self._load_model(checkpoint_path)
        self.model.eval()

        # Pipelines
        self.xray_pre_pipe = xray_pipeline()
        self.ct_pre_pipe = ct_pipeline()

    def set_conn(self, conn: Client):
        self.conn = conn

    def _load_model(self, checkpoint_path: str) -> X2CTMMe3D:
        model = X2CTMMe3D()
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        return model

    def _send(self, data: np.ndarray):
        buf = io.BytesIO()
        np.save(buf, data, allow_pickle=False)
        self.conn.send(buf.getvalue())

    def _receive(self) -> np.ndarray:
        data = self.conn.recv()
        return np.load(io.BytesIO(data))

    def __call__(self, frontal_img: Image.Image, lateral_img: Image.Image) -> dict:
        # Send X-rays to PerX2CT process
        self._send(np.array(frontal_img))
        self._send(np.array(lateral_img))

        # Receive reconstructed CT volume
        volume_np = self._receive()
        volume = torch.from_numpy(volume_np).unsqueeze(0)

        # Preprocess all inputs
        data = {
            'frontal': self.xray_pre_pipe(frontal_img).unsqueeze(0).to(self.device),
            'lateral': self.xray_pre_pipe(lateral_img).unsqueeze(0).to(self.device),
            'ct': self.ct_pre_pipe(volume).unsqueeze(0).to(self.device),
        }

        with torch.no_grad():
            pred = self.model(data)

        probs = torch.sigmoid(pred)
        preds = (probs > 0.5).float()

        return {
            'probability': float(probs[0][0]),
            'prediction': float(preds[0][0]),
        }
