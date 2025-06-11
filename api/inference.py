import logging
from os import path
import zlib
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import zoom
from torchcam.methods import GradCAM

from matplotlib import cm

from api.perx2ct_client import PerX2CTClient
from x2ct_mme3d.data.preprocess import xray_pipeline, ct_pipeline
from x2ct_mme3d.models.classifiers import X2CTMMed3D


class Inference:
    def __init__(self,
                 perx2ct: PerX2CTClient,
                 checkpoint_path: str,
                 device: torch.device,
                 volume_out_dir: str,
                 threshold: float = 0.5):
        self.volume_out_dir = volume_out_dir
        self.device = device
        self.perx2ct = perx2ct
        self.threshold = threshold
        self.model = self._load_model(checkpoint_path)
        self.grad_cam = self._load_grad_cam()

        self.xray_pre_pipe = xray_pipeline()
        self.ct_pre_pipe = ct_pipeline()

    def _load_model(self, checkpoint_path: str) -> X2CTMMed3D:
        model = X2CTMMed3D()
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    def _load_grad_cam(self):
        return GradCAM(model=self.model.ct_backbone,
                       target_layer=self.model.ct_backbone.backbone[-1][-1].conv2)

    def _ct_heat_map(self,
                     raw_volume: np.ndarray,
                     ct: torch.Tensor,
                     scores: torch.Tensor) -> np.ndarray:
        volume_norm = (raw_volume - raw_volume.min()) / (raw_volume.max() - raw_volume.min())
        self.model.ct_backbone(ct)

        raw_cam = self.grad_cam(class_idx=0, scores=scores)[0].squeeze().cpu().numpy()
        target_shape = volume_norm.shape

        zoom_factors = [t / c for t, c in zip(target_shape, raw_cam.shape)]

        upsampled_cam = zoom(raw_cam, zoom_factors, order=3, mode='reflect')
        upsampled_cam = (upsampled_cam - upsampled_cam.min()) / (
                            (upsampled_cam.max() - upsampled_cam.min()))

        ct_rgb = np.repeat(volume_norm[..., np.newaxis], 3, axis=3)

        cmap = cm.get_cmap('jet')
        cam_colored = cmap(upsampled_cam)[..., :3]

        overlay = (0.7 * ct_rgb + 0.3 * cam_colored).astype(np.float32)
        overlay = overlay.squeeze().squeeze()

        return overlay

    def _filename(self, frontal: np.ndarray, lateral: np.ndarray) -> str:
        """
        Create unique hash from frontal + lateral
        """
        hash_front = zlib.adler32(frontal.tobytes())
        hash_lat = zlib.adler32(lateral.tobytes())

        return hex(hash_front) + hex(hash_lat)

    def _store_volumes(self, filename: str, raw: np.ndarray, grad: np.ndarray) -> None:
        f_path_raw = path.join(self.volume_out_dir, f'{filename}.npy')
        f_path_grad = path.join(self.volume_out_dir, f'{filename}.grad.npy')

        np.save(f_path_raw, raw)
        np.save(f_path_grad, grad)

    def _load_cached_ct(self, filename: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        f_path_raw = path.join(self.volume_out_dir, f'{filename}.npy')
        f_path_grad = path.join(self.volume_out_dir, f'{filename}.grad.npy')

        if not path.exists(f_path_raw):
            return None

        logging.info('Loading CT scans from cache')

        raw: np.ndarray = np.load(f_path_raw)
        grad: np.ndarray = np.load(f_path_grad)

        return raw, grad

    def __call__(self, frontal_img: Image.Image, lateral_img: Image.Image) -> dict:
        np_frontal = np.array(frontal_img)
        np_lateral = np.array(lateral_img)

        filename = self._filename(np_frontal, np_lateral)

        cached = self._load_cached_ct(filename)
        is_cached = cached is not None

        if not is_cached:
            try:
                volume_np = self.perx2ct.generate(np.array(frontal_img),
                                                  np.array(lateral_img))
            except ValueError:
                raise RuntimeError('Unable to generate CT scan')
        else:
            volume_np, grad_cam = cached

        volume = torch.from_numpy(volume_np).unsqueeze(0)

        ct_preprocessed = self.ct_pre_pipe(volume).unsqueeze(0).to(self.device)

        data = {
            'frontal': self.xray_pre_pipe(frontal_img).unsqueeze(0).to(self.device),
            'lateral': self.xray_pre_pipe(lateral_img).unsqueeze(0).to(self.device),
            'ct': ct_preprocessed,
        }

        pred = self.model(data)
        prob = float(torch.sigmoid(pred).flatten()[0])

        if not is_cached:
            grad_cam = self._ct_heat_map(volume_np, ct_preprocessed, pred)
            self._store_volumes(filename, volume_np, grad_cam)

        return {
            'probability': prob,
            'diagnosis':  'healthy' if prob < self.threshold else 'disease',
            'path_raw_volume': path.join(self.volume_out_dir, f'{filename}.npy'),
            'path_grad_volume': path.join(self.volume_out_dir, f'{filename}.grad.npy'),
        }
