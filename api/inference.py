import uuid
from os import path

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

        print(raw_cam.shape)

        zoom_factors = [t / c for t, c in zip(target_shape, raw_cam.shape)]

        upsampled_cam = zoom(raw_cam, zoom_factors, order=3, mode='reflect')
        upsampled_cam = (upsampled_cam - upsampled_cam.min()) / (
                            (upsampled_cam.max() - upsampled_cam.min()))

#        raw_cam_up = F.interpolate(raw_cam, size=(64, 64, 64), mode='trilinear', align_corners=False)
#        raw_cam_up = F.interpolate(raw_cam_up, size=target_shape, mode='trilinear', align_corners=False)
#        raw_cam_up = raw_cam_up.cpu().detach().numpy()
#        raw_cam_up = gaussian_filter(raw_cam_up, sigma=2.5)

        ct_rgb = np.repeat(volume_norm[..., np.newaxis], 3, axis=3)

        cmap = cm.get_cmap('jet')
        cam_colored = cmap(upsampled_cam)[..., :3]

        overlay = (0.7 * ct_rgb + 0.3 * cam_colored).astype(np.float32)
        overlay = overlay.squeeze().squeeze()

        return overlay

    def _store_volumes(self, raw: np.ndarray, grad: np.ndarray) -> (str, str):
        uid = uuid.uuid1()
        f_path_raw = path.join(self.volume_out_dir, f'{uid}.npy')
        f_path_grad = path.join(self.volume_out_dir, f'{uid}.grad.npy')

        np.save(f_path_raw, raw)
        np.save(f_path_grad, grad)

        return f_path_raw, f_path_grad



    def __call__(self, frontal_img: Image.Image, lateral_img: Image.Image) -> dict:
        try:
            volume_np = self.perx2ct.generate(np.array(frontal_img),
                                              np.array(lateral_img))
        except ValueError:
            raise RuntimeError('Unable to generate CT scan')

        volume = torch.from_numpy(volume_np).unsqueeze(0)


        ct_preprocessed = self.ct_pre_pipe(volume).unsqueeze(0).to(self.device)

        data = {
            'frontal': self.xray_pre_pipe(frontal_img).unsqueeze(0).to(self.device),
            'lateral': self.xray_pre_pipe(lateral_img).unsqueeze(0).to(self.device),
            'ct': ct_preprocessed,
        }


        pred = self.model(data)
        prob = float(torch.sigmoid(pred).flatten()[0])

        grad_cam = self._ct_heat_map(volume_np, ct_preprocessed, pred)
        path_raw, path_grad = self._store_volumes(volume_np, grad_cam)


        return {
            'probability': prob,
            'diagnosis':  'healthy' if prob < self.threshold else 'disease',
            'path_raw_volume': path_raw,
            'path_grad_volume': path_grad,
        }
