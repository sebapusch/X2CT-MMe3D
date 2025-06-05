import numpy as np
import torch
from PIL import Image
from torchcam.methods import GradCAM
import matplotlib as cm
from scipy.ndimage import zoom

from api.perx2ct_client import PerX2CTClient
from x2ct_mme3d.data.preprocess import xray_pipeline, ct_pipeline
from x2ct_mme3d.models.classifiers import X2CTMMed3D


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

    def _load_model(self, checkpoint_path: str) -> X2CTMMed3D:
        model = X2CTMMed3D()
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        # Select the CT backbone only
        self.ct_model = model.ct_backbone

        # Choose the last conv layer (layer4 is deepest in Med3DBackbone)
        target_layer = self.ct_model.backbone[-3][-1].conv2  # last conv in layer4

        # Hook GradCAM
        self.cam_extractor = GradCAM(model=self.ct_model, target_layer=target_layer)
        return model

    def __call__(self, frontal_img: Image.Image, lateral_img: Image.Image) -> dict:
        self.perx2ct.send(np.array(frontal_img))
        self.perx2ct.send(np.array(lateral_img))

        try:
            volume_np = self.perx2ct.receive()
            # np.save('./data/app/volume.npy', volume_np)
        except ValueError:
            raise RuntimeError('Unable to generate CT scan')

        volume = torch.from_numpy(volume_np).unsqueeze(0)
        volume_norm = (volume_np - volume_np.min()) / (volume_np.max() - volume_np.min())

        data = {
            'frontal': self.xray_pre_pipe(frontal_img).unsqueeze(0).to(self.device),
            'lateral': self.xray_pre_pipe(lateral_img).unsqueeze(0).to(self.device),
            'ct': self.ct_pre_pipe(volume).unsqueeze(0).to(self.device),
        }


        pred = self.model(data)
        _ = self.ct_model(data['ct'])

        # Get raw CAM before colormap: shape (D, H, W)
        raw_cam = self.cam_extractor(class_idx=0, scores=pred)[0].detach().cpu().numpy()

        # Resize CAM to match CT shape if needed
        if raw_cam.shape != volume.shape:
            from scipy.ndimage import zoom
            zoom_factors = np.array(volume.shape) / np.array(raw_cam.shape)
            raw_cam = zoom(raw_cam, zoom=zoom_factors, order=1)

        # Normalize CT volume
        volume_norm = (volume_np - volume_np.min()) / (volume_np.max() - volume_np.min())

        # Apply colormap AFTER resizing CAM: (D, H, W, 4) → keep only RGB
        from matplotlib import cm
        cmap = cm.get_cmap('jet')
        cam_colored = cmap(raw_cam)[..., :3]  # Shape: (D, H, W, 3)

        # Convert CT to RGB
        ct_rgb = np.repeat(volume_norm[..., np.newaxis], 3, axis=3)  # Shape: (D, H, W, 3)

        # Blend: 70% CT + 30% CAM
        overlay = (0.7 * ct_rgb + 0.3 * cam_colored).astype(np.float32)

        # Save ove

        np.save('./data/app/volume.npy', overlay)

        prob = float(torch.sigmoid(pred).flatten()[0])

        return {
            'probability': prob,
            'diagnosis':  'healthy' if prob < self.threshold else 'disease',
        }
