import math
import os.path
from copy import deepcopy

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.cuda import device
from tqdm import tqdm

from main_test import load_config, change_config_file_to_fixed_val, load_vqgan

X_RAY_TARGET_SIZE = (320, 320)
NUM_SLICES = 128


class Inference:
    def __init__(self, config_path: str, ckpt_path: str, dev: device, loader: bool = True):
        torch.set_grad_enabled(False)

        self._device = dev
        self._loader = loader
        self._load_model(config_path, ckpt_path, dev)
        self._set_batch_presets(dev)

    def __call__(self, path_frontal: str, path_lateral: str) -> Tensor:
        if not os.path.exists(path_frontal): raise ValueError(f'\'{path_frontal}\' does not exist')
        if not os.path.exists(path_lateral): raise ValueError(f'\'{path_lateral}\' does not exist')

        batch = {k: self._batch_presets[k] for k in self._batch_presets}
        batch['PA'] = self._load_and_preprocess_xray(path_frontal, 'PA')
        batch['Lateral'] = self._load_and_preprocess_xray(path_lateral, 'Lateral')

        reconstructed = None

        range_ = tqdm(range(NUM_SLICES)) if self._loader else range(NUM_SLICES)
        for i in range_:
            batch['file_path_'] = [f'/axial_{i:03d}.h5']

            log = self._model.log_images(batch, split='val')

            reconstructed = (log['reconstructions'][:, 0]
                             if reconstructed is None
                             else torch.cat((reconstructed, log['reconstructions'][:, 0]), dim=0))

        return reconstructed

    def _set_batch_presets(self, dev: device):
        self._batch_presets = {
            'ctslice': torch.zeros((1, 128, 128, 3)).to(dev),
            'PA_cam': torch.tensor([0.0, 0.0],
                                   dtype=torch.float32,
                                   device=dev),
            'Lateral_cam': torch.tensor([math.pi / 2, math.pi / 2],
                                        dtype=torch.float32,
                                        device=dev),
        }

    def _load_model(self, config_path: str, checkpoint_path: str, dev: device):
        config = load_config(config_path, display=False)
        config['model']['params']['metadata']['encoder_params']['params']['zoom_resolution_div'] = 1
        config['model']['params']['metadata']['encoder_params']['params']['zoom_min_scale'] = 0

        config = change_config_file_to_fixed_val(deepcopy(config))

        self._model = (load_vqgan(config=config,
                           model_module=config['model']['target'],
                           ckpt_path=checkpoint_path)
                 .to(dev))

    def _load_and_preprocess_xray(self, path, cam_type, min_val=0, max_val=255) -> Tensor:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        # Apply camera-dependent transformations
        if cam_type == "PA":
            img = np.fliplr(img)
        elif cam_type == "Lateral":
            img = np.transpose(img, (1, 0))
            img = np.flipud(img)

        img = cv2.resize(img, X_RAY_TARGET_SIZE)

        img = np.expand_dims(img, -1)
        img = np.concatenate([img] * 3, axis=-1)
        img = (img - min_val) / (max_val - min_val)
        img = np.clip(img, 0, 1)

        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(self._device)

        return img_tensor
