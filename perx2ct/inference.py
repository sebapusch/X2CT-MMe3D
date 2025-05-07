import argparse
import glob
import math
import time
from copy import deepcopy

import imageio
import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from main import instantiate_from_config
from main_test import load_config, change_config_file_to_fixed_val, load_vqgan
from save_to_volume import save_nifti

def load_and_preprocess_xray(path, cam_type, min_val=0, max_val=255) -> Tensor:
    img = imageio.imread(path)

    if cam_type == "PA":
        img = np.fliplr(img)
    elif cam_type == "Lateral":
        img = np.transpose(img, (1, 0))
        img = np.flipud(img)

    img = np.expand_dims(img, -1)              # Shape: [H, W, 1]
    img = np.concatenate([img]*3, axis=-1)     # Shape: [H, W, 3]

    img = (img - min_val) / (max_val - min_val)
    img = np.clip(img, 0, 1)

    img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)

    return img_tensor

@torch.no_grad()
def test(dicts, images, save_dir):
    dummy = torch.zeros((1, 128, 128, 3)).to(dicts['device'])
    pa_cam = torch.tensor([0.0, 0.0], dtype=torch.float32, device=dicts['device'])
    lateral_cam = torch.tensor([math.pi / 2, math.pi / 2], dtype=torch.float32, device=dicts['device'])

    for path_frontal, path_lateral in images:
        start = time.time()

        frontal = load_and_preprocess_xray(path_frontal, 'PA').to(dicts['device'])
        lateral = load_and_preprocess_xray(path_lateral, 'Lateral').to(dicts['device'])

        reconstructed = None

        for i in tqdm(range(128)):
            batch = {
                'file_path_': [f'./experiment/ct128_CTSlice/5/ct/axial_{i:03d}.h5'],
                'ctslice': dummy,
                'PA': frontal,
                'Lateral': lateral,
                "PA_cam": pa_cam,
                "Lateral_cam": lateral_cam,
            }

            log = dicts['model'].log_images(batch, split='val')

            reconstructed = (log['reconstructions'][:, 0]
                             if reconstructed is None
                             else torch.cat((reconstructed, log['reconstructions'][:, 0]), dim=0))

        save_nifti(reconstructed.cpu().numpy(), f'{save_dir}/out.nii.gz')

        print(f'inference time: {time.time() - start:.4f} seconds')

@torch.no_grad()
def main_test(args):
    saved_log_root = args.save_dir
    ckpt_name = args.ckpt_path.split('/')[-1] if args.ckpt_path else args.ckpt_name
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    config_file = args.config_path or glob.glob(f"{saved_log_root}/configs/*-project.yaml")[0]
    config = load_config(config_file, display=False)
    config['model']['params']['metadata']['encoder_params']['params']['zoom_resolution_div'] = 1
    config['model']['params']['metadata']['encoder_params']['params']['zoom_min_scale'] = 0

    if args.sub_batch_size is None:
        args.sub_batch_size = config['input_ct_res']
    assert config['input_ct_res'] % args.sub_batch_size == 0

    config = change_config_file_to_fixed_val(deepcopy(config))

    model_file = args.ckpt_path or f"{saved_log_root}/checkpoints/{ckpt_name}"
    model = load_vqgan(config=config, model_module=config['model']['target'], ckpt_path=model_file).to(DEVICE)
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()

    dicts = {
        'saved_log_root': saved_log_root,
        'sub_folder': ckpt_name.replace('.ckpt', ''),
        'data': data,
        'model': model,
        'device': DEVICE,
        'val_test': args.val_test,
        'sub_batch_size': args.sub_batch_size
    }

    test(dicts, [[
       './experiment/ct128_plastimatch_xray/1_xray1.png',
       './experiment/ct128_plastimatch_xray/1_xray2.png',
    ]], args.save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CT Inference')
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--val_test', type=str, default='val')
    parser.add_argument('--sub_batch_size', type=int, default=1)
    parser.add_argument('--config_path', type=str)
    parser.add_argument('--ckpt_path', type=str)
    parser.add_argument('--ckpt_name', type=str, default='PerX2CT.ckpt')
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    main_test(args)
