from os import path

import torch
import pandas as pd
from argparse import Namespace, ArgumentParser

from tqdm import tqdm

from inference import Inference
from save_to_volume import save_nifti


CHECKPOINT_PATH = './checkpoints/PerX2CT.ckpt'
CONFIG_PATH = './configs/PerX2CT.yaml'
CT_EXTENSION = 'h5'
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def main(args: Namespace):
    reports = pd.read_csv(args.csv_reports_path)
    files = pd.read_csv(args.csv_projections_path)

    model = Inference(CONFIG_PATH, CHECKPOINT_PATH, DEVICE)

    updated_projections = pd.DataFrame(columns=files.columns)

    start = args.start_from if args.start_from is not None else 0
    end = args.end_at if args.end_at is not None else len(reports)
    reports_subset = reports.iloc[start:end]

    for uid in tqdm(reports_subset['uid']):
        frontal = files[(files['uid'] == uid) & (files['projection'] == 'Frontal')].iloc[0]
        lateral = files[(files['uid'] == uid) & (files['projection'] == 'Lateral')].iloc[0]

        path_front = str(path.join(args.projection_dir, frontal['filename']))
        path_lat   = str(path.join(args.projection_dir, lateral['filename']))

        volume = model(path_front, path_lat)
        filename = f'{uid}_ct_synthetic.{CT_EXTENSION}'
        save_nifti(volume.cpu().numpy(), path.join(args.save_dir, filename))

        row_df = pd.DataFrame([
            frontal.to_dict(),
            lateral.to_dict(),
            {'uid': uid, 'projection': 'Volume', 'filename': filename}
        ])

        updated_projections = pd.concat([updated_projections, row_df], ignore_index=True)

    updated_projections.to_csv(path.join(args.save_dir, 'projections_synth.csv'), index=False)

if __name__ == '__main__':
    parser = ArgumentParser(description='CT Inference')
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--projection_dir', type=str, required=True)
    parser.add_argument('--csv_reports_path', type=str, required=True)
    parser.add_argument('--csv_projections_path', type=str, required=True)
    parser.add_argument('--start_from', type=int, default=None)
    parser.add_argument('--end_at', type=int, default=None)

    main(parser.parse_args())