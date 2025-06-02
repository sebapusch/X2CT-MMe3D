import logging
from argparse import Namespace, ArgumentParser, BooleanOptionalAction
from os import path

import torch
from torch import nn
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm

from x2ct_mme3d.data.dataset import XRayDataset, X2CTDataset
from x2ct_mme3d.models.classifiers import BiplanarCheXNet, X2CTMMed3D


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8


def _load_model(args: Namespace) -> nn.Module:
    if args.baseline:
        model = BiplanarCheXNet(False)
    else:
        model = X2CTMMed3D(False)

    state_dict = torch.load(args.checkpoint)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model

def _load_dataset(args: Namespace) -> DataLoader:
    init = {
        'reports_csv_path': args.reports,
        'projections_csv_path': args.projections,
        'xray_dir': args.xray_dir,
    }
    if args.baseline:
        dataset = XRayDataset(**init)
    else:
        dataset = X2CTDataset(**init, ct_dir=args.ct_dir)

    return DataLoader(dataset,
                      shuffle=False,
                      batch_size=args.batch_size,
                      num_workers=4,
                      pin_memory=True)

def main(args: Namespace):
    assert path.exists(args.checkpoint)
    assert path.exists(args.reports)
    assert path.exists(args.projections)
    assert path.exists(path.dirname(args.out_dir))

    model = _load_model(args)
    dataset = _load_dataset(args)

    results = {
        'uids': [],
        'true': [],
        'pred': [],
    }

    logging.info(f'Evaluating {len(dataset)} samples...')
    with torch.no_grad():
        for inputs, labels in tqdm(dataset):
            inputs = {k: v.to(DEVICE) for k, v in ['ct', 'frontal', 'lateral']}

        output = model(inputs)

        results['uids'].extend(inputs['uids'])
        results['true'].extend(labels.cpu().numpy())
        results['pred'].extend(torch.sigmoid(output))

    pd.DataFrame(results).to_csv(args.csv_output_path, index=False)
    logging.info(f'Stored results at \'{args.csv_output_path}\'')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--reports', type=str, required=True)
    parser.add_argument('--projections', type=str, required=True)
    parser.add_argument('--ct-dir', type=str, required=True)
    parser.add_argument('--xray-dir', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--baseline', default=False, action=BooleanOptionalAction)
    parser.add_argument('--csv-out-path', type=str, required=True)

    main(parser.parse_args())