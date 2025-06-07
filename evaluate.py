import logging
from argparse import Namespace, ArgumentParser, BooleanOptionalAction
from os import path
import os
import re

import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from torch import nn
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm

from x2ct_mme3d.data.dataset import XRayDataset, X2CTDataset
from x2ct_mme3d.models.classifiers import BiplanarCheXNet, X2CTMMed3D


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8


def _extract_seed(name: str) -> int:
    match = re.search(r'(\d+)', name)
    return int(match.group(1)) if match else -1


def _load_model(checkpoint: str, baseline: bool) -> nn.Module:
    if baseline:
        model = BiplanarCheXNet(False)
    else:
        model = X2CTMMed3D(False)

    state_dict = torch.load(checkpoint)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model


def _load_dataset(args: Namespace) -> DataLoader:
    init = {
        'reports_csv_path': args.reports,
        'projections_csv_path': args.projections,
        'include_uid': True,
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


def evaluate(model: nn.Module, dataset: DataLoader) -> dict:
    labels  = np.array([])
    probs   = np.array([])
    preds   = np.array([])

    loss = BCEWithLogitsLoss()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(dataset):
            inputs = {k: v.to(DEVICE) if k in ['ct', 'frontal', 'lateral'] else v
                      for k, v in batch[0].items()}

            batch_labels = batch[1].float().to(DEVICE)
            outputs = model(inputs)

            total_loss += loss(outputs.squeeze(), batch_labels).item()
            batch_probs = torch.sigmoid(outputs).squeeze().cpu().numpy()

            labels  = np.concatenate((labels, batch_labels.cpu().numpy()))
            probs   = np.concatenate((probs, batch_probs))
            preds   = np.concatenate((preds, batch_probs > 0.5))

    return {
        'accuracy': (preds == labels).mean(),
        'precision': precision_score(labels, preds, zero_division=0.0),
        'recall': recall_score(labels, preds, zero_division=0.0),
        'f1': f1_score(labels, preds, zero_division=0.0),
        'auc': roc_auc_score(labels, probs),
        'loss': total_loss / len(dataset),
    }



def main(args: Namespace):
    assert path.exists(args.checkpoint)
    assert path.exists(args.reports)
    assert path.exists(args.projections)
    assert path.exists(path.dirname(args.csv_out_path))

    if os.path.isdir(args.checkpoint):
        checkpoints = [f for f in os.listdir(args.checkpoint)
                       if not f.startswith('.') and not path.isdir(f)]
        checkpoints.sort(key=_extract_seed)
    else:
        checkpoints = [args.checkpoint]

    logging.info(f'Located {len(checkpoints)} checkpoints')
    results = {}
    for k in ['seed', 'loss', 'accuracy', 'precision', 'recall', 'f1', 'auc']:
        results[k] = []

    dataset = _load_dataset(args)
    for checkpoint in checkpoints:
        seed = _extract_seed(checkpoint)
        logging.info(f'Evaluating \'{checkpoint}\' (seed: {seed})')

        results['seed'].append(_extract_seed(checkpoint))

        abs_path = path.join(args.checkpoint, checkpoint)
        model = _load_model(abs_path, args.baseline)

        metrics = evaluate(model, dataset)
        for k, v in metrics.items():
            logging.info(f'{k}: {v}')
            results[k].append(v)

    pd.DataFrame(results).to_csv(args.csv_out_path, index=False)
    logging.info(f'Stored results at \'{args.csv_out_path}\'')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

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