import random
from argparse import Namespace, ArgumentParser, BooleanOptionalAction
from datetime import datetime

import numpy as np
import torch
from sklearn.metrics import (precision_score, recall_score,
                             f1_score, roc_curve, auc)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import wandb

from x2ct_mme3d.data.dataset import X2CTDataset, XRayDataset
from x2ct_mme3d.models.classifiers import X2CTMMe3D, BiplanarCheXNet
from x2ct_mme3d.utils.early_stopping import EarlyStopping

RANDOM_SEED = 55

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

EPOCHS = 30
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
TEST_SIZE = 0.1
PATIENCE = 4
CHESTX_PATH = './models/checkpoints_/chexnet.pth.tar'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _load_dataset(args: Namespace) -> (DataLoader, DataLoader):
    if args.baseline_model:
        dataset = XRayDataset(
            reports_csv_path=args.reports,
            projections_csv_path=args.projections,
            xray_dir=args.xrays,
        )
    else:
        dataset = X2CTDataset(
            reports_csv_path=args.reports,
            projections_csv_path=args.projections,
            xray_dir=args.xrays,
            ct_dir=args.cts,
        )

    all_labels = np.array([dataset[i][1].item() for i in range(len(dataset))])
    train_ixs, val_ixs = train_test_split(
        range(len(dataset)),
        test_size=args.test_size,
        stratify=all_labels,
        random_state=42
    )

    train_loader = DataLoader(
        Subset(dataset, train_ixs),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        Subset(dataset, val_ixs),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader

def train_one_epoch(model: nn.Module, params: dict) -> float:
    model.train()
    running_loss = 0.0

    for inputs, labels in tqdm(params['train']):
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        labels = labels.float().unsqueeze(1).to(DEVICE)

        params['optimizer'].zero_grad()
        outputs = model(inputs)
        loss = params['loss'](outputs, labels)
        loss.backward()
        params['optimizer'].step()
        running_loss += loss.item()
    return running_loss / len(params['train'])

def evaluate(model: nn.Module, params: dict) -> (float, dict):
    model.eval()
    total_loss = 0.0

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in params['val']:
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            labels = labels.float().unsqueeze(1).to(DEVICE)

            outputs = model(inputs)
            loss = params['loss'](outputs, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    avg_loss = total_loss / len(params['val'])

    preds_cat = torch.cat(all_preds).numpy()
    probs_cat = torch.cat(all_probs).numpy()
    labels_cat = torch.cat(all_labels).numpy()

    fpr, tpr, _ = roc_curve(labels_cat, probs_cat)

    metrics = {'accuracy': (preds_cat == labels_cat).mean(),
               'precision': precision_score(labels_cat, preds_cat, zero_division=0.0),
               'recall': recall_score(labels_cat, preds_cat, zero_division=0.0),
               'f1': f1_score(labels_cat, preds_cat, zero_division=0.0),
               'auc': auc(fpr, tpr)}

    return avg_loss, metrics

def _load_model(args: Namespace) -> nn.Module:
    if args.baseline_model:
        return BiplanarCheXNet(args.pretrained)
    return X2CTMMe3D(args.pretrained)

def main(args: Namespace):
    train, val = _load_dataset(args)
    model = _load_model(args)
    model.to(DEVICE)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if args.wandb:
        wandb.init(project="x2ct-med3d",
                   name=f'{args.model_prefix}-{timestamp}',
                   config={
                        "epochs": EPOCHS,
                        "batch_size": BATCH_SIZE,
                        "learning_rate": LEARNING_RATE,
                        "architecture": "X2CTMMe3D"
                    })
        wandb.watch_called = False  # Avoid duplicate warnings
        wandb.watch(model, log="all", log_freq=100)

    optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=1e-4)

    params = {
        'train': train,
        'val': val,
        'loss': torch.nn.BCEWithLogitsLoss(),
        'optimizer': optimizer,
    }

    early_stop = EarlyStopping(args.patience)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    best_vloss = float('inf')
    best_f1 = 0.0

    print('Training on device:', DEVICE)

    for epoch in range(args.epochs):
        print("-------------------")
        print(f"EPOCH {epoch + 1}:")
        print("-------------------")

        train_loss = train_one_epoch(model, params)
        val_loss, metrics = evaluate(model, params)

        print(f'LOSS train {train_loss:.4f} valid {val_loss:.4f}')
        print({f'{metric}: {value}' for metric, value in metrics.items() })

        if args.wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                **{f'val_{metric}': value for metric, value in metrics.items() }
            })

        if val_loss < best_vloss or metrics['f1'] > best_f1:
            best_vloss = min(val_loss, best_vloss)
            best_f1 = max(best_f1, metrics['f1'])
            model_path = f'models/{args.model_prefix}_{timestamp}_epoch{epoch}'
            torch.save(model.state_dict(), model_path)
            print(f"Saved new best model to {model_path}")

        if early_stop(val_loss):
            print(f'triggered early stop as validation loss has not been increasing for {args.patience + 1} epochs')
            break


if __name__ == '__main__':
    parser = ArgumentParser(description='Train CT')
    parser.add_argument('--reports', type=str, default='./data/processed/indiana_reports.sample.csv')
    parser.add_argument('--projections', type=str, default='./data/processed/indiana_projections.csv')
    parser.add_argument('--xrays', type=str, required=True)
    parser.add_argument('--cts', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--test-size', type=float, default=TEST_SIZE)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--patience', type=float, default=PATIENCE)
    # (--no-pretrained to not initialize pretrained weights)
    parser.add_argument('--pretrained', default=True, action=BooleanOptionalAction)
    # (--no-wandb to disable wandb logging)
    parser.add_argument('--wandb', default=True, action=BooleanOptionalAction)
    parser.add_argument('--baseline-model', default=False, action=BooleanOptionalAction)
    parser.add_argument('--model-prefix', type=str, default='x2ct')

    main(parser.parse_args())