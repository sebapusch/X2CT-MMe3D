import random
from argparse import Namespace, ArgumentParser, BooleanOptionalAction
from datetime import datetime

import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import wandb

from x2ct_mme3d.data.dataset import X2CTDataset
from x2ct_mme3d.models.x2ct_mme3d import X2CTMMe3D

RANDOM_SEED = 55

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

EPOCHS = 30
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
TEST_SIZE = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _load_dataset(args: Namespace) -> (DataLoader, DataLoader):
    dataset = X2CTDataset(
        args.reports,
        args.projections,
        args.xrays,
        args.cts,
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

def train_one_epoch(model: X2CTMMe3D, params: dict) -> float:
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

def evaluate(model: X2CTMMe3D, params: dict) -> (float, dict):
        model.eval()
        total_loss = 0.0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in params['val']:
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                labels = labels.float().unsqueeze(1).to(DEVICE)

                outputs = model(inputs)
                loss = params['loss'](outputs, labels)
                total_loss += loss.item()

                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()

                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        avg_loss = total_loss / len(params['val'])

        preds_cat = torch.cat(all_preds).numpy()
        labels_cat = torch.cat(all_labels).numpy()

        metrics = {'accuracy': (preds_cat == labels_cat).mean(),
                   'precision': precision_score(labels_cat, preds_cat, zero_division=0.0),
                   'recall': recall_score(labels_cat, preds_cat, zero_division=0.0),
                   'f1': f1_score(labels_cat, preds_cat, zero_division=0.0)}

        return avg_loss, metrics

def main(args: Namespace):
    train, val = _load_dataset(args)
    model = X2CTMMe3D()
    model.to(DEVICE)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if args.wandb:
        wandb.init(project="x2ct-med3d",
                   name=f'{timestamp}',
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

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    best_vloss = float('inf')

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

        if val_loss < best_vloss:
            best_vloss = val_loss
            model_path = f'models/med3d_model_{timestamp}_epoch{epoch}'
            torch.save(model.state_dict(), model_path)
            print(f"Saved new best model to {model_path}")



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
    # (--no-wandb to disable wandb logging)
    parser.add_argument('--wandb', default=True, action=BooleanOptionalAction)
    main(parser.parse_args())