from datetime import datetime
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np

from x2ct_mme3d.data.dataset import XRayCTDataset
from x2ct_mme3d.models.med3d import X2CTMed3D

# Hyperparameters
EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset
dataset = XRayCTDataset(
    './data/processed/indiana_reports.test.csv',
    './data/processed/indiana_projections.csv',
    './data/processed/xrays',
    './data/processed/volumes',
)

# Use sklearn to split indices
all_indices = np.arange(len(dataset))
all_labels = np.array([dataset[i][1].item() for i in all_indices])
train_ixs, val_ixs = train_test_split(
    all_indices,
    test_size=0.2,
    stratify=all_labels,
    random_state=42
)

train_loader = DataLoader(Subset(dataset, train_ixs), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(Subset(dataset, val_ixs), batch_size=BATCH_SIZE, shuffle=False)

# Model
model = X2CTMed3D().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = torch.nn.BCEWithLogitsLoss()

def train_one_epoch():
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader):
        inputs = inputs['ct'].to(DEVICE)
        labels = labels.float().unsqueeze(1).to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)

def evaluate():
    model.eval()
    running_vloss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for vinputs, vlabels in val_loader:
            vinputs = vinputs['ct'].to(DEVICE)
            vlabels = vlabels.float().unsqueeze(1).to(DEVICE)

            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss.item()

            # Binary prediction: sigmoid + threshold
            probs = torch.sigmoid(voutputs)
            preds = (probs > 0.5).int()
            labels = vlabels.int()

            correct += (preds == labels).sum().item()
            total += labels.numel()

            # Debugging sample
            print("Preds:", preds.view(-1)[:5].cpu().numpy(),
                  "Labels:", labels.view(-1)[:5].cpu().numpy(),
                  "Probs:", probs.view(-1)[:5].cpu().numpy())

    avg_vloss = running_vloss / len(val_loader)
    accuracy = correct / total if total > 0 else 0.0
    return avg_vloss, accuracy

# Training Loop
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
best_vloss = float('inf')

for epoch in range(EPOCHS):
    print(f"\nEPOCH {epoch + 1}:")
    train_loss = train_one_epoch()
    val_loss, val_accuracy = evaluate()

    print(f"LOSS train {train_loss:.4f} valid {val_loss:.4f} | Accuracy: {val_accuracy:.4f}")

    if val_loss < best_vloss:
        best_vloss = val_loss
        model_path = f'models/med3d_model_{timestamp}_epoch{epoch}'
        torch.save(model.state_dict(), model_path)
        print(f"✅ Saved new best model to {model_path}")
