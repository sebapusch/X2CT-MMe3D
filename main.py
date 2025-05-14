from datetime import datetime

import torch
from torch.utils.data import DataLoader, Subset

from x2ct_mme3d.data.dataset import XRayCTDataset
from x2ct_mme3d.models.med3d import X2CTMed3D, Med3DBackbone

EPOCHS = 5

dataset = XRayCTDataset(
    './data/processed/indiana_reports.test.csv',
    './data/processed/indiana_projections.csv',
    './data/processed/xrays',
    './data/processed/volumes',
)

ixs = range(len(dataset))
train_loader = DataLoader(Subset(dataset, ixs[:len(dataset) - 200]), batch_size=16, shuffle=True)
val_loader   = DataLoader(Subset(dataset, ixs[len(dataset) - 200:]),   batch_size=16, shuffle=True)

model = X2CTMed3D()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.CrossEntropyLoss()

def train_one_epoch(epoch_index):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(train_loader):
        # Every data instance is an input + label pair
        inputs, labels = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs['ct'])

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            # tb_x = epoch_index * len(loader) + i + 1
            # tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss

# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
epoch_number = 0

best_vloss = 1_000_000.

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number)


    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(val_loader):
            vinputs, vlabels = vdata
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))


    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'models/med3d_model_{}_{}'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)

    epoch_number += 1

# batch1 = next(iter(loader))
#
# print(torch.sigmoid(model(batch1[0]['ct'])))
# print(batch1[1])

# print(batch1[0]['ct'].shape)
#
# features = model(batch1[0]['ct'])
#
# print(features.shape)
# print("Mean:", features.mean())
# print("Std:", features.std())
# print("Max:", features.max())
# print("Min:", features.min())