import io
from multiprocessing.connection import Client

import numpy as np
import torch
from torchvision import transforms
# from models.chexnet_model import build_chexnet  # Uncomment and adjust to your repo!
from PIL import Image
from x2ct_mme3d.data.preprocess import xray_pipeline, ct_pipeline
from x2ct_mme3d.models.x2ct_mme3d import X2CTMMe3D

xray_pre_pipe = xray_pipeline()
ct_pre_pipe = ct_pipeline()

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = X2CTMMe3D()
model.to(DEVICE)

# Define any preprocessing for your input images
def preprocess(frontal: Image.Image, lateral: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),   # Match your model input size
        transforms.ToTensor(),
    ])
    frontal_tensor = transform(frontal)
    lateral_tensor = transform(lateral)
    # Stack along the channel dimension [2, 224, 224]
    x = torch.cat([frontal_tensor, lateral_tensor], dim=0)
    x = x.unsqueeze(0)  # Add batch dimension: [1, 2, 224, 224]
    return x

def _send(conn: Client, data: np.ndarray):
    buf = io.BytesIO()
    np.save(buf, data, allow_pickle=False)
    conn.send(buf.getvalue())

def _receive(conn: Client) -> np.ndarray:
    data = conn.recv()
    return np.load(io.BytesIO(data))

def predict(perx2ct_conn: Client,
                  frontal: Image,
                  lateral: Image):
    _send(perx2ct_conn, np.array(frontal))
    _send(perx2ct_conn, np.array(lateral))

    volume = _receive(perx2ct_conn)
    volume = torch.from_numpy(volume).unsqueeze(0)

    data = {
        'frontal': xray_pre_pipe(frontal).unsqueeze(0).to(DEVICE),
        'lateral': xray_pre_pipe(lateral).unsqueeze(0).to(DEVICE),
        'ct': ct_pre_pipe(volume).unsqueeze(0).to(DEVICE),
    }

    with torch.no_grad():
        pred = model(data)

    probs = torch.sigmoid(pred)

    preds = (probs > 0.5).float()

    out = {
        'probability': float(probs[0][0]),
        'prediction': float(preds[0][0]),
    }

    return out