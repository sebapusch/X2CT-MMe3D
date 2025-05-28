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

async def predict(perx2ct_conn: Client,
                  frontal: np.ndarray,
                  lateral: np.ndarray):
    images = np.concat((frontal, lateral), axis=0)
    buf = io.BytesIO()
    np.save(buf, images, allow_pickle=False)
    perx2ct_conn.send(buf.getvalue())

    volume = perx2ct_conn.recv()
    data = {
        'frontal': xray_pre_pipe(frontal).to(DEVICE),
        'lateral': xray_pre_pipe(lateral).to(DEVICE),
        'ct': ct_pre_pipe(volume).to(DEVICE),
    }
    
    out = model(data)
    out = out.cpu().numpy()
    
    return out