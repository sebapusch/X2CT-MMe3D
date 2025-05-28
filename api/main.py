import subprocess
import time
from contextlib import asynccontextmanager
from multiprocessing.connection import Client

import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io

from api.inference import predict


listener = None
conn = None

@asynccontextmanager
async def lifespan(_: FastAPI):
    global listener, conn

    print('[client] Starting listener subprocess...')
    listener = subprocess.Popen([
        '/home/sebastianp/Programs/miniconda3/envs/perx2ct/bin/python',
        '../perx2ct/PerX2CT/listener.py'
    ])

    for i in range(20):
        try:
            conn = Client(('localhost', 6000))
            print('[client] Connected to listener!')
            break
        except Exception:
            time.sleep(0.2)
    else:
        raise RuntimeError("Could not connect to listener")

    yield

    # Cleanup on shutdown
    print("Shutting down listener subprocess...")
    if listener:
        listener.terminate()

app = FastAPI(lifespan=lifespan)

@app.get("/")
def read_root():
    return {"message": "API is running!"}

@app.post("/predict")
async def predict_endpoint(
    frontal: UploadFile = File(...),
    lateral: UploadFile = File(...)
):
    print('[client] Received file')
    frontal_img = Image.open(io.BytesIO(await frontal.read())).convert("L")
    lateral_img = Image.open(io.BytesIO(await lateral.read())).convert("L")


    return predict(conn,
                     frontal_img,
                     lateral_img)

