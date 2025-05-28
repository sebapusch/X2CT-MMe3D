import subprocess
import time
from multiprocessing.connection import Client

import numpy as np
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io

from api.inference import predict


proc_b = subprocess.Popen([
    '/home/sebastianp/Programs/miniconda3/envs/perx2ct/bin/python',
    '../perx2ct/PerX2CT/listener.py'
])
time.sleep(1)

conn = Client(('localhost', 6000))
buf = io.BytesIO()
np.save(buf, np.random.rand(2,2), allow_pickle=False)

app = FastAPI()



@app.get("/")
def read_root():
    return {"message": "API is running!"}

@app.post("/predict")
async def predict_endpoint(
    frontal: UploadFile = File(...),
    lateral: UploadFile = File(...)
):
    frontal_img = Image.open(io.BytesIO(await frontal.read())).convert("L")
    lateral_img = Image.open(io.BytesIO(await lateral.read())).convert("L")

    result = predict(None,
                     np.array(frontal_img),
                     np.array(lateral_img))
    return result

