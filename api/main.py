import argparse
import os.path
import subprocess
import time
from contextlib import asynccontextmanager
from multiprocessing.connection import Client

import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io

from api.inference import Inference

LISTENER_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             '..', 'perx2ct', 'PerX2CT', 'listener.py')

class ListenerManager:
    def __init__(self, python_path: str, script_path: str, port: int = 6000):
        self.python_path = python_path
        self.script_path = script_path
        self.port = port
        self.process = None
        self.client = None

    def start(self):
        print('[client] Starting listener subprocess...')
        self.process = subprocess.Popen([self.python_path, self.script_path])
        for _ in range(20):
            try:
                self.client = Client(('localhost', self.port))
                print('[client] Connected to listener!')
                return
            except Exception:
                time.sleep(0.2)
        raise RuntimeError("Could not connect to listener")

    def stop(self):
        print("[client] Shutting down listener subprocess...")
        if self.process:
            self.process.terminate()


def create_app(checkpoint_path: str, python_path: str) -> FastAPI:
    listener = ListenerManager(python_path, LISTENER_PATH)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    predict = Inference(checkpoint_path, device)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        listener.start()
        predict.set_conn(listener.client)
        yield
        listener.stop()

    app = FastAPI(lifespan=lifespan)

    @app.post("/predict")
    async def predict_endpoint(
        frontal: UploadFile = File(...),
        lateral: UploadFile = File(...)
    ):
        frontal_img = Image.open(io.BytesIO(await frontal.read())).convert("L")
        lateral_img = Image.open(io.BytesIO(await lateral.read())).convert("L")
        return predict(frontal_img, lateral_img)

    return app


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Run FastAPI with image prediction endpoint")
    parser.add_argument('--checkpoint', type=str, default='../models/checkpoints/resnet18_20250523_084333_epoch13',
                        help='Path to the model checkpoint')
    parser.add_argument('--python_path', type=str, default='/home/sebastianp/Programs/miniconda3/envs/perx2ct/bin/python',
                        help='Python executable path to run listener')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host for FastAPI')
    parser.add_argument('--port', type=int, default=8000, help='Port for FastAPI')

    args = parser.parse_args()

    app = create_app(args.checkpoint, args.python_path)
    uvicorn.run(app, host=args.host, port=args.port)
