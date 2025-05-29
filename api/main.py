import argparse
import logging
import os.path
from argparse import Namespace
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import io

from pydantic import BaseModel

from api.inference import Inference
from api.perx2ct_client import PerX2CTClient

LISTENER_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             '..', 'perx2ct', 'PerX2CT', 'listener.py')

from enum import Enum

class Diagnosis(str, Enum):
    healthy = "healthy"
    disease = "disease"


class PredictResponse(BaseModel):
    probability: float
    diagnosis: Diagnosis

def _validate_image(img: Image.Image):
    if img.mode not in ['RGB', 'L']:
        raise ValueError('Only rgb or grayscale images are supported.')
    if img.size[0] < 512 or img.size[1] < 512:
        raise ValueError(f'Image size must be at least 512x512, {img.size} given.')


def create_app(args: Namespace) -> FastAPI:
    perx2ct_client = PerX2CTClient(
        args.perx2ct_python_path,
        LISTENER_PATH,
        args.perx2ct_model_path,
        args.perx2ct_config_path,
        args.perx2ct_port)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    predict = Inference(perx2ct_client, args.checkpoint, device)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        perx2ct_client.start_process()
        yield
        perx2ct_client.stop_process()

    app = FastAPI(lifespan=lifespan)

    @app.post("/predict",
              response_model=PredictResponse,
              description="Returns the model prediction on the passed uploaded xray pairs "
                          "(frontal + lateral), using synthetic generated ct scan information",
              )
    async def predict_endpoint(
        frontal: UploadFile = File(...),
        lateral: UploadFile = File(...)
    ):
        frontal_img = Image.open(io.BytesIO(await frontal.read()))
        lateral_img = Image.open(io.BytesIO(await lateral.read()))

        try:
            _validate_image(frontal_img)
            _validate_image(lateral_img)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))

        try:
            out = predict(frontal_img, lateral_img)
        except RuntimeError as e:
            raise HTTPException(status_code=500,
                                detail=f'Unexpected error encountered: {e}')

        return out

    return app


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="[server] [%(levelname)s] %(message)s"
    )

    parser = argparse.ArgumentParser(description="Run FastAPI with image prediction endpoint")
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host for FastAPI')
    parser.add_argument('--port', type=int, default=8000, help='Port for FastAPI')
    parser.add_argument('--checkpoint', type=str, help='Path to the model checkpoint')

    parser.add_argument('--perx2ct_model_path', type=str, help='perx2ct model checkpoint path')
    parser.add_argument('--perx2ct_python_path', type=str, help='Python executable path to run perx2ct listener')
    parser.add_argument('--perx2ct_config_path', type=str, help='perx2ct configuration path')
    parser.add_argument('--perx2ct_port', type=int, default=6000, help='perx2ct port')

    args_ = parser.parse_args()

    app_ = create_app(args_)
    uvicorn.run(app_, host=args_.host, port=args_.port)
