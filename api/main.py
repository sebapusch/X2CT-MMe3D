from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io

from api.inference import predict

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
    result = predict(frontal_img, lateral_img)
    return result