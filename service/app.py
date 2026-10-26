from fastapi import FastAPI
import torch
from fastapi import UploadFile,HTTPException
from PIL import Image,UnidentifiedImageError
from io import BytesIO
from cifar_resnet.data import preprocess

import numpy as np
from cifar_resnet.model import ResNetCIFAR
from cifar_resnet.train import load_checkpoint
device = "cpu"

model = ResNetCIFAR(3)
load_checkpoint("model/resnet20.pt", model, None, device)
model.eval()
print(sum(p.numel() for p in model.parameters()),model.training)
mean = np.load("model/pixel_mean_50k.npy")
class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]
app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict_image(file: UploadFile):
    raw = await file.read()
    try:
        img = Image.open(BytesIO(raw)).convert("RGB")

    except UnidentifiedImageError:
        raise HTTPException(status_code=400,detail="Uploaded file is not a valid image")
    if img.size != (32,32):
        img = img.resize((32,32))
    arr = np.asarray(img)
    x = preprocess(arr,mean=mean).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x),dim=1)
        values,indices = torch.topk(probs,3)
        result_list = []
        for value, idx in zip(values[0], indices[0]):
            result_list.append({
            "class": class_names[idx.item()],
            "prob": value.item()
                                })

    return {"top3": result_list}

