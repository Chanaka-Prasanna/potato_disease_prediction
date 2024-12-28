import requests
from fastapi import FastAPI, File, UploadFile, HTTPException
from io import BytesIO
from PIL import Image
import numpy as np
import uvicorn
import tensorflow as tf

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

endpoint = "http://localhost:8501/v1/models/potatoes_model:predict"



CLASS_NAMES = ['Early Blight', 'Late Blight', 'Healthy']


@app.get('/ping')
async def ping():
    return {"message": "Hello, I am alive"}


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    try:
        image = read_file_as_image(await file.read())

        # Add preprocessing steps here
        image_batch = np.expand_dims(image, 0)

        json_data = {
            "instances":image_batch.tolist()
        }

        response = requests.post(endpoint,json=json_data)
        prediction = np.array(response.json()["predictions"][0])
        predicted_class=CLASS_NAMES[np.argmax(prediction)]
        confidence = np.max(prediction)
        return {
            'class': predicted_class,
            'confidence': float(confidence)
        }
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    uvicorn.run("main-tf-serving:app", host="localhost", port=8000, reload=True)
