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

try:
    MODEL = tf.keras.models.load_model("../saved_models/1.keras")
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")

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

        # Log image shape and type for debugging
        logger.info(f"Image shape: {image.shape}, dtype: {image.dtype}")

        # Add preprocessing steps here

        image_batch = np.expand_dims(image, 0)
        prediction = MODEL.predict(image_batch)

        index = np.argmax(prediction[0])
        predicted_class = CLASS_NAMES[index]
        confidence = np.max(prediction[0])

        return {
            'class': predicted_class,
            'confidence': float(confidence)
        }
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)
