from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
from inference import extract_fields

app = FastAPI()

@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image.save("temp_upload.jpg")
    result = extract_fields("temp_upload.jpg")
    return result