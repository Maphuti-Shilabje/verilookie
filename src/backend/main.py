import os
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import torch

# Create the temp directory if it doesn't exist
if not os.path.exists("temp"):
    os.makedirs("temp")

app = FastAPI()

# CORS Middleware
origins = ["http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model and Processor
MODEL_NAME = "prithivMLmods/Deep-Fake-Detector-v2-Model"

try:
    processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
    model = ViTForImageClassification.from_pretrained(MODEL_NAME)
except Exception as e:
    print(f"Error loading model: {e}")
    # Handle model loading failure gracefully
    processor = None
    model = None

class DetectionResult(BaseModel):
    confidence_score: float
    label: str
    explanation: str

class Detector:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor

    def detect_image(self, image_path: str) -> DetectionResult:
        if self.model is None or self.processor is None:
            raise HTTPException(status_code=503, detail="Model is not available")

        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt")
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            confidence, predicted_class_idx = torch.max(probabilities, dim=1)

            label = self.model.config.id2label[predicted_class_idx.item()]
            confidence_score = confidence.item()

            if label == "Realism":
                label = "Real"
                explanation = "The model has not found any significant signs of manipulation."
            else: # Deepfake
                label = "Fake"
                explanation = "The model has detected inconsistencies in the media, suggesting it might be a deepfake."

            return DetectionResult(
                confidence_score=confidence_score,
                label=label,
                explanation=explanation,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error during detection: {str(e)}")

detector = Detector(model, processor)

@app.post("/detect", response_model=DetectionResult)
async def detect(file: UploadFile = File(...)):
    temp_file_path = os.path.join("temp", file.filename)
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = None
    try:
        if file.content_type.startswith("image/"):
            result = detector.detect_image(temp_file_path)
        else:
            result = DetectionResult(
                confidence_score=0.0,
                label="Not Supported",
                explanation="This file type is not supported for detection. Please upload an image."
            )
    finally:
        # Clean up the temp file
        os.remove(temp_file_path)

    return result

@app.get("/")
def read_root():
    return {"message": "Verilookie API is running"}