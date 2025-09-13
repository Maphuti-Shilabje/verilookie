
import os
import shutil
import random
from typing import Optional

from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Create the temp directory if it doesn't exist
if not os.path.exists("temp"):
    os.makedirs("temp")

app = FastAPI()

origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DetectionResult(BaseModel):
    confidence_score: float
    label: str
    explanation: str

class ImageVideoDetector:
    def detect(self, file_path: str) -> float:
        # Mocked detection
        return random.uniform(0.6, 0.95)

class AudioDetector:
    def detect(self, file_path: str) -> float:
        # Mocked detection
        return random.uniform(0.6, 0.95)

class DemoDetector:
    def detect(self, file_path: str) -> str:
        # Mocked detection based on some placeholder logic
        return random.choice(["Fake", "Real"])

class Detector:
    def __init__(self):
        self.image_video_detector = ImageVideoDetector()
        self.audio_detector = AudioDetector()
        self.demo_detector = DemoDetector()

    def run_detection(self, file_path: str) -> DetectionResult:
        # For the purpose of this demo, we'll just use the demo detector
        # and generate random scores.
        label = self.demo_detector.detect(file_path)
        if label == "Fake":
            confidence = self.image_video_detector.detect(file_path)
            explanation = "The model has detected inconsistencies in the media, suggesting it might be a deepfake."
        else:
            confidence = 1 - self.image_video_detector.detect(file_path)
            explanation = "The model has not found any significant signs of manipulation."

        return DetectionResult(
            confidence_score=confidence,
            label=label,
            explanation=explanation,
        )

detector = Detector()

@app.post("/detect", response_model=DetectionResult)
async def detect(file: UploadFile = File(...)):
    temp_file_path = os.path.join("temp", file.filename)
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = detector.run_detection(temp_file_path)

    # Clean up the temp file
    os.remove(temp_file_path)

    return result

@app.get("/")
def read_root():
    return {"message": "Verilookie API is running"}

