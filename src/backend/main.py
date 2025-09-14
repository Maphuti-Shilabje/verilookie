
import os
import shutil
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import torch

# Import NVIDIA client
try:
    from nvidia_client import NvidiaAIClient
    nvidia_client = NvidiaAIClient()
    NVIDIA_API_AVAILABLE = True
except Exception as e:
    print(f"Warning: NVIDIA API not available: {e}")
    nvidia_client = None
    NVIDIA_API_AVAILABLE = False

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
    processor = None
    model = None

class DetectionResult(BaseModel):
    confidence_score: float
    label: str
    explanation: str

class AIGeneratedDetectionResult(BaseModel):
    confidence_score: float
    label: str
    explanation: str
    is_ai_generated: bool

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

            # Fix the label mapping - the model returns "Fake" for real images and "Realism" for fake images (reversed)
            if label == "Fake":
                label = "Real"
                explanation = "The model has not found any significant signs of manipulation."
            elif label == "Realism":
                label = "Fake"
                explanation = "The model has detected inconsistencies in the media, suggesting it might be a deepfake."
            else:
                # Handle any other labels the model might return
                # Default to Real for unknown labels to be safe
                label = "Real"
                explanation = f"The model detected this as potentially real with label: {label}"

            return DetectionResult(
                confidence_score=confidence_score,
                label=label,
                explanation=explanation,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error during detection: {str(e)}")
    
    def detect_ai_generated(self, image_path: str) -> AIGeneratedDetectionResult:
        """Detect if an image is AI-generated using NVIDIA API"""
        if not NVIDIA_API_AVAILABLE or nvidia_client is None:
            raise HTTPException(status_code=503, detail="NVIDIA API is not available")
        
        try:
            result = nvidia_client.detect_ai_generated_image(image_path)
            
            # Check if the detection was successful
            if not result.get("success", False):
                raise HTTPException(status_code=500, detail=f"AI detection failed: {result.get('error', 'Unknown error')}")
            
            # Extract the detection data
            ai_probability = result.get("ai_probability", 0.0)
            
            # Apply proper confidence calculation based on threshold logic
            # If score >= 0.5: AI-generated, confidence = score
            # If score < 0.5: Not AI-generated, confidence = 1 - score
            # If score == 0.5: Uncertain, confidence = 0.5
            if ai_probability > 0.5:
                is_ai_generated = True
                confidence_score = ai_probability
                label = "AI Generated"
                explanation = "This image has been detected as AI-generated based on artifacts and patterns characteristic of generative models."
            elif ai_probability < 0.5:
                is_ai_generated = False
                confidence_score = 1 - ai_probability
                label = "Not AI Generated"
                explanation = "This image does not show strong signs of being AI-generated."
            else:  # ai_probability == 0.5
                is_ai_generated = False  # Default to not AI-generated for uncertain cases
                confidence_score = 0.5
                label = "Uncertain"
                explanation = "The model is uncertain about whether this image is AI-generated or not."
            
            return AIGeneratedDetectionResult(
                confidence_score=confidence_score,
                label=label,
                explanation=explanation,
                is_ai_generated=is_ai_generated
            )
                
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error during AI-generated detection: {str(e)}")

    def detect_video(self, video_path: str, frames_to_process: int = 5) -> DetectionResult:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise HTTPException(status_code=500, detail="Error opening video file")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        results = []
        frame_interval = max(1, frame_count // frames_to_process)

        for i in range(frames_to_process):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * frame_interval)
            ret, frame = cap.read()
            if not ret:
                continue

            temp_frame_path = os.path.join("temp", f"frame_{i}.jpg")
            cv2.imwrite(temp_frame_path, frame)
            
            try:
                result = self.detect_image(temp_frame_path)
                results.append(result)
            finally:
                os.remove(temp_frame_path)

        cap.release()

        if not results:
            return DetectionResult(confidence_score=0.0, label="Error", explanation="Could not process any frames from the video.")

        fake_count = sum(1 for r in results if r.label == "Fake")
        real_count = len(results) - fake_count

        if fake_count > real_count:
            final_label = "Fake"
            avg_confidence = sum(r.confidence_score for r in results if r.label == "Fake") / fake_count if fake_count > 0 else 0.0
            explanation = f"{fake_count} out of {len(results)} analyzed frames were flagged as suspicious, suggesting the video may be a deepfake."
        else:
            final_label = "Real"
            avg_confidence = sum(r.confidence_score for r in results if r.label == "Real") / real_count if real_count > 0 else 1.0
            explanation = "The majority of analyzed frames appear to be authentic."

        return DetectionResult(confidence_score=avg_confidence, label=final_label, explanation=explanation)

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
        elif file.content_type.startswith("video/"):
            result = detector.detect_video(temp_file_path)
        else:
            result = DetectionResult(
                confidence_score=0.0,
                label="Not Supported",
                explanation="This file type is not supported. Please upload an image or video."
            )
    finally:
        os.remove(temp_file_path)

    return result

@app.post("/detect-ai-generated", response_model=AIGeneratedDetectionResult)
async def detect_ai_generated(file: UploadFile = File(...)):
    temp_file_path = os.path.join("temp", file.filename)
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = None
    try:
        if file.content_type.startswith("image/"):
            result = detector.detect_ai_generated(temp_file_path)
        else:
            raise HTTPException(status_code=400, detail="Only image files are supported for AI-generated detection.")
    finally:
        # Clean up the temp file
        os.remove(temp_file_path)

    return result

@app.get("/")
def read_root():
    return {"message": "Verilookie API is running"}
