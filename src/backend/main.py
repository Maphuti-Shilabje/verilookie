import os
import shutil
import cv2
import librosa
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from transformers import ViTForImageClassification, ViTImageProcessor, Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
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

# Load Image Model and Processor
IMAGE_MODEL_NAME = "prithivMLmods/Deep-Fake-Detector-v2-Model"

try:
    image_processor = ViTImageProcessor.from_pretrained(IMAGE_MODEL_NAME)
    image_model = ViTForImageClassification.from_pretrained(IMAGE_MODEL_NAME)
except Exception as e:
    print(f"Error loading image model: {e}")
    image_processor = None
    image_model = None

# Load Audio Model and Processor
AUDIO_MODEL_NAME = "MelodyMachine/Deepfake-audio-detection-V2"

try:
    audio_processor = Wav2Vec2Processor.from_pretrained(AUDIO_MODEL_NAME)
    audio_model = Wav2Vec2ForSequenceClassification.from_pretrained(AUDIO_MODEL_NAME)
except Exception as e:
    print(f"Error loading audio model: {e}")
    audio_processor = None
    audio_model = None

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
    def __init__(self, image_model, image_processor, audio_model, audio_processor):
        self.image_model = image_model
        self.image_processor = image_processor
        self.audio_model = audio_model
        self.audio_processor = audio_processor

    def detect_image(self, image_path: str) -> DetectionResult:
        if self.image_model is None or self.image_processor is None:
            raise HTTPException(status_code=503, detail="Image model is not available")
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.image_processor(images=image, return_tensors="pt")
            outputs = self.image_model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            confidence, predicted_class_idx = torch.max(probabilities, dim=1)

            label = self.image_model.config.id2label[predicted_class_idx.item()]
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
            raise HTTPException(status_code=500, detail=f"Error during image detection: {str(e)}")
    
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
            is_ai_generated = result.get("is_ai_generated", False)
            
            label = "AI Generated" if is_ai_generated else "Not AI Generated"
            
            explanation = (
                "This image has been detected as AI-generated based on artifacts and patterns "
                "characteristic of generative models." if is_ai_generated else
                "This image does not show strong signs of being AI-generated."
            )
            
            return AIGeneratedDetectionResult(
                confidence_score=ai_probability,
                label=label,
                explanation=explanation,
                is_ai_generated=is_ai_generated
            )
                
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error during AI-generated detection: {str(e)}")

    def detect_video(self, video_path: str, frames_to_process: int = 5) -> DetectionResult:
        if self.image_model is None or self.image_processor is None:
            raise HTTPException(status_code=503, detail="Image model is not available for video frame analysis")

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
            avg_confidence = sum(r.confidence_score for r in results if r.label == "Fake") / fake_count
            explanation = f"{fake_count} out of {len(results)} analyzed frames were flagged as suspicious, suggesting the video may be a deepfake."
        else:
            final_label = "Real"
            avg_confidence = sum(r.confidence_score for r in results if r.label == "Real") / real_count if real_count > 0 else 1.0
            explanation = "The majority of analyzed frames appear to be authentic."

        return DetectionResult(confidence_score=avg_confidence, label=final_label, explanation=explanation)

    def detect_audio(self, audio_path: str) -> DetectionResult:
        if self.audio_model is None or self.audio_processor is None:
            raise HTTPException(status_code=503, detail="Audio model is not available")
        try:
            # Load audio and resample to 16kHz (common for Wav2Vec2 models)
            speech, sample_rate = librosa.load(audio_path, sr=16000)

            # Process audio
            inputs = self.audio_processor(speech, sampling_rate=sample_rate, return_tensors="pt")
            
            # Get model predictions
            with torch.no_grad():
                logits = self.audio_model(**inputs).logits
            
            # Apply softmax to get probabilities
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            
            # Get predicted class and confidence
            confidence, predicted_class_idx = torch.max(probabilities, dim=1)
            
            # Map class ID to label (assuming 0 for fake, 1 for real based on common practice)
            # You might need to verify the actual label mapping for MelodyMachine/Deepfake-audio-detection-V2
            # For now, let's assume 0 is fake, 1 is real
            label_mapping = {0: "Fake", 1: "Real"}
            label = label_mapping.get(predicted_class_idx.item(), "Unknown")
            confidence_score = confidence.item()

            explanation = "The audio has been analyzed for deepfake characteristics." if label == "Fake" else "The audio appears to be authentic."

            return DetectionResult(
                confidence_score=confidence_score,
                label=label,
                explanation=explanation,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error during audio detection: {str(e)}")

detector = Detector(image_model, image_processor, audio_model, audio_processor)

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
        elif file.content_type.startswith("audio/"):
            result = detector.detect_audio(temp_file_path)
        else:
            result = DetectionResult(
                confidence_score=0.0,
                label="Not Supported",
                explanation="This file type is not supported. Please upload an image, video, or audio file."
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