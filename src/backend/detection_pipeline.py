import torch
from fastapi import HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import os

class UnifiedDetectionResult(BaseModel):
    type: str
    confidence: float
    explanation: str

class DetectionPipeline:
    def __init__(self, detector):
        self.detector = detector
    
    def analyze_image(self, image_path: str) -> UnifiedDetectionResult:
        """
        Unified image authenticity detection pipeline.
        
        Step 1: Run the Deepfake Analyzer.
        If the image is flagged as a deepfake, return the result immediately.
        If the Deepfake Analyzer cannot confidently classify the image, forward it to the AI Image Detector.
        
        Step 2: Run the AI Image Detector only if the Deepfake Analyzer is inconclusive.
        
        Step 3: Return a single, unified result to the user.
        """
        try:
            # Step 1: Deepfake analysis
            deepfake_result = self.detector.detect_image(image_path)
            
            # Check if deepfake detector is confident
            if deepfake_result.confidence_score >= 0.7:  # Higher threshold for deepfake detection
                if deepfake_result.label == "Fake":
                    return UnifiedDetectionResult(
                        type="Deepfake",
                        confidence=round(deepfake_result.confidence_score * 100, 2),
                        explanation="The image appears manipulated. Detected as a deepfake."
                    )
                # If it's real and confident, we can stop here for real images
                elif deepfake_result.label == "Real" and deepfake_result.confidence_score >= 0.8:
                    return UnifiedDetectionResult(
                        type="Authentic",
                        confidence=round(deepfake_result.confidence_score * 100, 2),
                        explanation="The image appears authentic and not manipulated."
                    )
            
            # Step 2: AI generation analysis (if deepfake detection was inconclusive)
            # Try to run AI detection if NVIDIA API is available
            try:
                ai_result = self.detector.detect_ai_generated(image_path)
                
                # Apply proper confidence calculation based on threshold logic
                if ai_result.confidence_score >= 0.5:
                    return UnifiedDetectionResult(
                        type="AI-generated",
                        confidence=round(ai_result.confidence_score * 100, 2),
                        explanation="The image was generated using AI (not authentic)."
                    )
                else:
                    return UnifiedDetectionResult(
                        type="Authentic",
                        confidence=round((1 - ai_result.confidence_score) * 100, 2),
                        explanation="The image does not appear to be AI-generated or a deepfake."
                    )
            except HTTPException:
                # If AI detection is not available, fall back to deepfake result
                if deepfake_result.label == "Fake":
                    return UnifiedDetectionResult(
                        type="Deepfake",
                        confidence=round(deepfake_result.confidence_score * 100, 2),
                        explanation="The image appears manipulated. Detected as a deepfake."
                    )
                else:
                    return UnifiedDetectionResult(
                        type="Authentic",
                        confidence=round(deepfake_result.confidence_score * 100, 2),
                        explanation="The image appears authentic and not manipulated."
                    )
                    
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error during unified detection: {str(e)}")
    
    def analyze_video(self, video_path: str, frames_to_process: int = 5) -> UnifiedDetectionResult:
        """
        Analyze a video using the unified detection pipeline.
        """
        try:
            # For videos, we'll use the video detection method which already processes multiple frames
            video_result = self.detector.detect_video(video_path, frames_to_process)
            
            # Apply the same logic as for images
            if video_result.confidence_score >= 0.7:
                if video_result.label == "Fake":
                    return UnifiedDetectionResult(
                        type="Deepfake",
                        confidence=round(video_result.confidence_score * 100, 2),
                        explanation="The video appears manipulated. Detected as a deepfake."
                    )
                elif video_result.label == "Real" and video_result.confidence_score >= 0.8:
                    return UnifiedDetectionResult(
                        type="Authentic",
                        confidence=round(video_result.confidence_score * 100, 2),
                        explanation="The video appears authentic and not manipulated."
                    )
            
            # For inconclusive video results, we'll classify as authentic for now
            # A more sophisticated approach would involve frame-by-frame analysis
            return UnifiedDetectionResult(
                type="Authentic",
                confidence=round(video_result.confidence_score * 100, 2),
                explanation="The video does not show strong signs of manipulation."
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error during video analysis: {str(e)}")