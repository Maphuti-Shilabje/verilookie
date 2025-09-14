import requests
import json

# Test the unified detection pipeline
def test_unified_pipeline():
    print("Testing Unified Detection Pipeline")
    print("=" * 40)
    
    # Test endpoint
    url = "http://localhost:8000/analyze"
    
    # You would need to provide an actual image file for testing
    # For now, let's just show the expected structure
    
    print("Unified Detection Pipeline Implementation:")
    print("1. Deepfake Analyzer runs first")
    print("2. If confident (score >= 0.7), return Deepfake result")
    print("3. If inconclusive, run AI Image Detector")
    print("4. Return unified result based on AI detector score")
    print()
    
    print("Expected Response Formats:")
    print()
    print("Deepfake Detected:")
    print(json.dumps({
        "type": "Deepfake",
        "confidence": 90.0,
        "explanation": "The image appears manipulated. Detected as a deepfake."
    }, indent=2))
    print()
    
    print("AI-generated Detected:")
    print(json.dumps({
        "type": "AI-generated",
        "confidence": 87.0,
        "explanation": "The image was generated using AI (not authentic)."
    }, indent=2))
    print()
    
    print("Authentic Image:")
    print(json.dumps({
        "type": "Authentic",
        "confidence": 88.0,
        "explanation": "The image does not appear to be AI-generated or a deepfake."
    }, indent=2))
    print()

if __name__ == "__main__":
    test_unified_pipeline()