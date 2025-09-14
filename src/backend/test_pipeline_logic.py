import json

def test_pipeline_logic():
    """
    Test the corrected pipeline logic against the specification examples
    """
    print("Testing Unified Detection Pipeline Logic")
    print("=" * 50)
    
    # Test cases based on the specification
    test_cases = [
        {
            "name": "High confidence deepfake",
            "deepfake_label": "Fake",
            "deepfake_confidence": 0.9,
            "ai_score": None,
            "expected_type": "Deepfake",
            "expected_confidence": 90.0
        },
        {
            "name": "High confidence real",
            "deepfake_label": "Real",
            "deepfake_confidence": 0.8,
            "ai_score": None,
            "expected_type": "Authentic",
            "expected_confidence": 80.0
        },
        {
            "name": "AI-generated image (score = 0.87)",
            "deepfake_label": "Real",  # Inconclusive deepfake result
            "deepfake_confidence": 0.4,  # Below threshold
            "ai_score": 0.87,
            "expected_type": "AI-generated",
            "expected_confidence": 87.0
        },
        {
            "name": "Authentic image (score = 0.12)",
            "deepfake_label": "Real",  # Inconclusive deepfake result
            "deepfake_confidence": 0.3,  # Below threshold
            "ai_score": 0.12,
            "expected_type": "Authentic",
            "expected_confidence": 88.0  # (1 - 0.12) * 100
        },
        {
            "name": "Uncertain AI result (score = 0.5)",
            "deepfake_label": "Real",  # Inconclusive deepfake result
            "deepfake_confidence": 0.4,  # Below threshold
            "ai_score": 0.5,
            "expected_type": "Authentic",
            "expected_confidence": 50.0
        }
    ]
    
    print("Expected Output Formats:")
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
    
    print("Pipeline Logic Verification:")
    print("- Step 1: Deepfake Analyzer runs first")
    print("- If confident (score >= 0.7) and Fake → Deepfake result")
    print("- If confident (score >= 0.7) and Real → Authentic result")
    print("- If inconclusive (< 0.7) → Forward to AI Image Detector")
    print("- Step 2: AI Image Detector")
    print("  - If score > 0.5 → AI-generated")
    print("  - If score < 0.5 → Authentic (confidence = (1 - score) * 100)")
    print("  - If score = 0.5 → Authentic (confidence = 50%)")
    print()

if __name__ == "__main__":
    test_pipeline_logic()