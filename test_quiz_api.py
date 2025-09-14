import requests
import json

# Test the quiz generation endpoint
def test_generate_quiz():
    url = "http://localhost:8000/quizzes/generate"
    payload = {
        "user_id": "test_user_123",
        "topic": "AI detection and deepfakes"
    }
    headers = {
        "Content-Type": "application/json"
    }
    
    response = requests.post(url, data=json.dumps(payload), headers=headers)
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    
    return response.json()

# Test the quiz submission endpoint
def test_submit_quiz(quiz_id):
    url = f"http://localhost:8000/quizzes/{quiz_id}/submit"
    payload = {
        "user_id": "test_user_123",
        "quiz_id": quiz_id,
        "answers": {
            "q_1": "Option A",
            "q_2": "Option B"
        }
    }
    headers = {
        "Content-Type": "application/json"
    }
    
    response = requests.post(url, data=json.dumps(payload), headers=headers)
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")

if __name__ == "__main__":
    print("Testing quiz generation...")
    quiz_data = test_generate_quiz()
    
    if "id" in quiz_data:
        print("\nTesting quiz submission...")
        test_submit_quiz(quiz_data["id"])