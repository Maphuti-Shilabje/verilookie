# Usage Guide

## ▶️ Running the Application
```bash
# Start the backend server
cd src/backend
uvicorn main:app --reload --port 8000
```

```bash
# Start the frontend (in a separate terminal)
cd src/frontend
npm start
```

## 🖥️ How to Use

### Deepfake and AI Image Detection
1. Open the application in your browser
2. Upload an image or video file using the upload box
3. Wait for the analysis to complete
4. View the detailed results about authenticity, confidence scores, and explanations

### Gamification System
1. Earn XP for each detection you perform
2. Take quizzes to earn additional XP
3. Track your progress through different levels in your profile
4. Unlock achievements as you use the application

### Personalized Quiz Feature
1. Navigate to the quiz section
2. Take AI-generated quizzes tailored to your knowledge level
3. Difficulty adapts based on your previous performance
4. Earn XP for quiz completion with bonuses for high scores
5. Retake quizzes for additional practice

See [Quiz Feature Documentation](quiz_feature.md) for technical details.

## 🎥 Demo
Check out the Demos: 
- [Demo Video](../demo/demo.mp4)
- [Demo Presentation](../demo/demo.pptx)

## 📌 Notes
- Make sure you have set up your GEMINI_API_KEY in the .env file for the quiz feature to work
- The application requires an internet connection to access the Gemini AI API
