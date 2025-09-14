# Personalized AI-Driven Quiz Generator

## Overview

This feature implements a personalized quiz generator that uses Google's Gemini AI to create dynamic quizzes tailored to each user's knowledge level. The system adapts to user performance, provides appropriate XP rewards, and allows users to retake quizzes for practice.

## Features

1. **AI-Powered Quiz Generation**: Uses Google's Gemini AI to generate relevant quiz questions
2. **Personalized Difficulty**: Adjusts quiz difficulty based on user's previous performance
3. **XP Rewards**: Awards XP based on quiz difficulty and user performance
4. **Quiz History**: Tracks user performance across multiple quizzes
5. **Repeatable Practice**: Allows users to retake quizzes for additional practice
6. **Manual Generation**: Users can click a button to generate a new personalized quiz

## Backend Implementation

### Main Components

1. **Quiz Generator** (`src/backend/quiz_generator.py`)
   - Uses Google's `google-genai` library to interface with Gemini API
   - Generates personalized quizzes based on topic and difficulty level
   - Calculates XP rewards based on performance and quiz difficulty

2. **Quiz Router** (`src/backend/routers/quizzes.py`)
   - Handles API endpoints for quiz generation, submission, and history
   - Stores quiz data and user performance in JSON files (would use a database in production)

3. **API Endpoints**
   - `POST /quizzes/generate` - Generate a new personalized quiz
   - `GET /quizzes/{quiz_id}` - Retrieve a specific quiz
   - `POST /quizzes/{quiz_id}/submit` - Submit quiz answers and calculate results
   - `GET /quizzes/user/{user_id}/history` - Get user's quiz history
   - `GET /quizzes/user/{user_id}/available` - Get quizzes available to the user

### How It Works

1. **Quiz Generation**
   - When a user clicks the "Generate Quiz" button, the system loads their previous performance data
   - Based on their accuracy history, it calculates an appropriate difficulty level
   - The Gemini AI generates questions tailored to the selected topic and difficulty
   - The quiz is saved and returned to the user

2. **Difficulty Adaptation**
   - New users start with medium-low difficulty (level 2)
   - Based on quiz performance:
     - 90%+ accuracy: Level 5 (Expert)
     - 75-89% accuracy: Level 4 (Advanced)
     - 60-74% accuracy: Level 3 (Intermediate)
     - 40-59% accuracy: Level 2 (Beginner)
     - Below 40% accuracy: Level 1 (Novice)

3. **XP Calculation**
   - Base XP = Quiz difficulty level × 10
   - Bonuses:
     - 90%+ accuracy: +20 XP
     - 70-89% accuracy: +10 XP
   - Penalties:
     - Below 30% accuracy: -10 XP

## Frontend Implementation

### Quiz Component (`src/frontend/src/components/Quiz.js`)

The Quiz component has been updated to work with the backend API and includes a manual "Generate Quiz" button:

1. **Start Screen**: Shows a button to generate a personalized quiz
2. **Loading State**: Displays while the quiz is being generated
3. **Quiz Display**: Shows the generated quiz questions one at a time
4. **Answer Tracking**: Records user answers as they progress through the quiz
5. **Submission**: Sends answers to the backend for scoring and XP calculation
6. **Results Display**: Shows scored results with correct/incorrect answer highlighting

## Setup Instructions

1. **Install Dependencies**
   ```bash
   pip install -r src/backend/requirements.txt
   ```

2. **Configure Environment**
   - Ensure your `.env` file in `src/backend` contains your `GEMINI_API_KEY`
   - Example:
     ```
     GEMINI_API_KEY=your_actual_api_key_here
     ```

3. **Run the Application**
   ```bash
   cd src/backend
   uvicorn main:app --reload --port 8000
   ```

## API Usage Examples

### Generate a Quiz
```bash
curl -X POST http://localhost:8000/quizzes/generate \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user123", "topic": "AI detection and deepfakes"}'
```

### Submit Quiz Answers
```bash
curl -X POST http://localhost:8000/quizzes/quiz_123/submit \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user123", "quiz_id": "quiz_123", "answers": {"q_1": "Option A", "q_2": "Option B"}}'
```

## Future Enhancements

1. **Database Integration**: Replace JSON file storage with a proper database
2. **Category System**: Add multiple quiz topics and categories
3. **Time Tracking**: Add time-based performance metrics
4. **Social Features**: Allow users to compare scores and compete
5. **Advanced Analytics**: Provide detailed performance insights and learning recommendations