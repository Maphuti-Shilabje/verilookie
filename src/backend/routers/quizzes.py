from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import json
import os
from datetime import datetime

from quiz_generator import quiz_generator, Quiz, UserPerformance

# Create router
router = APIRouter(prefix="/quizzes", tags=["quizzes"])

# Simple in-memory storage for quizzes and user performance
# In a production environment, this would be replaced with a database
QUIZZES_FILE = "quizzes_data.json"
USER_PERFORMANCE_FILE = "user_performance_data.json"

def load_quizzes():
    """Load quizzes from file"""
    if os.path.exists(QUIZZES_FILE):
        with open(QUIZZES_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_quizzes(quizzes):
    """Save quizzes to file"""
    with open(QUIZZES_FILE, 'w') as f:
        json.dump(quizzes, f)

def load_user_performance():
    """Load user performance data from file"""
    if os.path.exists(USER_PERFORMANCE_FILE):
        with open(USER_PERFORMANCE_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_user_performance(performance):
    """Save user performance data to file"""
    with open(USER_PERFORMANCE_FILE, 'w') as f:
        json.dump(performance, f)

class QuizRequest(BaseModel):
    user_id: str
    topic: Optional[str] = "AI detection and deepfakes"
    
class QuizSubmission(BaseModel):
    user_id: str
    quiz_id: str
    answers: dict  # question_id: selected_answer

class QuizResult(BaseModel):
    score: int
    total: int
    xp_earned: int
    feedback: str

@router.post("/generate", response_model=Quiz)
async def generate_quiz(request: QuizRequest):
    """
    Generate a personalized quiz for a user based on their previous performance.
    """
    try:
        # Load user performance data
        user_performance_data = load_user_performance()
        user_performances = user_performance_data.get(request.user_id, [])
        
        # Convert to UserPerformance objects
        previous_performances = []
        for perf_data in user_performances:
            perf = UserPerformance(
                user_id=perf_data["user_id"],
                quiz_id=perf_data["quiz_id"],
                score=perf_data["score"],
                total=perf_data["total"],
                timestamp=perf_data["timestamp"],
                xp_earned=perf_data["xp_earned"],
                difficulty_level=perf_data.get("difficulty_level", 1)
            )
            previous_performances.append(perf)
        
        # Generate personalized quiz
        quiz = await quiz_generator.create_personalized_quiz(
            user_id=request.user_id,
            topic=request.topic,
            previous_performance=previous_performances
        )
        
        # Validate the quiz before saving
        if not quiz or not quiz.questions:
            raise HTTPException(status_code=500, detail="Failed to generate valid quiz: No questions generated")
        
        # Validate each question
        for i, question in enumerate(quiz.questions):
            if not question.question or not question.options or not question.correct_answer:
                raise HTTPException(status_code=500, detail=f"Failed to generate valid quiz: Question {i+1} is invalid")
            
            # Ensure correct_answer is one of the options
            if question.correct_answer not in question.options:
                raise HTTPException(status_code=500, detail=f"Failed to generate valid quiz: Question {i+1} has invalid correct_answer")
        
        # Save the quiz
        quizzes = load_quizzes()
        quizzes[quiz.id] = quiz.dict()
        save_quizzes(quizzes)
        
        return quiz
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Log the full error for debugging
        import traceback
        error_details = f"Error generating quiz: {str(e)}\nTraceback: {traceback.format_exc()}"
        print(error_details)  # For immediate debugging
        raise HTTPException(status_code=500, detail=error_details)

@router.get("/{quiz_id}", response_model=Quiz)
async def get_quiz(quiz_id: str):
    """
    Retrieve a specific quiz by ID.
    """
    quizzes = load_quizzes()
    if quiz_id not in quizzes:
        raise HTTPException(status_code=404, detail="Quiz not found")
    
    quiz_data = quizzes[quiz_id]
    return Quiz(**quiz_data)

@router.post("/{quiz_id}/submit", response_model=QuizResult)
async def submit_quiz(quiz_id: str, submission: QuizSubmission):
    """
    Submit answers for a quiz and calculate the result.
    """
    # Load the quiz
    quizzes = load_quizzes()
    if quiz_id not in quizzes:
        raise HTTPException(status_code=404, detail="Quiz not found")
    
    quiz_data = quizzes[quiz_id]
    quiz = Quiz(**quiz_data)
    
    # Calculate score
    score = 0
    for question in quiz.questions:
        if str(question.id) in submission.answers:
            if submission.answers[str(question.id)] == question.correct_answer:
                score += 1
    
    # Calculate XP reward
    xp_earned = quiz_generator.calculate_xp_reward(quiz, score, len(quiz.questions))
    
    # Save user performance
    user_performance_data = load_user_performance()
    if submission.user_id not in user_performance_data:
        user_performance_data[submission.user_id] = []
    
    performance = UserPerformance(
            user_id=submission.user_id,
            quiz_id=quiz_id,
            score=score,
            total=len(quiz.questions),
            timestamp=datetime.now().isoformat(),
            xp_earned=xp_earned,
            difficulty_level=quiz.difficulty
        )
    
    user_performance_data[submission.user_id].append(performance.dict())
    save_user_performance(user_performance_data)
    
    # Generate feedback
    accuracy = score / len(quiz.questions) if len(quiz.questions) > 0 else 0
    if accuracy >= 0.9:
        feedback = "Excellent work! You're mastering this topic."
    elif accuracy >= 0.7:
        feedback = "Good job! You're on the right track."
    elif accuracy >= 0.5:
        feedback = "Not bad, but there's room for improvement."
    else:
        feedback = "Keep studying to improve your knowledge."
    
    return QuizResult(
        score=score,
        total=len(quiz.questions),
        xp_earned=xp_earned,
        feedback=feedback
    )

@router.get("/user/{user_id}/history")
async def get_user_quiz_history(user_id: str):
    """
    Get quiz history for a specific user.
    """
    user_performance_data = load_user_performance()
    return user_performance_data.get(user_id, [])

@router.get("/user/{user_id}/available")
async def get_user_available_quizzes(user_id: str):
    """
    Get quizzes that are available for a user to take.
    """
    quizzes = load_quizzes()
    user_performance_data = load_user_performance()
    user_performances = user_performance_data.get(user_id, [])
    
    # For simplicity, we'll return all quizzes that haven't been taken by the user
    # In a real implementation, you might want to filter based on user level, etc.
    taken_quiz_ids = {perf["quiz_id"] for perf in user_performances}
    available_quizzes = {
        quiz_id: quiz_data 
        for quiz_id, quiz_data in quizzes.items() 
        if quiz_id not in taken_quiz_ids
    }
    
    return available_quizzes