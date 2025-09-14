
import React, { useState } from 'react';

function Quiz({ addXp, onComplete }) {
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [correctAnswers, setCorrectAnswers] = useState(0);
  const [showScore, setShowScore] = useState(false);
  const [quiz, setQuiz] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [answers, setAnswers] = useState({});

  // Function to generate quiz when button is clicked
  const generateQuiz = async () => {
    try {
      setLoading(true);
      setError(null);
      
      // In a real implementation, you would get the user ID from authentication
      const userId = 'user123';
      
      const response = await fetch('http://localhost:8000/quizzes/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          user_id: userId,
          topic: 'AI detection and deepfakes'
        }),
      });
      
      if (!response.ok) {
        throw new Error('Failed to generate quiz');
      }
      
      const quizData = await response.json();
      setQuiz(quizData);
      setLoading(false);
    } catch (err) {
      setError(err.message);
      setLoading(false);
    }
  };

  const handleAnswer = (questionId, answer) => {
    // Track user selection
    setAnswers(prev => ({
      ...prev,
      [questionId]: answer
    }));
    
    const nextQuestion = currentQuestion + 1;
    if (nextQuestion < quiz.questions.length) {
      setCurrentQuestion(nextQuestion);
    } else {
      // Quiz completed
      setShowScore(true);
    }
  };

  const submitQuiz = async () => {
    if (!quiz) return;
    
    try {
      // In a real implementation, you would get the user ID from authentication
      const userId = 'user123';
      
      const response = await fetch(`http://localhost:8000/quizzes/${quiz.id}/submit`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          user_id: userId,
          quiz_id: quiz.id,
          answers: answers
        }),
      });
      
      if (!response.ok) {
        throw new Error('Failed to submit quiz');
      }
      
      const result = await response.json();
      
      // Update state with actual results
      setCorrectAnswers(result.score);
      
      // Add XP
      if (addXp) {
        addXp(result.xp_earned);
      }
      
      // Call onComplete callback
      if (onComplete) {
        onComplete(result.score, quiz.questions.length);
      }
    } catch (err) {
      setError(err.message);
    }
  };

  const restartQuiz = () => {
    setCurrentQuestion(0);
    setCorrectAnswers(0);
    setShowScore(false);
    setAnswers({});
    setQuiz(null);
  };

  const getOptionClass = (option, index, question) => {
    let className = "quiz-option";
    
    if (showScore && question && question.id) {
      // Mark correct answers
      if (option === question.correct_answer) {
        className += " correct-answer";
      } 
      // Mark user's incorrect answers
      else if (answers && answers[question.id] === option) {
        className += " incorrect-answer";
      }
    }
    
    return className;
  };

  return (
    <div className="quiz">
      <h2 className="quiz-title">🧠 Scam Awareness Quiz</h2>
      
      {!quiz && !loading && !error && (
        <div className="quiz-start">
          <p>Click the button below to generate a personalized quiz based on your previous performance.</p>
          <button onClick={generateQuiz} className="generate-quiz-button">
            Generate Quiz
          </button>
        </div>
      )}
      
      {loading && <div className="quiz">Generating quiz...</div>}
      
      {error && <div className="quiz">Error: {error}</div>}
      
      {quiz && showScore && (
        <div className="quiz-results">
          <div className="score-circle">
            <div className="score-value">{correctAnswers}</div>
            <div className="score-total">/ {quiz.questions.length}</div>
          </div>
          <div className="score-percentage">
            {Math.round((correctAnswers / quiz.questions.length) * 100)}% Correct
          </div>
          {correctAnswers === quiz.questions.length && (
            <div className="perfect-score">🎉 Perfect Score! Security Expert!</div>
          )}
          <div className="score-message">
            {correctAnswers >= quiz.questions.length * 0.8 ? "Excellent work! You're well-prepared to spot scams." :
             correctAnswers >= quiz.questions.length * 0.6 ? "Good job! You're on the right track." :
             "Keep learning to improve your scam detection skills!"}
          </div>
          <button onClick={submitQuiz} className="restart-button">
            🔄 Submit & Finish
          </button>
        </div>
      )}
      
      {quiz && !showScore && (
        <div className="quiz-content">
          <div className="question-section">
            <div className="question-progress">
              Question {currentQuestion + 1} of {quiz.questions.length}
            </div>
            <div className="question-text">
              {quiz.questions[currentQuestion].question}
            </div>
          </div>
          <div className="answer-section">
            {quiz.questions[currentQuestion].options.map((option, index) => (
              <button 
                key={index} 
                onClick={() => handleAnswer(quiz.questions[currentQuestion].id, option)}
                className={getOptionClass(option, index, quiz.questions[currentQuestion])}
              >
                {option}
              </button>
            ))}
          </div>
          <div className="quiz-progress">
            <div className="progress-bar-container">
              <div 
                className="progress-bar" 
                style={{ width: `${((currentQuestion + 1) / quiz.questions.length) * 100}%` }}
              ></div>
            </div>
            <div className="progress-text">
              {Math.round(((currentQuestion + 1) / quiz.questions.length) * 100)}% Complete
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default Quiz;
