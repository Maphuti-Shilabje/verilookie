import React, { useState, useEffect } from 'react';
import './QuizList.css';

function QuizList({ addXp, onComplete }) {
  const [quizzes, setQuizzes] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedQuiz, setSelectedQuiz] = useState(null);
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [correctAnswers, setCorrectAnswers] = useState(0);
  const [showScore, setShowScore] = useState(false);
  const [answers, setAnswers] = useState({});

  // Load quizzes from the backend
  useEffect(() => {
    fetchQuizzes();
  }, []);

  const fetchQuizzes = async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await fetch('http://localhost:8000/quizzes/user/user123/available');
      if (!response.ok) {
        throw new Error('Failed to fetch quizzes');
      }
      const quizzesData = await response.json();
      setQuizzes(Object.values(quizzesData));
      setLoading(false);
    } catch (err) {
      setError(err.message);
      setLoading(false);
    }
  };

  // Function to generate a new quiz
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
      
      // After generating a new quiz, refresh the list
      await fetchQuizzes();
    } catch (err) {
      setError(err.message);
      setLoading(false);
    }
  };

  const handleQuizSelect = (quiz) => {
    setSelectedQuiz(quiz);
    // Reset quiz state
    setCurrentQuestion(0);
    setCorrectAnswers(0);
    setShowScore(false);
    setAnswers({});
  };

  const handleBackToQuizList = () => {
    setSelectedQuiz(null);
    setCurrentQuestion(0);
    setCorrectAnswers(0);
    setShowScore(false);
    setAnswers({});
  };

  const handleAnswer = (questionId, answer) => {
    // Track user selection
    setAnswers(prev => ({
      ...prev,
      [questionId]: answer
    }));
    
    const nextQuestion = currentQuestion + 1;
    if (nextQuestion < selectedQuiz.questions.length) {
      setCurrentQuestion(nextQuestion);
    } else {
      // Quiz completed - automatically submit
      setTimeout(() => {
        submitQuiz();
      }, 500);
    }
  };

  const submitQuiz = async () => {
    if (!selectedQuiz) return;
    
    try {
      // In a real implementation, you would get the user ID from authentication
      const userId = 'user123';
      
      const response = await fetch(`http://localhost:8000/quizzes/${selectedQuiz.id}/submit`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          user_id: userId,
          quiz_id: selectedQuiz.id,
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
        onComplete(result.score, selectedQuiz.questions.length);
      }
      
      // After a short delay, go back to the quiz list
      setTimeout(() => {
        setShowScore(true);
      }, 500);
    } catch (err) {
      setError(err.message);
    }
  };

  const restartQuiz = () => {
    setSelectedQuiz(null);
    setCurrentQuestion(0);
    setCorrectAnswers(0);
    setShowScore(false);
    setAnswers({});
  };

  const getOptionClass = (option, question) => {
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

  if (loading && !selectedQuiz) return (
    <div className="quiz">
      <div className="loading-container">
        <div className="loading-bar">
          <div className="loading-progress"></div>
        </div>
        <p className="loading-text">Generating your personalized quiz...</p>
      </div>
    </div>
  );
  if (error && !selectedQuiz) return <div className="quiz">Error: {error}</div>;

  return (
    <div className="quiz">
      {!selectedQuiz && (
        <div className="quiz-list">
          <div className="quiz-header">
            <h3>Available Quizzes</h3>
            <button onClick={generateQuiz} className="generate-quiz-button" disabled={loading}>
              {loading ? 'Generating...' : 'Generate New Quiz'}
            </button>
          </div>
          {error && <div className="error-message">Error: {error}</div>}
          {loading && (
            <div className="loading-container">
              <div className="loading-bar">
                <div className="loading-progress"></div>
              </div>
              <p className="loading-text">Generating your personalized quiz...</p>
            </div>
          )}
          {!loading && quizzes.length === 0 ? (
            <p>No quizzes available at the moment.</p>
          ) : !loading ? (
            <div className="quiz-grid">
              {quizzes.map((quiz) => (
                <div key={quiz.id} className="quiz-card" onClick={() => handleQuizSelect(quiz)}>
                  <h4>{quiz.title}</h4>
                  <p>{quiz.questions.length} questions</p>
                  <p>Difficulty: {quiz.difficulty}/5</p>
                  <button className="take-quiz-button">Take Quiz</button>
                </div>
              ))}
            </div>
          ) : null}
        </div>
      )}
      
      {selectedQuiz && showScore && (
        <div className="quiz-results">
          <div className="quiz-header">
            <button onClick={handleBackToQuizList} className="back-button">
              ← Back to Quiz List
            </button>
          </div>
          <div className="score-circle">
            <div className="score-value">{correctAnswers}</div>
            <div className="score-total">/ {selectedQuiz.questions.length}</div>
          </div>
          <div className="score-percentage">
            {Math.round((correctAnswers / selectedQuiz.questions.length) * 100)}% Correct
          </div>
          {correctAnswers === selectedQuiz.questions.length && (
            <div className="perfect-score">🎉 Perfect Score! Security Expert!</div>
          )}
          <div className="score-message">
            {correctAnswers >= selectedQuiz.questions.length * 0.8 ? "Excellent work! You're well-prepared to spot scams." :
             correctAnswers >= selectedQuiz.questions.length * 0.6 ? "Good job! You're on the right track." :
             "Keep learning to improve your scam detection skills!"}
          </div>
          <button onClick={restartQuiz} className="restart-button">
            🔄 Back to Quiz List
          </button>
        </div>
      )}
      
      {selectedQuiz && !showScore && (
        <div className="quiz-content">
          <div className="quiz-header">
            <button onClick={handleBackToQuizList} className="back-button">
              ← Back to Quiz List
            </button>
          </div>
          <div className="question-section">
            <div className="question-progress">
              Question {currentQuestion + 1} of {selectedQuiz.questions.length}
            </div>
            <div className="question-text">
              {selectedQuiz.questions[currentQuestion].question}
            </div>
          </div>
          <div className="answer-section">
            {selectedQuiz.questions[currentQuestion].options.map((option, index) => (
              <button 
                key={index} 
                onClick={() => handleAnswer(selectedQuiz.questions[currentQuestion].id, option)}
                className={getOptionClass(option, selectedQuiz.questions[currentQuestion])}
              >
                {option}
              </button>
            ))}
          </div>
          <div className="quiz-progress">
            <div className="progress-bar-container">
              <div 
                className="progress-bar" 
                style={{ width: `${((currentQuestion + 1) / selectedQuiz.questions.length) * 100}%` }}
              ></div>
            </div>
            <div className="progress-text">
              {Math.round(((currentQuestion + 1) / selectedQuiz.questions.length) * 100)}% Complete
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default QuizList;