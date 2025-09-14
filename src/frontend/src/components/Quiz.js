
import React, { useState } from 'react';

const questions = [
  {
    question: "What is a common sign of a deepfake video?",
    options: ["Blurry background", "Unnatural blinking", "Loud audio", "Bright colors"],
    answer: "Unnatural blinking",
  },
  {
    question: "What is phishing?",
    options: ["A type of fishing", "A scam to steal personal information", "A video game", "A social media trend"],
    answer: "A scam to steal personal information",
  },
  {
    question: "Which of these is a sign of an AI-generated image?",
    options: ["Perfect symmetry", "Unnatural lighting", "Pixelation", "Watermarks"],
    answer: "Unnatural lighting",
  },
  {
    question: "What should you do if you suspect a deepfake?",
    options: ["Share it immediately", "Verify the source", "Ignore it", "Create more deepfakes"],
    answer: "Verify the source",
  },
  {
    question: "Which file type is most commonly used for deepfakes?",
    options: [".txt", ".mp4", ".pdf", ".exe"],
    answer: ".mp4",
  },
];

function Quiz({ addXp, onComplete }) {
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [correctAnswers, setCorrectAnswers] = useState(0);
  const [showScore, setShowScore] = useState(false);
  const [selectedAnswers, setSelectedAnswers] = useState([]);
  const [userSelections, setUserSelections] = useState({});

  const handleAnswer = (answer) => {
    // Track user selection
    setUserSelections(prev => ({
      ...prev,
      [currentQuestion]: answer
    }));
    
    // Check if answer is correct
    const isCorrect = answer === questions[currentQuestion].answer;
    if (isCorrect) {
      setCorrectAnswers(prev => prev + 1);
    }

    const nextQuestion = currentQuestion + 1;
    if (nextQuestion < questions.length) {
      setCurrentQuestion(nextQuestion);
    } else {
      // Quiz completed
      setShowScore(true);
      onComplete && onComplete(correctAnswers + (isCorrect ? 1 : 0), questions.length);
    }
  };

  const restartQuiz = () => {
    setCurrentQuestion(0);
    setCorrectAnswers(0);
    setShowScore(false);
    setSelectedAnswers([]);
    setUserSelections({});
  };

  const getOptionClass = (option, index) => {
    let className = "quiz-option";
    
    if (showScore && currentQuestion === index) {
      if (option === questions[currentQuestion].answer) {
        className += " correct-answer";
      } else if (userSelections[currentQuestion] === option) {
        className += " incorrect-answer";
      }
    }
    
    return className;
  };

  return (
    <div className="quiz">
      <h2 className="quiz-title">🧠 Scam Awareness Quiz</h2>
      {showScore ? (
        <div className="quiz-results">
          <div className="score-circle">
            <div className="score-value">{correctAnswers}</div>
            <div className="score-total">/ {questions.length}</div>
          </div>
          <div className="score-percentage">
            {Math.round((correctAnswers / questions.length) * 100)}% Correct
          </div>
          {correctAnswers === questions.length && (
            <div className="perfect-score">🎉 Perfect Score! Security Expert!</div>
          )}
          <div className="score-message">
            {correctAnswers >= questions.length * 0.8 ? "Excellent work! You're well-prepared to spot scams." :
             correctAnswers >= questions.length * 0.6 ? "Good job! You're on the right track." :
             "Keep learning to improve your scam detection skills!"}
          </div>
          <button onClick={restartQuiz} className="restart-button">
            🔄 Try Again
          </button>
        </div>
      ) : (
        <div className="quiz-content">
          <div className="question-section">
            <div className="question-progress">
              Question {currentQuestion + 1} of {questions.length}
            </div>
            <div className="question-text">
              {questions[currentQuestion].question}
            </div>
          </div>
          <div className="answer-section">
            {questions[currentQuestion].options.map((option, index) => (
              <button 
                key={index} 
                onClick={() => handleAnswer(option)}
                className={getOptionClass(option, currentQuestion)}
              >
                {option}
              </button>
            ))}
          </div>
          <div className="quiz-progress">
            <div className="progress-bar-container">
              <div 
                className="progress-bar" 
                style={{ width: `${((currentQuestion + 1) / questions.length) * 100}%` }}
              ></div>
            </div>
            <div className="progress-text">
              {Math.round(((currentQuestion + 1) / questions.length) * 100)}% Complete
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default Quiz;
