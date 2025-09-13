
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
];

function Quiz({ addXp }) {
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [correctAnswers, setCorrectAnswers] = useState(0);
  const [showScore, setShowScore] = useState(false);

  const handleAnswer = (answer) => {
    if (answer === questions[currentQuestion].answer) {
      addXp(5); // 5 XP for a correct answer
      setCorrectAnswers(correctAnswers + 1);
    }

    const nextQuestion = currentQuestion + 1;
    if (nextQuestion < questions.length) {
      setCurrentQuestion(nextQuestion);
    } else {
      setShowScore(true);
    }
  };

  const restartQuiz = () => {
    setCurrentQuestion(0);
    setCorrectAnswers(0);
    setShowScore(false);
  };

  return (
    <div className="quiz">
      <h2>Scam Awareness Quiz</h2>
      {showScore ? (
        <div>
          <p>You scored {correctAnswers} out of {questions.length}</p>
          <button onClick={restartQuiz}>Restart Quiz</button>
        </div>
      ) : (
        <>
          <div className="question-section">
            {questions[currentQuestion].question}
          </div>
          <div className="answer-section">
            {questions[currentQuestion].options.map((option, index) => (
              <button key={index} onClick={() => handleAnswer(option)}>
                {option}
              </button>
            ))}
          </div>
        </>
      )}
    </div>
  );
}

export default Quiz;
