
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

function Quiz() {
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [score, setScore] = useState(0);
  const [showScore, setShowScore] = useState(false);

  const handleAnswer = (answer) => {
    if (answer === questions[currentQuestion].answer) {
      setScore(score + 1);
    }

    const nextQuestion = currentQuestion + 1;
    if (nextQuestion < questions.length) {
      setCurrentQuestion(nextQuestion);
    } else {
      setShowScore(true);
    }
  };

  return (
    <div className="quiz">
      <h2>Scam Awareness Quiz</h2>
      {showScore ? (
        <div>You scored {score} out of {questions.length}</div>
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
