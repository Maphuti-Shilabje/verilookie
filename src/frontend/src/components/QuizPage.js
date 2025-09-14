import React from 'react';
import QuizList from './QuizList';
import './QuizPage.css';

const QuizPage = ({ addXp, onComplete }) => {
  return (
    <div className="quiz-page">
      <div className="quiz-page-header">
        <h1 className="quiz-title">🧠 Scam Awareness Quiz</h1>
        <p>Test your knowledge and earn XP by taking our quizzes</p>
      </div>
      <QuizList 
        addXp={addXp} 
        onComplete={onComplete}
      />
    </div>
  );
};

export default QuizPage;