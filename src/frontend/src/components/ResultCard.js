
import React from 'react';

function ResultCard({ result }) {
  if (!result) {
    return null;
  }

  const { confidence_score, label, explanation } = result;

  return (
    <div className="result-card">
      <h2>Detection Result</h2>
      <p><strong>Label:</strong> {label}</p>
      <p><strong>Confidence:</strong> {(confidence_score * 100).toFixed(2)}%</p>
      <p><strong>Explanation:</strong> {explanation}</p>
    </div>
  );
}

export default ResultCard;
