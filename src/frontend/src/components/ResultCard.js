
import React from 'react';

function ResultCard({ result }) {
  if (!result) {
    return null;
  }

  const { confidence_score, label, explanation } = result;
  
  // Determine the result type and styling
  const isFake = label === 'Fake' || label === 'AI Generated';
  const resultType = isFake ? 'fake' : 'real';
  
  // Calculate confidence level
  const confidencePercentage = (confidence_score * 100).toFixed(2);
  const confidenceLevel = confidencePercentage >= 80 ? 'High' : 
                         confidencePercentage >= 60 ? 'Medium' : 'Low';

  return (
    <div className={`result-card ${resultType}-result`}>
      <h2 className="result-title">🔍 Detection Result</h2>
      
      <div className="result-summary">
        <div className="result-label">
          <span className="label-text">Result:</span>
          <span className={`label-value ${resultType}`}>{label}</span>
        </div>
        
        <div className="result-confidence">
          <span className="confidence-text">Confidence:</span>
          <span className="confidence-value">{confidencePercentage}%</span>
          <span className={`confidence-level ${confidenceLevel.toLowerCase()}`}>
            {confidenceLevel}
          </span>
        </div>
      </div>
      
      <div className="result-explanation">
        <h3>📝 Explanation</h3>
        <p>{explanation}</p>
      </div>
      
      <div className="confidence-bar-container">
        <div className="confidence-bar-label">Confidence Level</div>
        <div className="confidence-bar">
          <div 
            className={`confidence-fill ${resultType}`}
            style={{ width: `${confidencePercentage}%` }}
          ></div>
        </div>
        <div className="confidence-percent">{confidencePercentage}%</div>
      </div>
    </div>
  );
}

export default ResultCard;
