import React, { useState } from 'react';
import './App.css';
import UploadBox from './components/UploadBox';
import ResultCard from './components/ResultCard';
import Gamification from './components/Gamification';
import Quiz from './components/Quiz';

function App() {
  const [detectionResult, setDetectionResult] = useState(null);
  const [aiGeneratedResult, setAiGeneratedResult] = useState(null);
  const [xp, setXp] = useState(0);

  const addXp = (amount) => {
    setXp(prevXp => prevXp + amount);
  };

  const handleDetection = (result) => {
    setDetectionResult(result);
    setAiGeneratedResult(null); // Clear AI generated result when we get a regular detection
    addXp(10); // 10 XP for each detection
  };

  const handleAIGeneratedDetection = (result) => {
    setAiGeneratedResult(result);
    setDetectionResult(null); // Clear regular detection result when we get an AI generated detection
    addXp(15); // Award more XP for AI generated detection
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Verilookie</h1>
        <p>Your friendly neighborhood deepfake detector.</p>
      </header>
      <main>
        <UploadBox onDetection={handleDetection} onAIGeneratedDetection={handleAIGeneratedDetection} />
        {detectionResult && <ResultCard result={detectionResult} />}
        {aiGeneratedResult && (
          <div className="result-card">
            <h2>AI Generated Detection Result</h2>
            <p><strong>Label:</strong> {aiGeneratedResult.label}</p>
            <p><strong>Confidence:</strong> {(aiGeneratedResult.confidence_score * 100).toFixed(2)}%</p>
            <p><strong>Explanation:</strong> {aiGeneratedResult.explanation}</p>
            <p><strong>AI Generated:</strong> {aiGeneratedResult.is_ai_generated ? 'Yes' : 'No'}</p>
          </div>
        )}
        <Gamification xp={xp} />
        <Quiz addXp={addXp} />
      </main>
    </div>
  );
}

export default App;