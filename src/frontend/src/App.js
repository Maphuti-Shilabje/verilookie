import React, { useState } from 'react';
import './App.css';
import UploadBox from './components/UploadBox';
import ResultCard from './components/ResultCard';
import Gamification from './components/Gamification';
import Quiz from './components/Quiz';

function App() {
  const [detectionResult, setDetectionResult] = useState(null);
  const [xp, setXp] = useState(0);

  const addXp = (amount) => {
    setXp(prevXp => prevXp + amount);
  };

  const handleDetection = (result) => {
    setDetectionResult(result);
    addXp(10); // 10 XP for each detection
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Verilookie</h1>
        <p>Your friendly neighborhood deepfake detector.</p>
      </header>
      <main>
        <UploadBox onDetection={handleDetection} />
        {detectionResult && <ResultCard result={detectionResult} />}
        <Gamification xp={xp} />
        <Quiz addXp={addXp} />
      </main>
    </div>
  );
}

export default App;