import React, { useState } from 'react';
import './App.css';
import UploadBox from './components/UploadBox';
import ResultCard from './components/ResultCard';
import Gamification from './components/Gamification';
import Quiz from './components/Quiz';

function App() {
  const [detectionResult, setDetectionResult] = useState(null);
  const [xp, setXp] = useState(0);

  const handleDetection = (result) => {
    setDetectionResult(result);
    setXp(prevXp => prevXp + 10);
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
        <Quiz />
      </main>
    </div>
  );
}

export default App;