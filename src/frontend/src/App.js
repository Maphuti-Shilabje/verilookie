import React, { useState, useEffect } from 'react';
import './App.css';
import Navbar from './components/Navbar';
import UploadBox from './components/UploadBox';
import ResultCard from './components/ResultCard';
import Gamification from './components/Gamification';
import QuizPage from './components/QuizPage';
import GamificationService from './services/GamificationService';

function App() {
  const [detectionResult, setDetectionResult] = useState(null);
  const [aiGeneratedResult, setAiGeneratedResult] = useState(null);
  const [unifiedResult, setUnifiedResult] = useState(null);
  const [xp, setXp] = useState(0);
  const [totalDetections, setTotalDetections] = useState(0);
  const [aiDetections, setAiDetections] = useState(0);
  const [quizPerfectScores, setQuizPerfectScores] = useState(0);
  const [currentStreak, setCurrentStreak] = useState(0);
  const [achievements, setAchievements] = useState([]);
  const [lastActivityDate, setLastActivityDate] = useState(null);
  const [level, setLevel] = useState(1);
  const [showGamificationModal, setShowGamificationModal] = useState(false);
  const [currentPage, setCurrentPage] = useState('home'); // Add state for current page

  // Initialize from localStorage if available
  useEffect(() => {
    const savedData = localStorage.getItem('verilookie_gamification');
    if (savedData) {
      const data = JSON.parse(savedData);
      setXp(data.xp || 0);
      setTotalDetections(data.totalDetections || 0);
      setAiDetections(data.aiDetections || 0);
      setQuizPerfectScores(data.quizPerfectScores || 0);
      setCurrentStreak(data.currentStreak || 0);
      setAchievements(data.achievements || []);
      setLastActivityDate(data.lastActivityDate || null);
      
      // Calculate initial level
      const currentLevel = GamificationService.calculateLevel(data.xp || 0);
      setLevel(currentLevel.level);
    }
  }, []);

  // Save to localStorage whenever state changes
  useEffect(() => {
    const gamificationData = {
      xp,
      totalDetections,
      aiDetections,
      quizPerfectScores,
      currentStreak,
      achievements,
      lastActivityDate
    };
    localStorage.setItem('verilookie_gamification', JSON.stringify(gamificationData));
  }, [xp, totalDetections, aiDetections, quizPerfectScores, currentStreak, achievements, lastActivityDate]);

  // Check for daily login bonus
  useEffect(() => {
    const today = new Date().toDateString();
    if (lastActivityDate !== today) {
      // Update streak
      const yesterday = new Date();
      yesterday.setDate(yesterday.getDate() - 1);
      
      if (lastActivityDate === yesterday.toDateString()) {
        // Continue streak
        setCurrentStreak(prev => prev + 1);
      } else {
        // Reset streak
        setCurrentStreak(1);
      }
      
      // Add daily login XP
      addXp(5);
      setLastActivityDate(today);
    }
  }, [lastActivityDate]);

  const addXp = (amount) => {
    // Add streak bonus
    const streakBonus = GamificationService.calculateStreakBonus(currentStreak);
    const totalXP = amount + streakBonus;
    
    setXp(prevXp => prevXp + totalXP);
  };

  const checkAndAddAchievements = (action, actionData = {}) => {
    const userData = {
      totalDetections,
      aiDetections,
      quizPerfectScores,
      currentStreak
    };
    
    const newAchievements = GamificationService.checkNewAchievements(userData, action, actionData);
    
    if (newAchievements.length > 0) {
      setAchievements(prev => {
        const updated = [...prev];
        newAchievements.forEach(achievement => {
          if (!updated.find(a => a.id === achievement)) {
            const badge = GamificationService.achievementBadges.find(b => b.id === achievement);
            if (badge) {
              updated.push(badge);
            }
          }
        });
        return updated;
      });
    }
  };

  const handleDetection = (result) => {
    setDetectionResult(result);
    setAiGeneratedResult(null); // Clear AI generated result when we get a regular detection
    setUnifiedResult(null); // Clear unified result when we get a regular detection
    
    // Add XP for detection
    addXp(10);
    
    // Update detection count
    setTotalDetections(prev => prev + 1);
    
    // Check for achievements
    checkAndAddAchievements('detection');
  };

  const handleAIGeneratedDetection = (result) => {
    setAiGeneratedResult(result);
    setDetectionResult(null); // Clear regular detection result when we get an AI generated detection
    setUnifiedResult(null); // Clear unified result when we get an AI generated detection
    
    // Add XP for AI detection
    addXp(15);
    
    // Update detection counts
    setTotalDetections(prev => prev + 1);
    setAiDetections(prev => prev + 1);
    
    // Check for achievements
    checkAndAddAchievements('ai_detection');
  };

  const handleUnifiedAnalysis = (result) => {
    setUnifiedResult(result);
    setDetectionResult(null); // Clear regular detection result
    setAiGeneratedResult(null); // Clear AI generated result
    
    // Add XP based on result type
    if (result.type === 'Deepfake') {
      addXp(20); // More XP for deepfake detection
    } else if (result.type === 'AI-generated') {
      addXp(15); // AI detection XP
    } else {
      addXp(10); // Authentic image XP
    }
    
    // Update detection counts
    setTotalDetections(prev => prev + 1);
    if (result.type === 'AI-generated') {
      setAiDetections(prev => prev + 1);
    }
    
    // Check for achievements
    if (result.type === 'Deepfake') {
      checkAndAddAchievements('deepfake_detection');
    } else if (result.type === 'AI-generated') {
      checkAndAddAchievements('ai_detection');
    } else {
      checkAndAddAchievements('detection');
    }
  };

  const handleQuizComplete = (score, total) => {
    // The XP is now added directly in the Quiz component when submitting
    setQuizPerfectScores(prev => prev + (score === total ? 1 : 0));
    checkAndAddAchievements('quiz', { score, total });
  };

  const handleLevelUp = (newLevel) => {
    // Could add special effects or notifications here
    console.log(`Level up! You are now level ${newLevel.level} (${newLevel.name})`);
    setLevel(newLevel.level);
  };

  const handleGamificationClick = () => {
    setShowGamificationModal(true);
  };

  const closeGamificationModal = () => {
    setShowGamificationModal(false);
  };

  const handlePageChange = (page) => {
    setCurrentPage(page);
  };

  // Calculate current level
  const levelData = GamificationService.calculateLevel(xp);
  
  // Get unlocked badges
  const badges = GamificationService.getUnlockedBadges(xp, achievements.map(a => a.id));

  // Render content based on current page
  const renderContent = () => {
    if (currentPage === 'quiz') {
      return (
        <QuizPage 
          addXp={addXp} 
          onComplete={handleQuizComplete}
        />
      );
    }

    // Default to home page content
    return (
      <>
        <UploadBox 
          onDetection={handleDetection} 
          onAIGeneratedDetection={handleAIGeneratedDetection} 
          onUnifiedAnalysis={handleUnifiedAnalysis}
        />
        {detectionResult && (
          <div className={`result-card ${detectionResult.label === 'Fake' ? 'fake-result' : 'real-result'}`}>
            <h2 className="result-title">🔍 Deepfake Detection Result</h2>
            <div className="result-summary">
              <div className="result-label">
                <span className="label-text">Result:</span>
                <span className={`label-value ${detectionResult.label === 'Fake' ? 'fake' : 'real'}`}>
                  {detectionResult.label}
                </span>
              </div>
              <div className="result-confidence">
                <span className="confidence-text">Confidence:</span>
                <span className="confidence-value">{(detectionResult.confidence * 100).toFixed(2)}%</span>
              </div>
            </div>
            <div className="result-explanation">
              <h3>📝 Explanation</h3>
              <p>{detectionResult.explanation}</p>
            </div>
            <div className="result-rewards">
              <h3>🏆 Rewards</h3>
              <p>+10 XP for completing this analysis</p>
              <p>Current Level: {level}</p>
              <p>Total XP: {xp}</p>
            </div>
          </div>
        )}
        {aiGeneratedResult && (
          <div className={`result-card ${aiGeneratedResult.label === 'AI Generated' ? 'fake-result' : 'real-result'}`}>
            <h2 className="result-title">🤖 AI Generation Detection Result</h2>
            <div className="result-summary">
              <div className="result-label">
                <span className="label-text">Result:</span>
                <span className={`label-value ${aiGeneratedResult.label === 'AI Generated' ? 'fake' : 'real'}`}>
                  {aiGeneratedResult.label}
                </span>
              </div>
              <div className="result-confidence">
                <span className="confidence-text">Confidence:</span>
                <span className="confidence-value">{(aiGeneratedResult.confidence_score * 100).toFixed(2)}%</span>
              </div>
            </div>
            <div className="result-explanation">
              <h3>📝 Explanation</h3>
              <p>{aiGeneratedResult.explanation}</p>
            </div>
            <div className="ai-specific">
              <h3>🤖 AI Detection</h3>
              <p>This content has been analyzed for AI-generated characteristics.</p>
              {aiGeneratedResult.label === 'Uncertain' ? (
                <p><strong>AI Generated:</strong> Uncertain (50% AI-generated vs 50% not AI-generated)</p>
              ) : (
                <p><strong>AI Generated:</strong> {aiGeneratedResult.label === 'AI Generated' ? 'Yes' : 'No'}</p>
              )}
            </div>
            <div className="result-rewards">
              <h3>🏆 Rewards</h3>
              <p>+15 XP for completing this AI analysis</p>
              <p>Current Level: {level}</p>
              <p>Total XP: {xp}</p>
            </div>
          </div>
        )}
        {unifiedResult && (
          <div className={`result-card ${unifiedResult.type === 'Deepfake' || unifiedResult.type === 'AI-generated' ? 'fake-result' : 'real-result'}`}>
            <h2 className="result-title">🔍 Unified Analysis Result</h2>
            <div className="result-summary">
              <div className="result-label">
                <span className="label-text">Result:</span>
                <span className={`label-value ${unifiedResult.type === 'Deepfake' || unifiedResult.type === 'AI-generated' ? 'fake' : 'real'}`}>
                  {unifiedResult.type}
                </span>
              </div>
              <div className="result-confidence">
                <span className="confidence-text">Confidence:</span>
                <span className="confidence-value">{unifiedResult.confidence}%</span>
              </div>
            </div>
            <div className="result-explanation">
              <h3>📝 Explanation</h3>
              <p>{unifiedResult.explanation}</p>
            </div>
            <div className="ai-specific">
              <h3>🔍 Detection Pipeline</h3>
              <p>This result was generated using our unified detection pipeline that combines deepfake and AI generation detection.</p>
            </div>
            <div className="result-rewards">
              <h3>🏆 Rewards</h3>
              <p>{unifiedResult.type === 'Deepfake' ? '+20 XP for completing this deepfake analysis' : unifiedResult.type === 'AI-generated' ? '+15 XP for completing this AI analysis' : '+10 XP for completing this analysis'}</p>
              <p>Current Level: {level}</p>
              <p>Total XP: {xp}</p>
            </div>
          </div>
        )}
      </>
    );
  };

  return (
    <div>
      <Navbar 
        level={level} 
        xp={xp} 
        onGamificationClick={handleGamificationClick} 
        onPageChange={handlePageChange}
      />
      <div className="App">
        <header className="App-header">
          <h1>Verilookie</h1>
          <p>Your friendly neighborhood deepfake detector.</p>
        </header>
        <main>
          {renderContent()}
          <Gamification 
            xp={xp}
            level={level}
            badges={badges}
            streak={currentStreak}
            achievements={achievements}
            onLevelUp={handleLevelUp}
            show={showGamificationModal}
            onClose={closeGamificationModal}
          />
        </main>
      </div>
    </div>
  );
}

export default App;