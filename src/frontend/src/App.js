import React, { useState, useEffect } from 'react';
import './App.css';
import UploadBox from './components/UploadBox';
import ResultCard from './components/ResultCard';
import Gamification from './components/Gamification';
import Quiz from './components/Quiz';
import Modal from './components/Modal';
import GamificationService from './services/GamificationService';

function App() {
  const [currentResult, setCurrentResult] = useState(null);
  const [showLevelUp, setShowLevelUp] = useState(false);
  const [levelUpData, setLevelUpData] = useState({ level: 1, xpGained: 0 });
  const [xp, setXp] = useState(0);
  const [totalDetections, setTotalDetections] = useState(0);
  const [aiDetections, setAiDetections] = useState(0);
  const [quizPerfectScores, setQuizPerfectScores] = useState(0);
  const [currentStreak, setCurrentStreak] = useState(0);
  const [achievements, setAchievements] = useState([]);
  const [lastActivityDate, setLastActivityDate] = useState(null);
  const [level, setLevel] = useState(1);

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
    
    const newXp = xp + totalXP;
    setXp(newXp);
    
    // Check for level up
    const oldLevel = level;
    const newLevel = GamificationService.calculateLevel(newXp).level;
    
    if (newLevel > oldLevel) {
      setLevel(newLevel);
      setLevelUpData({ level: newLevel, xpGained: totalXP });
      // Show level up notification after a short delay to ensure result modal is closed
      setTimeout(() => {
        setShowLevelUp(true);
      }, 500);
    }
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
    setCurrentResult(result);
    setShowResultModal(true);
    
    // Add XP for detection
    addXp(10);
    
    // Update detection count
    setTotalDetections(prev => prev + 1);
    
    // Check for achievements
    checkAndAddAchievements('detection');
  };

  const handleAIGeneratedDetection = (result) => {
    setCurrentResult(result);
    setShowResultModal(true);
    
    // Add XP for AI detection
    addXp(15);
    
    // Update detection counts
    setTotalDetections(prev => prev + 1);
    setAiDetections(prev => prev + 1);
    
    // Check for achievements
    checkAndAddAchievements('ai_detection');
  };

  const handleUnifiedAnalysis = (result) => {
    setCurrentResult(result);
    setShowResultModal(true);
    
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
    // Check for perfect score
    if (score === total && total > 0) {
      addXp(20); // Bonus for perfect score
      setQuizPerfectScores(prev => prev + 1);
      checkAndAddAchievements('quiz', { score, total });
    } else if (score > 0) {
      // Regular XP for correct answers (5 XP per correct answer)
      addXp(score * 5);
    }
  };

  const handleLevelUp = (newLevel) => {
    // Could add special effects or notifications here
    console.log(`Level up! You are now level ${newLevel.level} (${newLevel.name})`);
    setLevel(newLevel.level);
  };

  // Calculate current level
  const levelData = GamificationService.calculateLevel(xp);
  
  // Get unlocked badges
  const badges = GamificationService.getUnlockedBadges(xp, achievements.map(a => a.id));

  return (
    <div className="App">
      {/* Professional, clean header */}
      <header className="app-header">
        <div className="header-container">
          <div className="header-left">
            <h1 className="app-logo">Verilookie</h1>
          </div>
          <div className="header-right">
            <nav className="main-nav">
              <a href="#how-it-works" className="nav-link">How it Works</a>
              {/* Clickable level/XP indicator in navbar */}
              <div className="user-level" onClick={() => console.log('Show level details')}>
                <span className="level-text">Level {level}</span>
                <span className="xp-text">{xp} XP</span>
              </div>
            </nav>
          </div>
        </div>
      </header>
      
      <main>
        <UploadBox 
          onDetection={handleDetection} 
          onAIGeneratedDetection={handleAIGeneratedDetection} 
          onUnifiedAnalysis={handleUnifiedAnalysis}
        />
        <Quiz 
          addXp={addXp} 
          onComplete={handleQuizComplete}
        />
        <Gamification 
          xp={xp}
          level={level}
          badges={badges}
          streak={currentStreak}
          achievements={achievements}
          onLevelUp={handleLevelUp}
        />
      </main>
      
      {/* Result Modal */}
      <Modal 
        isOpen={showResultModal} 
        onClose={() => setShowResultModal(false)}
        title="Analysis Result"
      >
        <ResultCard result={currentResult} />
      </Modal>
      
      {/* Level Up Modal */}
      <Modal 
        isOpen={showLevelUp} 
        onClose={() => setShowLevelUp(false)}
        title="Level Up!"
      >
        <div className="level-up-content">
          <div className="level-up-icon">🎉</div>
          <h3 className="level-up-title">Congratulations!</h3>
          <p className="level-up-text">You've reached level {levelUpData.level}</p>
          <p className="xp-gained-text">+{levelUpData.xpGained} XP gained</p>
          <button 
            className="btn btn-primary"
            onClick={() => setShowLevelUp(false)}
          >
            Continue
          </button>
        </div>
      </Modal>
    </div>
  );
}

export default App;