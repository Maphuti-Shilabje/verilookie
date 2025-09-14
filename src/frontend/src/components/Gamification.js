import React, { useState, useEffect } from 'react';
import GamificationService from '../services/GamificationService';

function Gamification({ 
  xp, 
  level, 
  badges, 
  streak, 
  achievements,
  onLevelUp 
}) {
  const [showAchievement, setShowAchievement] = useState(null);
  const [newBadge, setNewBadge] = useState(null);
  const [xpGain, setXpGain] = useState(0);

  // Check for level up
  useEffect(() => {
    const currentLevel = GamificationService.calculateLevel(xp);
    if (currentLevel.level > level) {
      onLevelUp && onLevelUp(currentLevel);
    }
  }, [xp, level, onLevelUp]);

  // XP progress calculation
  const currentLevelData = GamificationService.calculateLevel(xp);
  const xpInCurrentLevel = xp - currentLevelData.minXP;
  const xpForNextLevel = currentLevelData.maxXP === Infinity 
    ? GamificationService.xpToNextLevel(xp)
    : currentLevelData.maxXP - currentLevelData.minXP;
  const progressPercentage = xpForNextLevel > 0 
    ? Math.min(100, (xpInCurrentLevel / xpForNextLevel) * 100)
    : 100;

  // Streak bonus calculation
  const streakBonus = GamificationService.calculateStreakBonus(streak);

  // Handle XP gain animation
  const handleXpGain = (amount) => {
    setXpGain(amount);
    setTimeout(() => setXpGain(0), 2000);
  };

  return (
    <div className="gamification">
      <h2 className="gamification-title">🏆 Your Progress</h2>
      
      {/* Level Card */}
      <div className="level-card">
        <div className="level-header">
          <div className="level-badge">Level {currentLevelData.level}</div>
          <div className="level-title">{currentLevelData.name}</div>
        </div>
        
        <div className="xp-info">
          <div className="xp-total">{xp} <span className="xp-label">XP</span></div>
          <div className="xp-to-next">
            {currentLevelData.maxXP !== Infinity 
              ? `${xpInCurrentLevel}/${xpForNextLevel} XP to next level`
              : '👑 Max level reached!'}
          </div>
        </div>
        
        {/* Progress bar with animation */}
        <div className="progress-container">
          <div 
            className={`progress-bar ${xpGain > 0 ? 'gaining' : ''}`}
            style={{ width: `${progressPercentage}%` }}
          >
            {xpGain > 0 && (
              <div className="xp-gain-animation">+{xpGain} XP</div>
            )}
          </div>
        </div>
      </div>
      
      {/* Streak Section */}
      {streak > 0 && (
        <div className="streak-card">
          <div className="streak-content">
            <div className="streak-icon">🔥</div>
            <div className="streak-info">
              <div className="streak-days">{streak} day{streak !== 1 ? 's' : ''} streak!</div>
              {streakBonus > 0 && (
                <div className="streak-bonus">+{streakBonus} XP bonus</div>
              )}
            </div>
          </div>
        </div>
      )}
      
      {/* Badges Section */}
      <div className="badges-section">
        <div className="section-header">
          <h3>🏅 Badges ({badges.length})</h3>
          <div className="section-subtitle">Earned through your achievements</div>
        </div>
        
        {badges.length > 0 ? (
          <div className="badges-grid">
            {badges.map((badge, index) => (
              <div key={index} className="badge-card" title={badge.description}>
                <div className="badge-icon">
                  {badge.type === 'xp' ? '⭐' : '🎯'}
                </div>
                <div className="badge-content">
                  <div className="badge-name">{badge.name}</div>
                  <div className="badge-detail">
                    {badge.type === 'xp' 
                      ? `${badge.xp} XP` 
                      : badge.description}
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="no-badges">
            <div className="no-badges-icon">🏅</div>
            <div className="no-badges-text">No badges yet. Keep using Verilookie to earn badges!</div>
          </div>
        )}
      </div>
      
      {/* Achievements Section */}
      {achievements && achievements.length > 0 && (
        <div className="achievements-section">
          <div className="section-header">
            <h3>🌟 Recent Achievements</h3>
            <div className="section-subtitle">Your latest accomplishments</div>
          </div>
          
          <div className="achievements-list">
            {achievements.slice(-3).reverse().map((achievement, index) => (
              <div key={index} className="achievement-item">
                <div className="achievement-icon">🏆</div>
                <div className="achievement-content">
                  <div className="achievement-name">{achievement.name}</div>
                  <div className="achievement-desc">{achievement.description}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default Gamification;