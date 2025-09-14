import React, { useState, useEffect } from 'react';
import './Gamification.css';
import GamificationService from '../services/GamificationService';

const Gamification = ({ xp, level, badges, streak, achievements, onLevelUp, show, onClose }) => {
  const [showLevelUp, setShowLevelUp] = useState(false);
  const [levelUpData, setLevelUpData] = useState(null);
  const [prevLevel, setPrevLevel] = useState(level);

  // Calculate current level data
  const levelData = GamificationService.calculateLevel(xp);
  
  // Check for level up
  useEffect(() => {
    if (level > prevLevel) {
      setLevelUpData(levelData);
      setShowLevelUp(true);
      onLevelUp(levelData);
      
      // Auto-hide level up notification after 3 seconds
      const timer = setTimeout(() => {
        setShowLevelUp(false);
      }, 3000);
      
      return () => clearTimeout(timer);
    }
    setPrevLevel(level);
  }, [level, prevLevel, levelData, onLevelUp]);

  // Calculate XP to next level
  const xpToNext = levelData.xpForNext - xp;
  
  // Calculate progress percentage
  const progressPercent = levelData.currentLevelXp > 0 
    ? Math.min(100, Math.max(0, (xp - levelData.currentLevelXp) / (levelData.xpForNext - levelData.currentLevelXp) * 100))
    : 0;

  // Calculate streak bonus
  const streakBonus = GamificationService.calculateStreakBonus(streak);

  // Get unlocked badges
  const unlockedBadges = GamificationService.getUnlockedBadges(xp, achievements.map(a => a.id));

  return (
    <div className={`gamification-modal ${show ? 'open' : ''}`}>
      <div className="gamification-overlay" onClick={onClose}></div>
      <div className="gamification-content">
        <button className="close-button" onClick={onClose}>×</button>
        <h2 className="gamification-title">Your Progress</h2>
        
        {/* Level Up Notification */}
        {showLevelUp && levelUpData && (
          <div className="level-up-notification">
            <h3>Congratulations! 🎉</h3>
            <p>You've reached Level {levelUpData.level} - {levelUpData.name}!</p>
          </div>
        )}
        
        {/* Level Card */}
        <div className="level-card">
          <div className="level-header">
            <div className="level-badge">Level {levelData.level}</div>
            <h3 className="level-title">{levelData.name}</h3>
          </div>
          <div className="xp-info">
            <div className="xp-total">
              <div className="xp-value">{xp}</div>
              <div className="xp-label">Total XP</div>
            </div>
            <div className="xp-to-next">
              {xpToNext > 0 ? `${xpToNext} XP to next level` : 'Max level reached!'}
            </div>
          </div>
          <div className="progress-container">
            <div 
              className={`progress-bar ${xp > levelData.currentLevelXp ? 'gaining' : ''}`}
              style={{ width: `${progressPercent}%` }}
            ></div>
          </div>
        </div>
        
        {/* Streak Card */}
        <div className="streak-card">
          <div className="streak-content">
            <div className="streak-icon">🔥</div>
            <div className="streak-info">
              <div className="streak-days">{streak} day{streak !== 1 ? 's' : ''} streak!</div>
              <div className="streak-bonus">Daily bonus: +{streakBonus} XP</div>
            </div>
          </div>
        </div>
        
        {/* Badges Section */}
        <div className="section">
          <div className="section-header">
            <h3>Badges</h3>
            <div className="section-subtitle">Earned achievements</div>
          </div>
          {unlockedBadges.length > 0 ? (
            <div className="badges-grid">
              {unlockedBadges.map((badge, index) => (
                <div key={index} className="badge-card">
                  <div className="badge-icon">{badge.icon}</div>
                  <div className="badge-content">
                    <div className="badge-name">{badge.name}</div>
                    <div className="badge-detail">{badge.description}</div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="no-badges">
              <div className="no-badges-icon">🏆</div>
              <div className="no-badges-text">No badges earned yet. Complete activities to earn your first badge!</div>
            </div>
          )}
        </div>
        
        {/* Achievements Section */}
        <div className="section">
          <div className="section-header">
            <h3>Achievements</h3>
            <div className="section-subtitle">Your accomplishments</div>
          </div>
          {achievements.length > 0 ? (
            <div className="achievements-list">
              {achievements.map((achievement, index) => (
                <div key={index} className="achievement-item">
                  <div className="achievement-icon">{achievement.icon}</div>
                  <div className="achievement-content">
                    <div className="achievement-name">{achievement.name}</div>
                    <div className="achievement-desc">{achievement.description}</div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="no-badges">
              <div className="no-badges-icon">⭐</div>
              <div className="no-badges-text">No achievements yet. Keep using Verilookie to unlock achievements!</div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Gamification;