// Gamification Service
// This service manages all gamification logic including XP, levels, badges, and streaks

class GamificationService {
  // Badge definitions
  static badgeTiers = [
    { xp: 1000, name: 'Master Guardian', description: 'Reach 1000 XP' },
    { xp: 500, name: 'AI Sentinel', description: 'Reach 500 XP' },
    { xp: 300, name: 'Veteran Analyst', description: 'Reach 300 XP' },
    { xp: 150, name: 'Deepfake Detective', description: 'Reach 150 XP' },
    { xp: 50, name: 'Rookie Detector', description: 'Reach 50 XP' },
    { xp: 0, name: 'Newbie', description: 'Start your journey' }
  ];

  // Achievement badges
  static achievementBadges = [
    { id: 'first_detection', name: 'First Detection', description: 'Complete your first detection' },
    { id: 'perfect_score', name: 'Perfect Score', description: 'Get 100% on a quiz' },
    { id: 'streak_master', name: 'Streak Master', description: 'Maintain a 7-day streak' },
    { id: 'quiz_master', name: 'Quiz Master', description: 'Get 100% on 5 quizzes' },
    { id: 'detection_specialist', name: 'Detection Specialist', description: 'Complete 50 detections' },
    { id: 'ai_expert', name: 'AI Expert', description: 'Complete 25 AI detections' }
  ];

  // Level definitions
  static levels = [
    { level: 1, minXP: 0, maxXP: 99, name: 'Novice' },
    { level: 2, minXP: 100, maxXP: 249, name: 'Apprentice' },
    { level: 3, minXP: 250, maxXP: 499, name: 'Expert' },
    { level: 4, minXP: 500, maxXP: 999, name: 'Master' },
    { level: 5, minXP: 1000, maxXP: Infinity, name: 'Grandmaster' }
  ];

  // Calculate level based on XP
  static calculateLevel(xp) {
    for (let i = this.levels.length - 1; i >= 0; i--) {
      if (xp >= this.levels[i].minXP) {
        return this.levels[i];
      }
    }
    return this.levels[0]; // Default to first level
  }

  // Get unlocked badges based on XP
  static getUnlockedBadges(xp, achievements = []) {
    const xpBadges = this.badgeTiers
      .filter(tier => xp >= tier.xp)
      .map(tier => ({ ...tier, type: 'xp' }));

    const achievementBadges = this.achievementBadges
      .filter(badge => achievements.includes(badge.id))
      .map(badge => ({ ...badge, type: 'achievement' }));

    return [...xpBadges, ...achievementBadges];
  }

  // Calculate XP needed for next level
  static xpToNextLevel(xp) {
    const currentLevel = this.calculateLevel(xp);
    if (currentLevel.level < this.levels.length) {
      return this.levels[currentLevel.level].minXP - xp;
    }
    return 0; // Max level reached
  }

  // Calculate streak bonus
  static calculateStreakBonus(streak) {
    if (streak >= 7) return 10;
    if (streak >= 5) return 5;
    if (streak >= 3) return 2;
    return 0;
  }

  // Check for new achievements
  static checkNewAchievements(userData, action, actionData) {
    const newAchievements = [];
    
    // First detection achievement
    if (action === 'detection' && userData.totalDetections === 0) {
      newAchievements.push('first_detection');
    }
    
    // Perfect score achievement
    if (action === 'quiz' && actionData.score === actionData.total && actionData.total > 0) {
      newAchievements.push('perfect_score');
    }
    
    // Streak master achievement
    if (action === 'daily_login' && userData.currentStreak >= 7) {
      newAchievements.push('streak_master');
    }
    
    // Quiz master achievement
    if (action === 'quiz' && userData.quizPerfectScores >= 5) {
      newAchievements.push('quiz_master');
    }
    
    // Detection specialist achievement
    if ((action === 'detection' || action === 'ai_detection') && userData.totalDetections >= 50) {
      newAchievements.push('detection_specialist');
    }
    
    // AI expert achievement
    if (action === 'ai_detection' && userData.aiDetections >= 25) {
      newAchievements.push('ai_expert');
    }
    
    return newAchievements;
  }
}

export default GamificationService;