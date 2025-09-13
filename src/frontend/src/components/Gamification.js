
import React from 'react';

const badgeTiers = [
  { xp: 300, name: 'AI Sentinel' },
  { xp: 150, name: 'Veteran Analyst' },
  { xp: 50, name: 'Deepfake Detective' },
  { xp: 10, name: 'Rookie Detector' },
];

function Gamification({ xp }) {
  const level = Math.floor(xp / 50);

  const unlockedBadges = badgeTiers
    .filter(tier => xp >= tier.xp)
    .map(tier => tier.name);

  return (
    <div className="gamification">
      <h2>Your Progress</h2>
      <p><strong>XP:</strong> {xp}</p>
      <p><strong>Level:</strong> {level}</p>
      <p><strong>Badges:</strong> {unlockedBadges.length > 0 ? unlockedBadges.join(', ') : 'None'}</p>
    </div>
  );
}

export default Gamification;
