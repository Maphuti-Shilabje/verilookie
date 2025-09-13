
import React from 'react';

function Gamification({ xp }) {
  const level = Math.floor(xp / 50);
  const badges = xp >= 50 ? ['Deepfake Detective'] : [];

  return (
    <div className="gamification">
      <h2>Your Progress</h2>
      <p><strong>XP:</strong> {xp}</p>
      <p><strong>Level:</strong> {level}</p>
      <p><strong>Badges:</strong> {badges.join(', ') || 'None'}</p>
    </div>
  );
}

export default Gamification;
