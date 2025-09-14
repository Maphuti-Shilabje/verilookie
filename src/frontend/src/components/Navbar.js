import React from 'react';
import './Navbar.css';

const Navbar = ({ level, xp, onGamificationClick, onPageChange }) => {
  // Calculate level name based on XP
  const getLevelName = (xp) => {
    if (xp < 100) return 'Novice';
    if (xp < 500) return 'Beginner';
    if (xp < 1000) return 'Intermediate';
    if (xp < 2000) return 'Advanced';
    return 'Expert';
  };

  const levelName = getLevelName(xp);

  return (
    <nav className="navbar">
      <div className="navbar-container">
        <div className="navbar-logo">
          <h2>Verilookie</h2>
        </div>
        <ul className="navbar-menu">
          <li className="navbar-item">
            <a href="#home" className="navbar-link" onClick={(e) => {
              e.preventDefault();
              onPageChange('home');
            }}>Home</a>
          </li>
          <li className="navbar-item">
            <a href="#detect" className="navbar-link" onClick={(e) => {
              e.preventDefault();
              onPageChange('home');
            }}>Detect</a>
          </li>
          <li className="navbar-item">
            <a href="#quiz" className="navbar-link" onClick={(e) => {
              e.preventDefault();
              onPageChange('quiz');
            }}>Quiz</a>
          </li>
          <li className="navbar-item">
            <div className="gamification-pill" onClick={onGamificationClick}>
              <span className="level">Level {level}</span>
              <span className="xp">{xp} XP</span>
              <span className="stage">{levelName}</span>
            </div>
          </li>
        </ul>
      </div>
    </nav>
  );
};

export default Navbar;