
import React, { useState } from 'react';

function UploadBox({ onDetection, onAIGeneratedDetection }) {
  const [file, setFile] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  const handleSubmit = async (event, isAIGenerated = false) => {
    event.preventDefault();
    if (!file) {
      alert('Please select a file first!');
      return;
    }

    setIsLoading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const endpoint = isAIGenerated ? 'http://localhost:8000/detect-ai-generated' : 'http://localhost:8000/detect';
      const response = await fetch(endpoint, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Detection failed');
      }

      const result = await response.json();
      if (isAIGenerated) {
        onAIGeneratedDetection(result);
      } else {
        onDetection(result);
      }
    } catch (error) {
      console.error('Error during detection:', error);
      alert('An error occurred during detection. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="upload-box">
      <h2 className="upload-title">🔍 File Analysis</h2>
      <div className="upload-instructions">
        <p>Upload an image or video file to check for deepfakes or AI generation</p>
      </div>
      <form onSubmit={(e) => e.preventDefault()} className="upload-form">
        <div className="file-input-container">
          <label className="file-input-label">
            <span className="file-input-text">
              {file ? '📁 ' + file.name : 'Choose File'}
            </span>
            <input 
              type="file" 
              onChange={handleFileChange} 
              className="file-input"
              accept="image/*,video/*"
            />
          </label>
          {file && (
            <div className="file-info">
              <span className="file-name">{file.name}</span>
              <span className="file-size">({(file.size / 1024 / 1024).toFixed(2)} MB)</span>
            </div>
          )}
        </div>
        <div className="button-group">
          <button 
            type="button" 
            onClick={(e) => handleSubmit(e, false)} 
            disabled={isLoading}
            className="detect-button"
          >
            {isLoading ? '🔍 Analyzing...' : '🕵️ Deepfake Detection'}
          </button>
          <button 
            type="button" 
            onClick={(e) => handleSubmit(e, true)} 
            disabled={isLoading}
            className="ai-button"
          >
            {isLoading ? '🤖 Checking...' : '🤖 AI Generation Check'}
          </button>
        </div>
      </form>
    </div>
  );
}

export default UploadBox;
