
import React, { useState } from 'react';

function UploadBox({ onDetection, onAIGeneratedDetection, onUnifiedAnalysis }) {
  const [file, setFile] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  const handleSubmit = async (event, detectionType = 'unified') => {
    event.preventDefault();
    if (!file) {
      alert('Please select a file first!');
      return;
    }

    setIsLoading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      let endpoint, response;
      
      switch (detectionType) {
        case 'deepfake':
          endpoint = 'http://localhost:8000/detect';
          response = await fetch(endpoint, {
            method: 'POST',
            body: formData,
          });
          
          if (!response.ok) {
            throw new Error('Deepfake detection failed');
          }
          
          const deepfakeResult = await response.json();
          onDetection(deepfakeResult);
          break;
          
        case 'ai':
          endpoint = 'http://localhost:8000/detect-ai-generated';
          response = await fetch(endpoint, {
            method: 'POST',
            body: formData,
          });
          
          if (!response.ok) {
            throw new Error('AI detection failed');
          }
          
          const aiResult = await response.json();
          onAIGeneratedDetection(aiResult);
          break;
          
        case 'unified':
        default:
          endpoint = 'http://localhost:8000/analyze';
          response = await fetch(endpoint, {
            method: 'POST',
            body: formData,
          });
          
          if (!response.ok) {
            throw new Error('Unified analysis failed');
          }
          
          const unifiedResult = await response.json();
          onUnifiedAnalysis(unifiedResult);
          break;
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
            onClick={(e) => handleSubmit(e, 'unified')} 
            disabled={isLoading}
            className="detect-button"
          >
            {isLoading ? '🔍 Analyzing...' : '🕵️ Unified Analysis'}
          </button>
          <button 
            type="button" 
            onClick={(e) => handleSubmit(e, 'deepfake')} 
            disabled={isLoading}
            className="detect-button"
          >
            {isLoading ? '🔍 Checking...' : '🕵️ Deepfake Detection'}
          </button>
          <button 
            type="button" 
            onClick={(e) => handleSubmit(e, 'ai')} 
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
