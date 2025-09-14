
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
      alert('An error occurred during detection.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="upload-box">
      <h2>Upload a file to check</h2>
      <form onSubmit={(e) => e.preventDefault()}>
        <input type="file" onChange={handleFileChange} />
        <div style={{ marginTop: '10px' }}>
          <button 
            type="button" 
            onClick={(e) => handleSubmit(e, false)} 
            disabled={isLoading}
            style={{ marginRight: '10px' }}
          >
            {isLoading ? 'Analyzing...' : 'Analyze for Deepfakes'}
          </button>
          <button 
            type="button" 
            onClick={(e) => handleSubmit(e, true)} 
            disabled={isLoading}
          >
            {isLoading ? 'Checking...' : 'Check for AI Generation'}
          </button>
        </div>
      </form>
    </div>
  );
}

export default UploadBox;
