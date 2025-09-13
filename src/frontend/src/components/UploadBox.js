
import React, { useState } from 'react';

function UploadBox({ onDetection }) {
  const [file, setFile] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!file) {
      alert('Please select a file first!');
      return;
    }

    setIsLoading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8000/detect', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Detection failed');
      }

      const result = await response.json();
      onDetection(result);
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
      <form onSubmit={handleSubmit}>
        <input type="file" onChange={handleFileChange} />
        <button type="submit" disabled={isLoading}>
          {isLoading ? 'Analyzing...' : 'Analyze'}
        </button>
      </form>
    </div>
  );
}

export default UploadBox;
