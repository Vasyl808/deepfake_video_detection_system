import React, { useState } from 'react';
import axios from 'axios';
import './VideoUpload.css';

function VideoUpload() {
  const [videoFile, setVideoFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [dragActive, setDragActive] = useState(false);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setVideoFile(e.target.files[0]);
    }
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setVideoFile(e.dataTransfer.files[0]);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!videoFile) return;
    setResult(null);
    setLoading(true);
    const formData = new FormData();
    formData.append('file', videoFile);

    try {
      const response = await axios.post(
        'https://8000-vasyl808-fastapiapp-yey39s72moa.ws-eu118.gitpod.io/analyze',
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        }
      );
      setResult(response.data);
    } catch (error) {
      console.error('Помилка під час аналізу відео:', error);
      if (error.response) {
        console.error('Server responded with an error:', error.response.data);
      } else if (error.request) {
        console.error('No response received:', error.request);
      } else {
        console.error('Error setting up request:', error.message);
      }
      alert('Сталася помилка під час завантаження відео. Перевірте своє мережеве з\'єднання та спробуйте ще раз.');
    }
    setLoading(false);
  };

  return (
    <div className="upload-section">
      <form onSubmit={handleSubmit} className="upload-form">
        <div
          className={`drop-area ${dragActive ? 'active' : ''}`}
          onDragEnter={handleDrag}
          onDragOver={handleDrag}
          onDragLeave={handleDrag}
          onDrop={handleDrop}
          onClick={() => document.getElementById('video-upload').click()}
        >
          {videoFile ? (
            <p>
              Вибрано файл: <strong>{videoFile.name}</strong>
            </p>
          ) : (
            <p>Перетягніть відео файл сюди або натисніть для вибору</p>
          )}
          <input 
            type="file" 
            accept="video/*" 
            id="video-upload" 
            onChange={handleFileChange} 
            className="file-input"
          />
        </div>
        <button type="submit" disabled={loading || !videoFile} className="submit-button">
          {loading ? 'Обробка...' : 'Завантажити та перевірити'}
        </button>
      </form>
      {loading && (
        <div className="spinner-container">
          <div className="spinner"></div>
          <p>Файл обробляється...</p>
        </div>
      )}
      {result && (
        <div className="results">
          <h2>Результати аналізу</h2>
          {result.sequences.map((seq, index) => (
            <div key={index} className="sequence-card">
              <h3>Послідовність {index + 1}</h3>
              <p>Тип: {seq.is_fake ? 'Фейк' : 'Реальна'}</p>
              <div className="frames-container">
                {seq.frames.map((frame) => (
                  <div key={frame.frame_number} className="frame-card">
                    <img
                      src={`data:image/jpeg;base64,${frame.image}`}
                      alt={`Кадр ${frame.frame_number}`}
                      className="frame-image"
                    />
                    <p>Кадр {frame.frame_number}</p>
                  </div>
                ))}
              </div>
              <div className="gradcam-container">
                <h4>Grad-CAM </h4>
                <div className="frames-container">
                  {seq.gradcam.map((frame) => (
                    <div key={frame.frame_number} className="frame-card">
                      <img
                        src={`data:image/jpeg;base64,${frame.image}`}
                        alt={`Кадр ${frame.frame_number}`}
                        className="frame-image"
                      />
                      <p>Кадр {frame.frame_number}</p>
                    </div>
                  ))}
                </div>
                <p className="explanation">{seq.explanation}</p>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default VideoUpload;