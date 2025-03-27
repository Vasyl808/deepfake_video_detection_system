import React, { useState, useRef } from 'react';
import axios from 'axios';
import '../styles/VideoUpload.css';

function VideoUpload() {
  const [videoFile, setVideoFile] = useState(null);
  const [videoURL, setVideoURL] = useState('');
  const [videoDuration, setVideoDuration] = useState(0);
  const [selectedStartTime, setSelectedStartTime] = useState(0);
  const [manualStartTime, setManualStartTime] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  // Стан для зберігання, які послідовності розгорнуті
  const [expandedSequences, setExpandedSequences] = useState({});

  const videoRef = useRef(null);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setVideoFile(file);
      const url = URL.createObjectURL(file);
      setVideoURL(url);
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
      const file = e.dataTransfer.files[0];
      setVideoFile(file);
      const url = URL.createObjectURL(file);
      setVideoURL(url);
    }
  };

  const handleLoadedMetadata = (e) => {
    const duration = e.target.duration;
    setVideoDuration(duration);
    if (duration >= 10) {
      setSelectedStartTime(0);
      setManualStartTime("0");
    }
  };

  // Handler для слайдера
  const handleStartTimeChange = (e) => {
    const value = Number(e.target.value);
    setSelectedStartTime(value);
    setManualStartTime(value.toString());
  };

  // Handler для ручного вводу часу
  const handleManualInputChange = (e) => {
    const value = e.target.value;
    setManualStartTime(value);
    const numericValue = Number(value);
    if (!isNaN(numericValue) && numericValue >= 0 && numericValue <= Math.floor(videoDuration - 10)) {
      setSelectedStartTime(numericValue);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!videoFile) return;
    setResult(null);
    setLoading(true);

    const formData = new FormData();
    formData.append('file', videoFile);
    formData.append('startTime', selectedStartTime.toString());
    formData.append('duration', '10');

    try {
      const response = await axios.post(
        'https://8000-vasyl808-deepfakevideod-imjjq7m1xnf.ws-eu118.gitpod.io/analyze',
        formData,
        { headers: { 'Content-Type': 'multipart/form-data' } }
      );
      setResult(response.data);
    } catch (error) {
      console.error('Error during upload:', error);
      alert('Помилка під час завантаження відео. Спробуйте ще раз.');
    }
    setLoading(false);
  };

  // Функція для перемикання відображення деталей для послідовності
  const toggleSequence = (index) => {
    setExpandedSequences(prev => ({
      ...prev,
      [index]: !prev[index]
    }));
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
            style={{display: "none"}}
          />
        </div>
        {videoURL && (
          <div className="video-preview">
            <video 
              src={videoURL} 
              controls 
              ref={videoRef} 
              onLoadedMetadata={handleLoadedMetadata}
              style={{maxWidth: "100%"}}
            />
            {videoDuration >= 10 && (
              <div className="trim-controls">
                <label>
                  Оберіть початок 10-секундного сегмента (від 0 до {Math.floor(videoDuration - 10)} сек):
                </label>
                <input 
                  type="range" 
                  min="0"
                  max={Math.floor(videoDuration - 10)}
                  value={selectedStartTime}
                  onChange={handleStartTimeChange}
                />
                <p>
                  Вибрано проміжок: {selectedStartTime} сек - {selectedStartTime + 10} сек
                </p>
                <label>
                  Або введіть число:
                  <input 
                    type="number"
                    min="0"
                    max={Math.floor(videoDuration - 10)}
                    value={manualStartTime}
                    onChange={handleManualInputChange}
                  />
                </label>
              </div>
            )}
          </div>
        )}
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
          {result.sequences && result.sequences.map((seq, index) => (
            <div key={index} className="sequence-card">
              <div className="sequence-header">
                <h3>Послідовність {index + 1}</h3>
                <button 
                  onClick={() => toggleSequence(index)} 
                  className="toggle-details-btn"
                >
                  {expandedSequences[index] ? 'Сховати деталі' : 'Показати деталі'}
                </button>
              </div>
              <p>Тип: {seq.is_fake ? 'Фейк' : 'Реальна'}</p>
              {expandedSequences[index] && (
                <>
                  <div className="frames-container">
                    {seq.frames && seq.frames.map((frame) => (
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
                    <h4>Grad-CAM</h4>
                    <div className="frames-container">
                      {seq.gradcam && seq.gradcam.map((frame) => (
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
                </>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default VideoUpload;