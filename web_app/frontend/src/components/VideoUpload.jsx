import React, { useState, useRef, useEffect, useCallback, useId } from 'react';
import '../styles/VideoUpload.css';

const API_URL = 'https://8000-vasyl808-deepfakevideod-jlmmzvo0yfl.ws-eu118.gitpod.io';
const MAX_FILE_SIZE = 500 * 1024 * 1024; // 500MB в байтах

export default function VideoUpload() {
  const inputId = useId();
  const videoRef = useRef(null);
  const [videoFile, setVideoFile] = useState(null);
  const [videoURL, setVideoURL] = useState('');
  const [socialMediaURL, setSocialMediaURL] = useState('');
  const [duration, setDuration] = useState(0);
  const [startTime, setStartTime] = useState(0);
  const [manualTime, setManualTime] = useState('0');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [expanded, setExpanded] = useState({});
  const [uploadMethod, setUploadMethod] = useState('file');
  const [error, setError] = useState('');

  useEffect(() => {
    return () => {
      if (videoURL && videoURL.startsWith('blob:')) URL.revokeObjectURL(videoURL);
    };
  }, [videoURL]);

  useEffect(() => {
    if (videoRef.current) videoRef.current.load();
  }, [videoURL]);

  const resetState = useCallback(() => {
    setDuration(0);
    setStartTime(0);
    setManualTime('0');
    setResult(null);
    setError('');
  }, []);

  const handleFile = useCallback((file) => {
    if (!file) return;
    if (file.size > MAX_FILE_SIZE) {
      setError(`Файл занадто великий. Максимальний розмір: 500MB`);
      return;
    }
    resetState();
    setVideoFile(file);
    const url = URL.createObjectURL(file);
    setVideoURL(url);
  }, [resetState]);

  const handleFileChange = useCallback((e) => {
    handleFile(e.target.files?.[0] || null);
  }, [handleFile]);

  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    handleFile(e.dataTransfer.files?.[0] || null);
  }, [handleFile]);

  const onLoadedMetadata = useCallback((e) => {
    const d = Math.floor(e.target.duration);
    setDuration(d);
    setStartTime(0);
    setManualTime('0');
  }, []);

  const clamp = (value, min, max) => Math.min(Math.max(value, min), max);

  const handleSliderChange = useCallback((e) => {
    const val = clamp(Number(e.target.value), 0, duration - 10);
    setStartTime(val);
    setManualTime(String(val));
  }, [duration]);

  const handleManualChange = useCallback((e) => {
    const val = e.target.value;
    setManualTime(val);
    const num = Number(val);
    if (!isNaN(num) && num >= 0 && num <= duration - 10) {
      setStartTime(Math.floor(num));
    }
  }, [duration]);

  const handleSocialMediaURLChange = useCallback((e) => {
    setSocialMediaURL(e.target.value);
  }, []);

  // ОНОВЛЕНИЙ БЛОК: Завантаження відео з соцмережі без кукі
  const fetchVideoFromURL = async () => {
    if (!socialMediaURL.trim()) {
      setError('Будь ласка, введіть URL відео');
      return;
    }
    setLoading(true);
    setError('');
    setResult(null);
    try {
      const response = await fetch(`${API_URL}/download-from-url`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ url: socialMediaURL.trim() }),
      });

      if (!response.ok) {
        const errorData = await response.text();
        throw new Error(errorData || 'Не вдалося завантажити відео');
      }

      const data = await response.json();
      setVideoURL(`${API_URL}${data.video_url}`);

      // Створення об'єкта File з URL для подальшої відправки
      const response2 = await fetch(`${API_URL}${data.video_url}`);
      const blob = await response2.blob();
      const file = new File([blob], data.filename, { type: blob.type });
      setVideoFile(file);
    } catch (err) {
      setError(`Помилка: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = useCallback(async (e) => {
    e.preventDefault();
    if (!videoFile) return;

    setLoading(true);
    setResult(null);
    setError('');

    try {
      const form = new FormData();
      form.append('file', videoFile);
      form.append('startTime', String(startTime));
      form.append('duration', '10');

      const response = await fetch(`${API_URL}/analyze`, { method: 'POST', body: form });
      if (!response.ok) {
        const message = await response.text();
        throw new Error(message || 'Помилка завантаження');
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      console.error(err);
      setError(`Сталася помилка: ${err.message}`);
    } finally {
      setLoading(false);
    }
  }, [videoFile, startTime]);

  const toggle = useCallback((idx) => {
    setExpanded(prev => ({ ...prev, [idx]: !prev[idx] }));
  }, []);

  const toggleUploadMethod = useCallback((method) => {
    setUploadMethod(method);
    resetState();
    if (method === 'file') {
      setSocialMediaURL('');
    } else {
      if (videoURL && videoURL.startsWith('blob:')) URL.revokeObjectURL(videoURL);
      setVideoURL('');
      setVideoFile(null);
    }
  }, [resetState, videoURL]);

  return (
    <div className="upload-section">
      {error && <div className="error-message">{error}</div>}

      <form
          onSubmit={handleSubmit}
          className={`upload-form${uploadMethod === 'url' ? ' url-mode' : ''}`}>
        <div className="upload-tabs">
          <button
            type="button"
            className={`upload-tab ${uploadMethod === 'file' ? 'active' : ''}`}
            onClick={() => toggleUploadMethod('file')}
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="upload-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"></path>
              <polyline points="13 2 13 9 20 9"></polyline>
            </svg>
            З комп'ютера
          </button>
          <button
            type="button"
            className={`upload-tab ${uploadMethod === 'url' ? 'active' : ''}`}
            onClick={() => toggleUploadMethod('url')}
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="upload-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path>
              <polyline points="15 3 21 3 21 9"></polyline>
              <line x1="10" y1="14" x2="21" y2="3"></line>
            </svg>
            З соцмережі
          </button>
        </div>

        <div className="upload-content">
          {uploadMethod === 'file' ? (
            <div
              className={`drop-area ${dragActive ? 'active' : ''} ${videoFile ? 'has-file' : ''}`}
              onDragEnter={handleDrag}
              onDragOver={handleDrag}
              onDragLeave={handleDrag}
              onDrop={handleDrop}
              onClick={() => document.getElementById(inputId).click()}
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="upload-big-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                <polyline points="17 8 12 3 7 8"></polyline>
                <line x1="12" y1="3" x2="12" y2="15"></line>
              </svg>

              {videoFile
                ? <p className="file-name">{videoFile.name}</p>
                : <p>Перетягніть відео сюди або клікніть для вибору</p>
              }
              <p className="file-size-limit">Максимальний розмір: 500MB</p>

              <input
                id={inputId}
                type="file"
                accept="video/*"
                className="file-input"
                onChange={handleFileChange}
                style={{ display: 'none' }}
              />
            </div>
          ) : (
            <div className="url-input-wrapper">
              <div className="url-input-container">
                <svg xmlns="http://www.w3.org/2000/svg" className="url-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"></path>
                  <path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"></path>
                </svg>
                <input
                  type="text"
                  value={socialMediaURL}
                  onChange={handleSocialMediaURLChange}
                  placeholder="Вставте посилання на відео (YouTube, Instagram, TikTok...)"
                  className="url-input"
                />
                <button
                  type="button"
                  onClick={fetchVideoFromURL}
                  className="url-submit-btn"
                  disabled={loading || !socialMediaURL.trim()}
                >
                  <svg xmlns="http://www.w3.org/2000/svg" className="url-btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                    <polyline points="14 2 14 8 20 8"></polyline>
                    <line x1="9" y1="15" x2="15" y2="15"></line>
                  </svg>
                </button>
              </div>
              <div className="social-icons">
                <span className="social-icon youtube"></span>
                <span className="social-icon instagram"></span>
                <span className="social-icon tiktok"></span>
                <span className="social-icon facebook"></span>
              </div>
            </div>
          )}
        </div>

        {videoURL && (
          <div className="video-preview fade-in">
            <h3 className="preview-title">Перегляд відео</h3>
            <video
              ref={videoRef}
              src={videoURL}
              controls
              onLoadedMetadata={onLoadedMetadata}
              className="video-player"
            />

            {duration >= 10 && (
              <div className="trim-controls">
                <h4 className="trim-title">Виберіть 10-секундний сегмент для аналізу</h4>
                <div className="slider-container">
                  <input
                    type="range"
                    min="0"
                    max={duration - 10}
                    value={startTime}
                    onChange={handleSliderChange}
                    className="time-slider"
                  />
                  <div className="time-labels">
                    <span>0с</span>
                    <span>{Math.floor(duration - 10)}с</span>
                  </div>
                </div>
                <div className="time-display">
                  Вибраний інтервал: <span className="time-value">{startTime}с</span> – <span className="time-value">{startTime + 10}с</span>
                  <div className="manual-time">
                    Точний час початку:
                    <input
                      type="number"
                      min="0"
                      max={duration - 10}
                      value={manualTime}
                      onChange={handleManualChange}
                      className="manual-time-input"
                    />
                    <span>с</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        <button
          type="submit"
          className="submit-button"
          disabled={loading || !videoFile}
        >
          {loading ? (
            <>
              <span className="spinner-small"></span>
              Обробка...
            </>
          ) : (
            <>
              <svg xmlns="http://www.w3.org/2000/svg" className="check-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="20 6 9 17 4 12"></polyline>
              </svg>
              Перевірити відео
            </>
          )}
        </button>
      </form>

      {loading && (
        <div className="spinner-container fade-in">
          <div className="spinner" />
          <p className="processing-text">Відео обробляється...</p>
          <p className="wait-text">Будь ласка, зачекайте</p>
        </div>
      )}

      {result?.sequences?.length > 0 && (
        <div className="results fade-in">
          <h2 className="results-title">Результати аналізу</h2>
          {result.sequences.map((seq, idx) => (
            <div key={idx} className={`sequence-card ${seq.is_fake ? 'fake' : 'real'}`}>
              <div className="sequence-header">
                <div className="sequence-info">
                  <h3>Послідовність {idx + 1}</h3>
                  <div className={`sequence-type ${seq.is_fake ? 'fake' : 'real'}`}>
                    {seq.is_fake ? 'Фейк' : 'Реальне відео'}
                  </div>
                </div>
                <button onClick={() => toggle(idx)} className="toggle-details-btn">
                  {expanded[idx] ? 'Сховати деталі' : 'Показати деталі'}
                </button>
              </div>

              {expanded[idx] && (
                <div className="sequence-details fade-in">
                  <div className="frames-section">
                    <h4 className="frames-title">Проаналізовані кадри</h4>
                    <div className="frames-container">
                      {seq.frames?.map(({ frame_number, image }) => (
                        <div key={frame_number} className="frame-card">
                          <img
                            src={`data:image/jpeg;base64,${image}`}
                            alt={`Кадр ${frame_number}`}
                            className="frame-image"
                          />
                          <p className="frame-number">Кадр {frame_number}</p>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="gradcam-section">
                    <h4 className="gradcam-title">Grad-CAM візуалізація</h4>
                    <div className="gradcam-info">
                      <p className="gradcam-description">
                        Яскраві області показують ділянки, які алгоритм вважає підозрілими
                      </p>
                    </div>
                    <div className="frames-container">
                      {seq.gradcam?.map(({ frame_number, image }) => (
                        <div key={frame_number} className="frame-card">
                          <img
                            src={`data:image/jpeg;base64,${image}`}
                            alt={`Grad-CAM ${frame_number}`}
                            className="frame-image"
                          />
                          <p className="frame-number">Кадр {frame_number}</p>
                        </div>
                      ))}
                    </div>
                    <div className="explanation-box">
                      <h5 className="explanation-title">Пояснення:</h5>
                      <p className="explanation">{seq.explanation}</p>
                    </div>
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}