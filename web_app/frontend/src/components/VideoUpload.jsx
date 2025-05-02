import React, { useState, useRef, useEffect, useCallback, useId } from 'react';
import JSZip from 'jszip';
import '../styles/VideoUpload.css';

const API_URL = 'https://8000-vasyl808-deepfakevideod-vxui3rgqzba.ws-eu118.gitpod.io';
const MAX_FILE_SIZE = 500 * 1024 * 1024; // 500MB

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
  const [downloadLoadingIdx, setDownloadLoadingIdx] = useState(null);
  const [downloadErrorIdx, setDownloadErrorIdx] = useState({});

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

  // Соцмережа
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

  // --- Download all frames/gradcam of a sequence in ZIP ---
  async function fetchImageAsBlob(path) {
    // path is like "/analyzed_frames/uuid/sequence_0/frame_0.jpg"
    // API_URL might end without a slash, so avoid double slashes
    const url = path.startsWith('/') ? `${API_URL}${path}` : `${API_URL}/${path}`;
    const resp = await fetch(url);
    if (!resp.ok) throw new Error('Не вдалося завантажити зображення');
    return await resp.blob();
  }

  function addExplanationToZip(zip, explanationText) {
    if (explanationText) {
      zip.file('explanation.txt', explanationText);
    }
  }

  // Завантажити звичайні кадри однієї послідовності
  const handleDownloadFrames = async (seqIdx) => {
    setDownloadLoadingIdx(seqIdx);
    setDownloadErrorIdx(prev => ({ ...prev, [seqIdx]: '' }));
    try {
      const seq = result?.sequences?.[seqIdx];
      if (!seq) throw new Error('Не знайдено послідовність для завантаження');
      const framesArray = seq.frames;
      if (!framesArray || !framesArray.length) throw new Error('Немає кадрів для завантаження');
      const zip = new JSZip();
      for (let i = 0; i < framesArray.length; ++i) {
        const { frame_number, image } = framesArray[i];
        const blob = await fetchImageAsBlob(image);
        zip.file(`frame_${frame_number}.jpg`, blob);
      }
      const content = await zip.generateAsync({ type: 'blob' });
      const url = window.URL.createObjectURL(content);
      const a = document.createElement('a');
      a.href = url;
      a.download = `sequence_${seqIdx + 1}_frames.zip`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      setDownloadErrorIdx(prev => ({ ...prev, [seqIdx]: `Помилка завантаження кадрів: ${err.message}` }));
    } finally {
      setDownloadLoadingIdx(null);
    }
  };

  // Завантажити Grad-CAM кадри однієї послідовності + пояснення
  const handleDownloadGradcam = async (seqIdx) => {
    setDownloadLoadingIdx(seqIdx);
    setDownloadErrorIdx(prev => ({ ...prev, [seqIdx]: '' }));
    try {
      const seq = result?.sequences?.[seqIdx];
      if (!seq) throw new Error('Не знайдено послідовність для завантаження');
      const gradcamArray = seq.gradcam;
      if (!gradcamArray || !gradcamArray.length) throw new Error('Немає Grad-CAM кадрів для завантаження');
      const zip = new JSZip();
      for (let i = 0; i < gradcamArray.length; ++i) {
        const { frame_number, image } = gradcamArray[i];
        const blob = await fetchImageAsBlob(image);
        zip.file(`gradcam_${frame_number}.jpg`, blob);
      }
      addExplanationToZip(zip, seq.explanation);
      const content = await zip.generateAsync({ type: 'blob' });
      const url = window.URL.createObjectURL(content);
      const a = document.createElement('a');
      a.href = url;
      a.download = `sequence_${seqIdx + 1}_gradcam.zip`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      setDownloadErrorIdx(prev => ({ ...prev, [seqIdx]: `Помилка завантаження Grad-CAM: ${err.message}` }));
    } finally {
      setDownloadLoadingIdx(null);
    }
  };

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
                    {seq.is_fake ? 'Фейк' : 'Реальна'}
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
                            src={`${API_URL}${image}`}
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
                    <div className="frames-container">
                      {seq.gradcam?.map(({ frame_number, image }) => (
                        <div key={frame_number} className="frame-card">
                          <img
                            src={`${API_URL}${image}`}
                            alt={`Grad-CAM ${frame_number}`}
                            className="frame-image"
                          />
                          <p className="frame-number">Кадр {frame_number}</p>
                        </div>
                      ))}
                    </div>
                    <div className="explanation-box">
                      <h5 className="explanation-title">Результат аналізу:</h5>
                      <p className="explanation">{seq.explanation}</p>
                    </div>
                  </div>

                  {/* --- DOWNLOAD BUTTONS --- */}
                  <div className="download-frames-section">
                    <h4>Завантажити кадри цієї послідовності</h4>
                    <div className="download-frames-controls">
                      <button
                        className={`download-frames-btn modern-blue ${downloadLoadingIdx === idx ? 'loading' : ''}`}
                        disabled={downloadLoadingIdx === idx}
                        onClick={() => handleDownloadFrames(idx)}
                        type="button"
                      >
                        {downloadLoadingIdx === idx
                          ? (<><span className="loader-dots"></span> Завантаження...</>)
                          : (<><svg className="download-icon" width="18" height="18" viewBox="0 0 20 20" fill="none"><path d="M10 4v8m0 0l-3-3m3 3l3-3m-8 6h14" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/></svg> Завантажити кадри</>)
                        }
                      </button>
                      <button
                        className={`download-frames-btn modern-green ${downloadLoadingIdx === idx ? 'loading' : ''}`}
                        disabled={downloadLoadingIdx === idx}
                        onClick={() => handleDownloadGradcam(idx)}
                        type="button"
                      >
                        {downloadLoadingIdx === idx
                          ? (<><span className="loader-dots"></span> Завантаження...</>)
                          : (<><svg className="download-icon" width="18" height="18" viewBox="0 0 20 20" fill="none"><path d="M10 4v8m0 0l-3-3m3 3l3-3m-8 6h14" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/></svg> Кадри з Grad-CAM</>)
                        }
                      </button>
                    </div>
                    {downloadErrorIdx[idx] && (
                      <div className="error-message">{downloadErrorIdx[idx]}</div>
                    )}
                    <div className={`download-explanation ${downloadLoadingIdx === idx ? 'active' : ''}`}>
                      <svg className="info-icon" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2" style={{marginRight: 6, verticalAlign:'middle'}} viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>
                      <span>
                        Ви можете завантажити <b>всі кадри цієї послідовності</b> у zip-архіві, або отримати <b>Grad-CAM</b> з поясненням для глибшого аналізу!
                      </span>
                    </div>
                  </div>
                  {/* --- END DOWNLOAD BUTTONS --- */}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}