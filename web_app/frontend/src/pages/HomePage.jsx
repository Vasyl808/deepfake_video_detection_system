import React, { useState } from 'react';
import Header from '../components/Header';
import VideoUpload from '../components/VideoUpload';
import Footer from '../components/Footer';
import '../styles/HomePage.css';

function HomePage() {
  const [showUpload, setShowUpload] = useState(false);

  return (
    <div className="home-page">
      <Header title="Deepfake Detector" />
      <main className="main-content">
        {!showUpload ? (
          <>
            <div className="hero-section">
              <h2>Виявлення підроблених відео за допомогою ШІ</h2>
              <p>Наш сервіс дозволяє перевірити автентичність відео та виявити признаки можливих маніпуляцій.</p>
              <div className="hero-cta">
                <button
                  className="btn btn-primary"
                  onClick={() => setShowUpload(true)}
                >
                  Спробувати зараз
                </button>
                <button className="btn btn-outline">
                  Дізнатись більше
                </button>
              </div>
            </div>

            <section className="features-section">
              <h3>Як це працює</h3>
              <div className="features-grid">
                <div className="feature-card">
                  <div className="feature-icon">
                    {/* SVG іконка */}
                    📁
                  </div>
                  <h4>Завантаження</h4>
                  <p>Просто завантажте відео, яке хочете перевірити.</p>
                </div>
                <div className="feature-card">
                  <div className="feature-icon">
                    {/* SVG іконка */}
                    🔍
                  </div>
                  <h4>Аналіз</h4>
                  <p>Наш ШІ аналізує відео та виявляє ознаки маніпуляцій.</p>
                </div>
                <div className="feature-card">
                  <div className="feature-icon">
                    {/* SVG іконка */}
                    ✅
                  </div>
                  <h4>Результат</h4>
                  <p>Отримайте детальний звіт щодо справжності відео.</p>
                </div>
              </div>
            </section>
          </>
        ) : (
          <section className="upload-section">
            <h3>Завантажте ваше відео</h3>
            <VideoUpload />
          </section>
        )}
      </main>
      <Footer />
    </div>
  );
}

export default HomePage;
