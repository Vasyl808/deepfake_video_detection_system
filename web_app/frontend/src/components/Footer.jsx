import React from 'react';
import '../styles/Footer.css';

function Footer() {
  const currentYear = new Date().getFullYear();
  
  return (
    <footer className="footer">
      <div className="footer-container">
        <div className="footer-top">
          <div className="footer-brand">
            <h3>Deepfake Detector</h3>
            <p className="footer-tagline">Виявляємо підробки, захищаємо правду</p>
          </div>
          <div className="footer-links">
            <div className="footer-links-column">
              <h4>Навігація</h4>
              <ul>
                <li><a href="/">Головна</a></li>
                <li><a href="/about">Про сервіс</a></li>
                <li><a href="/faq">FAQ</a></li>
                <li><a href="/contact">Контакти</a></li>
              </ul>
            </div>
            <div className="footer-links-column">
              <h4>Ресурси</h4>
              <ul>
                <li><a href="/blog">Блог</a></li>
                <li><a href="/api-docs">API документація</a></li>
                <li><a href="/research">Дослідження</a></li>
              </ul>
            </div>
            <div className="footer-links-column">
              <h4>Зв'язок</h4>
              <ul>
                <li><a href="mailto:info@deepfakedetector.com">Email</a></li>
                <li><a href="https://twitter.com" target="_blank" rel="noopener noreferrer">Twitter</a></li>
                <li><a href="https://linkedin.com" target="_blank" rel="noopener noreferrer">LinkedIn</a></li>
              </ul>
            </div>
          </div>
        </div>
        <div className="footer-bottom">
          <p className="copyright">© {currentYear} Deepfake Detector. Всі права захищено.</p>
          <div className="footer-legal">
            <a href="/privacy">Політика конфіденційності</a>
            <a href="/terms">Умови використання</a>
          </div>
        </div>
      </div>
    </footer>
  );
}

export default Footer;