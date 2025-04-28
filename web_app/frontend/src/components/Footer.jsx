import React from 'react';
import '../styles/Footer.css';

function Footer() {
  const currentYear = new Date().getFullYear();
  
  return (
    <footer className="footer">
      <div className="footer-container">
        <div className="footer-brand">
          <h3>Deepfake Detector</h3>
        </div>
        
        <div className="footer-links">
          <a href="/">Головна</a>
          <a href="/about">Про сервіс</a>
          <a href="/faq">FAQ</a>
          <a href="/contact">Контакти</a>
        </div>
        
        <p className="copyright">© {currentYear} Deepfake Detector</p>
      </div>
    </footer>
  );
}

export default Footer;