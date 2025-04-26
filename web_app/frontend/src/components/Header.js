import React, { useState, useEffect } from 'react';
import '../styles/Header.css';

function Header({ title }) {
  const [scrolled, setScrolled] = useState(false);
  
  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 10);
    };
    
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <header className={`header ${scrolled ? 'header-scrolled' : ''}`}>
      <div className="header-container">
        <div className="header-logo">
          <img src="/api/placeholder/32/32" alt="Logo" className="logo-image" />
          <h1>{title}</h1>
        </div>
        <nav className="header-nav">
          <ul className="nav-links">
            <li><a href="/" className="nav-link active">Головна</a></li>
            <li><a href="/about" className="nav-link">Про сервіс</a></li>
            <li><a href="/faq" className="nav-link">FAQ</a></li>
          </ul>
        </nav>
        <div className="header-actions">
          <button className="btn-dark-mode">
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"></path>
            </svg>
          </button>
        </div>
      </div>
    </header>
  );
}

export default Header;