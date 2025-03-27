import React from 'react';
import Header from '../components/Header';
import VideoUpload from '../components/VideoUpload';
import Footer from '../components/Footer';
import '../styles/HomePage.css';

function HomePage() {
  return (
    <div className="home-page">
      <Header title="Deepfake detector" />
      <main className="main-content">
        <VideoUpload />
      </main>
      <Footer />
    </div>
  );
}

export default HomePage;