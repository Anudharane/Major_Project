import React, { useState } from 'react';
import Header from './components/Header';
import ImageUploader from './components/ImageUploader';
import WebcamCapture from './components/WebcamCapture';
import ResultCard from './components/ResultCard';
import InfoGuide from './components/InfoGuide';
import { Upload, Camera, Sparkles, AlertCircle } from 'lucide-react';
import './App.css';

export default function App() {
  const [activeTab, setActiveTab] = useState('upload'); // 'upload' or 'webcam'
  const [selectedImage, setSelectedImage] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleImageSelected = (base64Data, file) => {
    setSelectedImage(base64Data);
    setSelectedFile(file || null);
    setResult(null);
    setError(null);
  };

  const handleClearImage = () => {
    setSelectedImage(null);
    setSelectedFile(null);
    setResult(null);
    setError(null);
  };

  const handleAnalyze = async () => {
    if (!selectedImage && !selectedFile) {
      setError('Please upload an image or capture one using your webcam first.');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      // Create FormData
      const formData = new FormData();
      if (selectedFile) {
        formData.append('file', selectedFile);
      } else if (selectedImage) {
        formData.append('image_base64', selectedImage);
      }

      // Send to FastAPI backend
      let response;
      try {
        response = await fetch('/api/predict', {
          method: 'POST',
          body: formData,
        });
      } catch {
        response = await fetch('http://localhost:8000/predict', {
          method: 'POST',
          body: formData,
        });
      }

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server error (${response.status})`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      console.error('Prediction error:', err);
      setError(
        err.message || 'Failed to communicate with AI Backend. Make sure backend is running on http://localhost:8000'
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <Header />

      <div className="main-grid">
        {/* Left Card: Input Acquisition */}
        <div className="glass-card">
          <div className="card-title-group">
            <h2 className="card-title">1. Eye Image Acquisition</h2>
            <p className="card-desc">
              Provide an eye or corneal photograph for automated AI spectacle screening.
            </p>
          </div>

          <div className="tab-switcher">
            <button
              className={`tab-btn ${activeTab === 'upload' ? 'active' : ''}`}
              onClick={() => setActiveTab('upload')}
            >
              <Upload size={16} /> Upload Image
            </button>
            <button
              className={`tab-btn ${activeTab === 'webcam' ? 'active' : ''}`}
              onClick={() => setActiveTab('webcam')}
            >
              <Camera size={16} /> Live Webcam
            </button>
          </div>

          {activeTab === 'upload' ? (
            <ImageUploader
              selectedImage={selectedImage}
              onImageSelected={handleImageSelected}
              onClearImage={handleClearImage}
            />
          ) : (
            <WebcamCapture
              onCapture={(base64) => {
                handleImageSelected(base64, null);
                setActiveTab('upload');
              }}
              onCancel={() => setActiveTab('upload')}
            />
          )}

          {error && (
            <div
              style={{
                marginTop: 14,
                padding: '10px 14px',
                background: '#fef2f2',
                border: '1px solid #fecaca',
                borderRadius: 8,
                color: '#b91c1c',
                fontSize: '0.84rem',
                display: 'flex',
                alignItems: 'center',
                gap: 8,
              }}
            >
              <AlertCircle size={16} />
              <span>{error}</span>
            </div>
          )}

          <button
            className="btn-primary"
            onClick={handleAnalyze}
            disabled={loading || !selectedImage}
          >
            <Sparkles size={18} />
            {loading ? 'Screening Eye Features...' : 'Run Vision Screening'}
          </button>
        </div>

        {/* Right Card: Screening Results */}
        <div className="glass-card">
          <div className="card-title-group">
            <h2 className="card-title">2. Screening Diagnosis</h2>
            <p className="card-desc">
              Direct prediction of spectacle requirement and confidence score.
            </p>
          </div>

          <ResultCard result={result} loading={loading} />
        </div>
      </div>

      {/* Educational & Technical Guide */}
      <InfoGuide />
    </div>
  );
}
