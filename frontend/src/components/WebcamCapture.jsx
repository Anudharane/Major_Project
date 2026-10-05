import React, { useRef, useState, useEffect } from 'react';
import { Camera, RefreshCw, CheckCircle, AlertTriangle } from 'lucide-react';

export default function WebcamCapture({ onCapture, onCancel }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [stream, setStream] = useState(null);
  const [error, setError] = useState(null);
  const [isReady, setIsReady] = useState(false);

  useEffect(() => {
    startCamera();
    return () => {
      stopCamera();
    };
  }, []);

  const startCamera = async () => {
    setError(null);
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
        audio: false,
      });
      setStream(mediaStream);
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
        videoRef.current.onloadedmetadata = () => {
          videoRef.current.play();
          setIsReady(true);
        };
      }
    } catch (err) {
      console.error('Camera access error:', err);
      setError('Unable to access webcam. Please check browser camera permissions.');
    }
  };

  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      setStream(null);
    }
  };

  const handleCapture = () => {
    if (!videoRef.current || !canvasRef.current) return;
    const video = videoRef.current;
    const canvas = canvasRef.current;

    const vw = video.videoWidth || 640;
    const vh = video.videoHeight || 480;

    // Crop centered square eye region matching the target guide
    const cropSize = Math.floor(Math.min(vw, vh) * 0.55);
    const startX = Math.floor((vw - cropSize) / 2);
    const startY = Math.floor((vh - cropSize) / 2);

    canvas.width = 256;
    canvas.height = 256;

    const ctx = canvas.getContext('2d');
    // Mirror horizontally to match webcam mirror preview
    ctx.translate(256, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(video, startX, startY, cropSize, cropSize, 0, 0, 256, 256);

    const base64Image = canvas.toDataURL('image/jpeg', 0.95);
    stopCamera();
    onCapture(base64Image);
  };

  return (
    <div>
      {error ? (
        <div className="upload-dropzone" style={{ borderColor: '#ef4444', background: '#fef2f2' }}>
          <AlertTriangle size={32} color="#ef4444" />
          <div style={{ color: '#b91c1c', fontSize: '0.9rem', fontWeight: 600 }}>{error}</div>
          <button className="btn-secondary" onClick={startCamera}>
            <RefreshCw size={16} /> Retry Camera
          </button>
        </div>
      ) : (
        <div>
          <div className="webcam-box">
            <video ref={videoRef} autoPlay playsInline muted className="webcam-video" />
            <div className="webcam-overlay">
              <div className="eye-target-guide" title="Align eye within guide"></div>
            </div>
          </div>
          <canvas ref={canvasRef} style={{ display: 'none' }} />
          <div className="webcam-controls">
            <button
              className="btn-primary"
              style={{ marginTop: 0 }}
              onClick={handleCapture}
              disabled={!isReady}
            >
              <Camera size={18} /> Capture Eye Snapshot
            </button>
            <button
              className="btn-secondary"
              onClick={() => {
                stopCamera();
                onCancel();
              }}
            >
              Cancel
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
