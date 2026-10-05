import React from 'react';
import { Layers, Cpu, Eye, CheckCircle } from 'lucide-react';

export default function InfoGuide() {
  return (
    <div className="info-section">
      <div className="card-title-group">
        <h2 className="card-title">
          <Layers size={20} color="#2563eb" /> How the AI Vision Screening Works
        </h2>
        <p className="card-desc">
          Automated ocular feature screening powered by Deep Transfer Learning (MobileNetV2).
        </p>
      </div>

      <div className="info-grid">
        <div className="info-item">
          <div className="info-item-title">
            <Cpu size={18} color="#2563eb" /> 1. Transfer Learning
          </div>
          <div className="info-item-desc">
            Leverages a deep MobileNetV2 architecture pre-trained on millions of features to detect fine corneal contours and reflection anomalies.
          </div>
        </div>

        <div className="info-item">
          <div className="info-item-title">
            <Eye size={18} color="#10b981" /> 2. Spectacle Prediction
          </div>
          <div className="info-item-desc">
            Binary decision head directly classifies whether an individual requires corrective prescription spectacles or demonstrates normal vision.
          </div>
        </div>

        <div className="info-item">
          <div className="info-item-title">
            <CheckCircle size={18} color="#f59e0b" /> 3. Extensible Dataset
          </div>
          <div className="info-item-desc">
            Structured standard training pipeline ready to seamlessly train on new real-world clinical or fundus/anterior datasets.
          </div>
        </div>
      </div>
    </div>
  );
}
