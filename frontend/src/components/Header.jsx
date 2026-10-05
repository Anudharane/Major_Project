import React from 'react';
import { Eye, ShieldCheck } from 'lucide-react';

export default function Header() {
  return (
    <header className="app-header">
      <div className="brand-section">
        <div className="brand-icon">
          <Eye size={26} />
        </div>
        <div>
          <h1 className="brand-title">AI Vision Screening System</h1>
          <p className="brand-subtitle">
            Automated Spectacle Requirement Determination via Deep Transfer Learning
          </p>
        </div>
      </div>
      <div className="badge-status">
        <span className="status-dot"></span>
        <span>MobileNetV2 CNN Active</span>
      </div>
    </header>
  );
}
