import React from 'react';
import { CheckCircle2, AlertOctagon, Glasses, Sparkles, Clock, Shield, Scan, Sliders, Cpu } from 'lucide-react';

export default function ResultCard({ result, loading }) {
  if (loading) {
    return (
      <div className="empty-placeholder">
        <div className="brand-icon" style={{ animation: 'spin 1.5s linear infinite' }}>
          <Sparkles size={24} />
        </div>
        <div style={{ fontWeight: 700, fontSize: '1rem', marginTop: 12 }}>
          Extracting Ocular Features & Screening...
        </div>
        <div style={{ fontSize: '0.82rem', color: '#64748b' }}>
          ResNet-18 model is analyzing spectacle requirement...
        </div>
        <style>{`
          @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
          }
        `}</style>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="empty-placeholder">
        <Glasses className="placeholder-icon" size={56} />
        <div style={{ fontWeight: 600, color: '#475569' }}>No Screening Performed Yet</div>
        <div style={{ fontSize: '0.82rem', maxWidth: 320 }}>
          Upload an eye photograph or use the webcam to check whether spectacles are needed.
        </div>
      </div>
    );
  }

  const isSpectacleNeeded = result.spectacles_needed;
  const badgeClass = isSpectacleNeeded ? 'warning' : 'success';
  const features = result.extracted_features || null;
  const refinedPreview = result.refined_image_preview || null;

  return (
    <div>
      {/* Primary Result Banner */}
      <div className={`result-banner ${badgeClass}`}>
        <div className={`banner-icon-box ${badgeClass}`}>
          {isSpectacleNeeded ? '👓' : '✅'}
        </div>
        <div>
          <div className="banner-heading">
            {isSpectacleNeeded ? 'Spectacles Needed' : 'No Spectacles Needed'}
          </div>
          <div className="banner-confidence">
            Confidence: <strong>{result.confidence_percentage}%</strong>
          </div>
        </div>
      </div>

      {/* Refined Image ROI Preview */}
      {refinedPreview && (
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: 12,
          padding: '10px 14px',
          background: '#f1f5f9',
          borderRadius: 10,
          border: '1px solid #cbd5e1',
          marginBottom: 14,
          marginTop: 6,
        }}>
          <img
            src={refinedPreview}
            alt="Refined Ocular ROI"
            style={{
              width: 56,
              height: 56,
              borderRadius: 8,
              objectFit: 'cover',
              border: '2px solid #3b82f6',
              boxShadow: '0 2px 6px rgba(0,0,0,0.1)'
            }}
          />
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: '0.74rem', color: '#1e40af', fontWeight: 700 }}>
              <Scan size={14} /> Refined Ocular Region (ROI)
            </div>
            <div style={{ fontSize: '0.76rem', color: '#475569', marginTop: 2 }}>
              Eye centered · CLAHE contrast equalized · Focal blur extracted
            </div>
          </div>
        </div>
      )}

      {/* Spectacle Probability Meters */}
      <div className="prob-meter-group">
        <div className="prob-label-row">
          <span>Spectacles Needed</span>
          <span>{result.probabilities?.['Spectacles Needed'] ?? (isSpectacleNeeded ? result.confidence_percentage : 100 - result.confidence_percentage)}%</span>
        </div>
        <div className="prob-bar-track">
          <div
            className="prob-bar-fill warning"
            style={{
              width: `${result.probabilities?.['Spectacles Needed'] ?? (isSpectacleNeeded ? result.confidence_percentage : 100 - result.confidence_percentage)}%`,
            }}
          ></div>
        </div>
      </div>

      <div className="prob-meter-group" style={{ marginBottom: 18 }}>
        <div className="prob-label-row">
          <span>No Spectacles Needed</span>
          <span>{result.probabilities?.['No Spectacles Needed'] ?? (!isSpectacleNeeded ? result.confidence_percentage : 100 - result.confidence_percentage)}%</span>
        </div>
        <div className="prob-bar-track">
          <div
            className="prob-bar-fill success"
            style={{
              width: `${result.probabilities?.['No Spectacles Needed'] ?? (!isSpectacleNeeded ? result.confidence_percentage : 100 - result.confidence_percentage)}%`,
            }}
          ></div>
        </div>
      </div>

      {/* Extracted Optical Features Box */}
      {features && (
        <div style={{
          background: '#f8fafc',
          border: '1px solid #e2e8f0',
          borderRadius: 10,
          padding: '12px 14px',
          marginBottom: 16,
        }}>
          <div style={{ fontSize: '0.74rem', fontWeight: 700, color: '#475569', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.06em', display: 'flex', alignItems: 'center', gap: 6 }}>
            <Sliders size={13} /> Extracted Optical Image Features
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, fontSize: '0.76rem' }}>
            <div style={{ background: '#ffffff', padding: '6px 8px', borderRadius: 6, border: '1px solid #e2e8f0' }}>
              <div style={{ color: '#94a3b8', fontSize: '0.70rem' }}>Focal Sharpness</div>
              <div style={{ fontWeight: 700, color: '#0f172a' }}>{features.focal_sharpness_score}</div>
            </div>
            <div style={{ background: '#ffffff', padding: '6px 8px', borderRadius: 6, border: '1px solid #e2e8f0' }}>
              <div style={{ color: '#94a3b8', fontSize: '0.70rem' }}>Meridional Asymmetry</div>
              <div style={{ fontWeight: 700, color: '#0f172a' }}>{features.meridional_asymmetry_index}</div>
            </div>
            <div style={{ background: '#ffffff', padding: '6px 8px', borderRadius: 6, border: '1px solid #e2e8f0' }}>
              <div style={{ color: '#94a3b8', fontSize: '0.70rem' }}>Pupil Contrast Ratio</div>
              <div style={{ fontWeight: 700, color: '#0f172a' }}>{features.pupil_iris_contrast_ratio}</div>
            </div>
            <div style={{ background: '#ffffff', padding: '6px 8px', borderRadius: 6, border: '1px solid #e2e8f0' }}>
              <div style={{ color: '#94a3b8', fontSize: '0.70rem' }}>Reflex Symmetry</div>
              <div style={{ fontWeight: 700, color: '#0f172a' }}>{features.reflex_symmetry_score}%</div>
            </div>
          </div>
        </div>
      )}

      {/* Clinical Guidance Advice Card */}
      <div className="advice-card">
        <div className="advice-title">Screening Assessment & Guidance</div>
        <div className="advice-text">{result.clinical_advice}</div>
      </div>

      {/* Meta info footer */}
      <div className="meta-footer">
        <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          <Cpu size={14} /> ResNet-18 Transfer Learning · Binary Screening
        </span>
        {result.execution_time_ms && (
          <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <Clock size={14} /> {result.execution_time_ms}ms
          </span>
        )}
      </div>
    </div>
  );
}
