import React from 'react';
import { CheckCircle2, AlertOctagon, Glasses, Sparkles, Clock, Shield, Eye } from 'lucide-react';

const CONDITION_ICONS = {
  "Normal Ocular Vision (Emmetropic)": "👁️",
  "Myopia (Nearsightedness)": "🔍",
  "Hyperopia (Farsightedness)": "🔭",
  "Astigmatism (Corneal Asymmetry)": "🌀",
};

const CONDITION_COLORS = {
  "Normal Ocular Vision (Emmetropic)": "#22c55e",
  "Myopia (Nearsightedness)": "#f97316",
  "Hyperopia (Farsightedness)": "#8b5cf6",
  "Astigmatism (Corneal Asymmetry)": "#ec4899",
};

export default function ResultCard({ result, loading }) {
  if (loading) {
    return (
      <div className="empty-placeholder">
        <div className="brand-icon" style={{ animation: 'spin 1.5s linear infinite' }}>
          <Sparkles size={24} />
        </div>
        <div style={{ fontWeight: 700, fontSize: '1rem', marginTop: 12 }}>
          Screening Ocular Features...
        </div>
        <div style={{ fontSize: '0.82rem', color: '#64748b' }}>
          AI Vision Model is analyzing your eye image...
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
        <div style={{ fontWeight: 600, color: '#475569' }}>No Analysis Performed Yet</div>
        <div style={{ fontSize: '0.82rem', maxWidth: 320 }}>
          Upload an eye photograph or use the webcam to screen for spectacle requirement.
        </div>
      </div>
    );
  }

  const isSpectacleNeeded = result.spectacles_needed;
  const badgeClass = isSpectacleNeeded ? 'warning' : 'success';
  const detectedCondition = result.detected_condition || (isSpectacleNeeded ? 'Refractive Error Detected' : 'Normal Ocular Vision (Emmetropic)');
  const conditionIcon = CONDITION_ICONS[detectedCondition] || '👁️';
  const conditionColor = CONDITION_COLORS[detectedCondition] || '#64748b';
  const conditionBreakdown = result.condition_breakdown || null;

  return (
    <div>
      {/* Primary Result Banner */}
      <div className={`result-banner ${badgeClass}`}>
        <div className={`banner-icon-box ${badgeClass}`}>
          {isSpectacleNeeded ? '👓' : '✅'}
        </div>
        <div>
          <div className="banner-heading">
            {isSpectacleNeeded ? 'Spectacles Required' : 'No Spectacles Needed'}
          </div>
          <div className="banner-confidence">
            AI Confidence: <strong>{result.confidence_percentage}%</strong>
          </div>
        </div>
      </div>

      {/* Detected Condition Badge */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: 10,
        padding: '10px 14px',
        background: isSpectacleNeeded ? 'rgba(249,115,22,0.08)' : 'rgba(34,197,94,0.08)',
        border: `1px solid ${conditionColor}30`,
        borderRadius: 10,
        marginBottom: 16,
        marginTop: 4,
      }}>
        <span style={{ fontSize: '1.4rem' }}>{conditionIcon}</span>
        <div>
          <div style={{ fontSize: '0.72rem', color: '#94a3b8', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
            {isSpectacleNeeded ? 'Detected Condition' : 'Eye Status'}
          </div>
          <div style={{ fontWeight: 700, color: conditionColor, fontSize: '0.95rem' }}>
            {detectedCondition}
          </div>
        </div>
      </div>

      {/* Model Probabilities Breakdown */}
      <div className="prob-meter-group">
        <div className="prob-label-row">
          <span>Spectacles Required</span>
          <span>{result.probabilities?.['Spectacles Required'] ?? (isSpectacleNeeded ? result.confidence_percentage : 100 - result.confidence_percentage)}%</span>
        </div>
        <div className="prob-bar-track">
          <div
            className="prob-bar-fill warning"
            style={{
              width: `${result.probabilities?.['Spectacles Required'] ?? (isSpectacleNeeded ? result.confidence_percentage : 100 - result.confidence_percentage)}%`,
            }}
          ></div>
        </div>
      </div>

      <div className="prob-meter-group" style={{ marginBottom: conditionBreakdown ? 16 : 24 }}>
        <div className="prob-label-row">
          <span>No Spectacles Needed (Normal)</span>
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

      {/* Condition Breakdown */}
      {conditionBreakdown && (
        <div style={{
          background: '#f8fafc',
          border: '1px solid #e2e8f0',
          borderRadius: 10,
          padding: '12px 14px',
          marginBottom: 16,
        }}>
          <div style={{ fontSize: '0.75rem', fontWeight: 700, color: '#64748b', marginBottom: 10, textTransform: 'uppercase', letterSpacing: '0.06em', display: 'flex', alignItems: 'center', gap: 6 }}>
            <Eye size={13} /> Condition Probability Breakdown
          </div>
          {Object.entries(conditionBreakdown).map(([cond, prob]) => {
            const isDetected = detectedCondition.toLowerCase().includes(cond.toLowerCase().split(' ')[0]);
            return (
              <div key={cond} style={{ marginBottom: 7 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem', marginBottom: 3 }}>
                  <span style={{ color: isDetected ? '#1e293b' : '#64748b', fontWeight: isDetected ? 700 : 400 }}>
                    {cond === 'Normal Vision' ? '✓ ' : isDetected ? '⚠ ' : ''}{cond}
                  </span>
                  <span style={{ fontWeight: 600, color: isDetected ? conditionColor : '#94a3b8' }}>{prob}%</span>
                </div>
                <div style={{ height: 5, borderRadius: 4, background: '#e2e8f0', overflow: 'hidden' }}>
                  <div style={{
                    height: '100%',
                    borderRadius: 4,
                    width: `${prob}%`,
                    background: isDetected ? conditionColor : '#cbd5e1',
                    transition: 'width 0.6s ease',
                  }}></div>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Clinical Guidance Advice Card */}
      <div className="advice-card">
        <div className="advice-title">Screening Summary & Advice</div>
        <div className="advice-text">{result.clinical_advice}</div>
      </div>

      {/* Meta info footer */}
      <div className="meta-footer">
        <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          <Shield size={14} /> SpectacleVisionNet CNN · 96.7% Accuracy
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
