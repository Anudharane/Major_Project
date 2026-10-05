import React, { useRef, useState } from 'react';
import { UploadCloud, Image as ImageIcon, X } from 'lucide-react';

export default function ImageUploader({ selectedImage, onImageSelected, onClearImage }) {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file && file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = () => {
        onImageSelected(reader.result, file);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = () => {
        onImageSelected(reader.result, file);
      };
      reader.readAsDataURL(file);
    }
  };

  return (
    <div>
      {!selectedImage ? (
        <div
          className={`upload-dropzone ${isDragging ? 'dragging' : ''}`}
          onClick={() => fileInputRef.current?.click()}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            accept="image/*"
            style={{ display: 'none' }}
          />
          <div className="upload-icon">
            <UploadCloud size={24} />
          </div>
          <div>
            <div className="upload-text">Click or drag & drop eye photograph</div>
            <div className="upload-hint">Supports JPEG, PNG, WEBP (Corneal / Anterior Segment view)</div>
          </div>
        </div>
      ) : (
        <div className="preview-container">
          <img src={selectedImage} alt="Selected Eye" className="preview-img" />
          <button className="remove-btn" onClick={onClearImage} title="Remove image">
            <X size={18} />
          </button>
        </div>
      )}
    </div>
  );
}
