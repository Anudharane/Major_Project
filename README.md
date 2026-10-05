# AI-Based Vision Screening of Refractive Errors Using Deep Learning

**Final Year Engineering Project (50% Milestone Prototype)**

---

## 📌 Project Overview
This project presents an intelligent screening prototype that utilizes **Convolutional Neural Networks (CNN)** and **Transfer Learning (MobileNetV2)** to analyze ocular images captured via webcam or photo upload and determine **whether an individual requires corrective spectacles or demonstrates normal vision**.

### Key Objective:
- Screen eye images (corneal contours, reflections, pupil margin clarity).
- **Direct Output**:
  - `Spectacles Required: YES` (Refractive Anomaly Detected)
  - `No Spectacles Needed: NO` (Normal Vision / Emmetropic Eye)
  - Confidence Percentage (e.g., $94.5\%$)
  - Clinical Advice & Probability Breakdown

---

## 🏗️ Project Architecture & Tech Stack

```
Project/
├── backend/
│   ├── app.py                     # FastAPI REST API (inference & health endpoints)
│   ├── model.py                   # Transfer Learning (MobileNetV2) & Custom CNN definitions
│   ├── train.py                   # Model training script with checkpoints & early stopping
│   ├── generate_sample_data.py   # Synthetic ocular dataset generator & folder structure initializer
│   ├── requirements.txt           # Python dependencies (FastAPI, TensorFlow, Pillow, etc.)
│   ├── dataset/                   # Dataset directory for training
│   │   ├── train/
│   │   │   ├── no_spectacles/
│   │   │   └── spectacles_needed/
│   │   └── val/
│   │       ├── no_spectacles/
│   │       └── spectacles_needed/
│   └── saved_models/              # Saved model weights (.keras)
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Header.jsx         # Navigation bar & system status
│   │   │   ├── ImageUploader.jsx  # Drag & drop photo upload
│   │   │   ├── WebcamCapture.jsx  # Live webcam stream & snapshot capture
│   │   │   ├── ResultCard.jsx     # Spectacle requirement verdict & probability bars
│   │   │   └── InfoGuide.jsx      # System architecture & vision screening guide
│   │   ├── App.jsx                # Main application state & API integration
│   │   ├── App.css                # Medical dashboard styling
│   │   ├── main.jsx               # React entry point
│   │   └── index.css              # Global design tokens
│   ├── index.html
│   ├── package.json               # Frontend dependencies
│   └── vite.config.js             # Vite dev server configuration & API proxy
└── README.md
```

- **Frontend**: React.js (Vite), Lucide Icons, Modern Responsive CSS
- **Backend**: Python, FastAPI, Uvicorn, Pillow, NumPy
- **Deep Learning Core**: Transfer Learning using **MobileNetV2** (pre-trained on ImageNet) & Custom CNN Baseline (TensorFlow / Keras)

---

## 🚀 Step-by-Step Execution Guide

### 1. Backend Setup & Running

Open a terminal in the `backend/` folder:

```bash
cd backend

# 1. (Optional) Create and activate a virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate initial sample dataset (creates dataset folders and sample images)
python generate_sample_data.py

# 4. Train the Transfer Learning Model
python train.py --epochs 10 --batch_size 16 --model_type transfer_learning

# 5. Start the FastAPI backend server
python app.py
```
> The backend server will be live at: **`http://localhost:8000`**  
> Interactive API Docs (Swagger UI): **`http://localhost:8000/docs`**

---

### 2. Frontend Setup & Running

Open a second terminal in the `frontend/` folder:

```bash
cd frontend

# 1. Install Node.js dependencies
npm install

# 2. Launch the React development server
npm run dev
```
> Open your browser at: **`http://localhost:5173`**

---

## 🧪 How to Add Your Real Dataset Later

When you obtain your real clinical or Kaggle eye dataset (corneal, anterior segment, or fundus photographs):

1. Navigate to `backend/dataset/`:
   - Place images where spectacles are **not required** (normal eyes) into:
     `backend/dataset/train/no_spectacles/` and `backend/dataset/val/no_spectacles/`
   - Place images where spectacles are **required** (myopia, hyperopia, astigmatism) into:
     `backend/dataset/train/spectacles_needed/` and `backend/dataset/val/spectacles_needed/`
2. Retrain the model:
   ```bash
   python train.py --epochs 25 --batch_size 16 --model_type transfer_learning
   ```
3. The newly trained model will automatically save to `backend/saved_models/spectacle_screening_model.keras` and be immediately served by the API.

---

## 📊 50% Project Milestone Status

| Component | Status | Description |
| :--- | :---: | :--- |
| **Project Structure** | ✅ Done | Frontend + Backend separate modular folders |
| **Spectacle Need Determination** | ✅ Done | Primary binary screening decision output |
| **Transfer Learning CNN** | ✅ Done | MobileNetV2 feature extractor with custom dense head |
| **Dataset Structure & Samples** | ✅ Done | Formatted `train/` and `val/` folders + sample generator |
| **Backend REST API** | ✅ Done | FastAPI `/predict` supporting file upload and webcam stream |
| **React UI Dashboard** | ✅ Done | Live Webcam capture, upload dropzone, confidence bars |
| **Real Dataset Training** | ⏳ Remaining | Train on larger clinical / ophthalmology dataset |
| **Hyperparameter Optimization** | ⏳ Remaining | Learning rate tuning, Grad-CAM attention heatmap |
