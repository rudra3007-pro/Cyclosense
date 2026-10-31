# 🌀 CycloSense

> **AI/ML-Based Tropical Cyclone Identification, Classification & Prediction System**
> SIH 2026 · Version 2.0

CycloSense is an end-to-end cyclone intelligence platform that ingests satellite imagery, detects and classifies cyclones, forecasts future tracks and intensity, and presents everything through an interactive GIS dashboard — with decision-support tools for disaster managers, port operators, and coastal communities.

---

## 📸 Pipeline Overview

```
Satellite Image
      │
      ▼
Model 1 — YOLOv8-nano          → Cyclone detected? + Bounding Box
      │
      ▼
Model 2 — EfficientNet-B0      → IMD Category (D / DD / CS / SCS / VSCS)
      │
      ▼
Model 3 — 2-layer LSTM         → Future track + intensity (+3h / +6h / +12h / +24h)
      │
      ▼
Flask API  ←→  GeoJSON Map Routes  ←→  Frontend Dashboard (Leaflet + Plotly)
```

---

## 🗂️ Project Structure

```
CycloSense/
│
├── backend/                         # Flask API server
│   ├── run.py                       # Entry point — python run.py
│   ├── requirements.txt
│   ├── weights/                     # Pre-trained model weights (tracked in Git)
│   │   ├── model1_detection.pt      # YOLOv8-nano (~6 MB)
│   │   ├── model2_classification.pt # EfficientNet-B0 (~16 MB)
│   │   ├── model3_forecasting.pt    # 2-layer LSTM (~200 KB)
│   │   └── model3_scaler.pkl        # sklearn StandardScaler
│   ├── app/
│   │   ├── __init__.py              # Flask app factory (create_app)
│   │   ├── routes/
│   │   │   ├── predict.py           # /api/predict/* — all inference routes
│   │   │   └── map.py               # /api/map/*    — GeoJSON routes
│   │   ├── services/
│   │   │   └── model_loader.py      # Loads all 3 models at startup
│   │   └── utils/
│   │       └── image_utils.py       # Image preprocessing helpers
│   └── tests/
│       └── test_api.py              # Smoke tests (all passing ✅)
│
├── frontend/                        # Dashboard UI
│   ├── pages/index.html
│   ├── css/style.css
│   └── js/main.js
│
├── ml/                              # ML training pipeline
│   ├── data/raw/                    # Raw satellite datasets (not tracked)
│   ├── data/processed/              # Preprocessed datasets (not tracked)
│   ├── models/                      # Saved model checkpoints per experiment
│   ├── src/
│   │   ├── preprocessing/           # Data preprocessing scripts
│   │   ├── training/                # Model training scripts
│   │   └── evaluation/              # Evaluation & metrics scripts
│   └── notebooks/                   # Jupyter notebooks for experimentation
│
├── outputs/
│   ├── logs/                        # Runtime prediction logs
│   └── predictions/                 # Saved prediction results
│
├── reports/figures/                 # Charts and evaluation figures
├── docs/                            # Documentation
├── .gitignore
└── README.md
```

---

## 🤖 Models

| Model | Architecture | Dataset | Key Metric |
|---|---|---|---|
| **Detection** | YOLOv8-nano (COCO pretrained) | INCYDE — 16,038 INSAT images | mAP@50 = **0.699**, Recall = **0.801** |
| **Classification** | EfficientNet-B0 (ImageNet pretrained) | INCYDE — 103,368 cropped images | Accuracy **46%**, VSCS F1 = **0.71** |
| **Forecasting** | 2-layer LSTM + StandardScaler | IBTrACS NIO 2009–2023 — 142 storms | +3h error = **101 km**, +24h = **391 km** |

**IMD Intensity Classes:** `D` → `DD` → `CS` → `SCS` → `VSCS`

> ⚠️ CycloSense is a research/decision-support prototype. It is **not** an official meteorological warning service.

---

## 🚀 Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/your-org/CycloSense.git
cd CycloSense
```

### 2. Set up the backend

```bash
cd backend
pip install -r requirements.txt
python run.py
```

The API starts at **`http://localhost:5000`**

### 3. Open the frontend

Open `frontend/pages/index.html` in your browser (or serve it with any static file server).

---

## 🌐 API Reference

### Base URL: `http://localhost:5000`

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Service discovery — lists all endpoints |
| `GET` | `/api/predict/health` | Liveness check — confirms models are loaded |
| `POST` | `/api/predict/detect` | **Model 1** — Cyclone detection from satellite image |
| `POST` | `/api/predict/classify` | **Model 2** — Intensity classification from cropped image |
| `POST` | `/api/predict/forecast` | **Model 3** — Track & intensity forecast from 8-point history |
| `POST` | `/api/predict/full` | **Full pipeline** — image + track → detect + classify + forecast |
| `POST` | `/api/map/track` | Convert forecast to GeoJSON track for Leaflet |
| `POST` | `/api/map/impact` | Generate impact-zone GeoJSON polygons |
| `GET` | `/api/map/basins` | Indian Ocean domain bounds for map initialisation |

### Example — Forecast

```bash
curl -X POST http://localhost:5000/api/predict/forecast \
  -H "Content-Type: application/json" \
  -d '{
    "track": [
      {"lat": 10.2, "lon": 80.1, "wind_speed": 65},
      {"lat": 10.5, "lon": 80.5, "wind_speed": 68},
      {"lat": 10.8, "lon": 80.9, "wind_speed": 71},
      {"lat": 11.1, "lon": 81.3, "wind_speed": 74},
      {"lat": 11.4, "lon": 81.7, "wind_speed": 77},
      {"lat": 11.7, "lon": 82.1, "wind_speed": 80},
      {"lat": 12.0, "lon": 82.5, "wind_speed": 83},
      {"lat": 12.3, "lon": 82.9, "wind_speed": 86}
    ]
  }'
```

**Response:**
```json
{
  "forecast": [
    {"step": 1, "hours": 3,  "lat": 12.44, "lon": 83.32, "wind_speed": 88.1},
    {"step": 2, "hours": 6,  "lat": 12.61, "lon": 83.71, "wind_speed": 90.4},
    {"step": 4, "hours": 12, "lat": 12.94, "lon": 84.38, "wind_speed": 93.2},
    {"step": 8, "hours": 24, "lat": 13.52, "lon": 85.61, "wind_speed": 96.8}
  ],
  "all_steps": [ ... ]
}
```

### Example — Detect (image upload)

```bash
curl -X POST http://localhost:5000/api/predict/detect \
  -F "image=@satellite_image.jpg"
```

---

## 🧪 Testing & Verification

### 🚀 Universal Test Suite (Run Everything At Once)

To run **all 35 tests** across Environment, Weights, API Routes, Live Inference, and Model Evaluation Metrics in one single command:

```bash
python test_all.py
```

*(On Windows Command Prompt or PowerShell, you can also simply run `test_all` or `test_all.bat`)*

---

### 1. Backend API Tests

```bash
cd backend
python tests/test_api.py
```

All 9 smoke tests should pass.

### 2. Model Evaluation & Benchmark Scorecard

To inspect model evaluation metrics, test scores, per-class classification breakdowns, and trajectory errors:

```bash
# Full scorecard report for all 3 models
python evaluate.py

# Include live forward-pass latency benchmarks on loaded weights
python evaluate.py --live

# Inspect a specific model (1 = YOLO, 2 = EfficientNet, 3 = LSTM)
python evaluate.py --model 1
python evaluate.py --model 2
python evaluate.py --model 3

# Export results as JSON
python evaluate.py --json
python evaluate.py --export evaluation_results.json
```

On Windows, you can also run directly:
```cmd
evaluate.bat
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **AI / ML** | Python, PyTorch, YOLOv8 (Ultralytics), EfficientNet-B0, LSTM, scikit-learn |
| **Backend** | Flask, Flask-CORS |
| **GIS** | GeoJSON, Leaflet.js |
| **Frontend** | HTML, CSS, JavaScript, Leaflet, Plotly |
| **Data** | INCYDE (INSAT-3D/3DR), IBTrACS (North Indian Ocean) |

---

## 👥 Team

| Member | Role |
|---|---|
| 1 — Data Engineer | Satellite datasets, preprocessing, synchronisation |
| 2 — Detection | YOLOv8, bounding boxes, localisation |
| 3 — Classification | EfficientNet, Grad-CAM, intensity estimation |
| 4 — Forecasting | LSTM, track/intensity prediction, uncertainty |
| 5 — GIS / Frontend | Leaflet, Plotly, impact layers, scenario UI |
| 6 — Backend / DevOps | Flask API, database, Docker, integration |

---

## 📋 Roadmap

- [x] Model 1 — Cyclone Detection (YOLOv8-nano)
- [x] Model 2 — Intensity Classification (EfficientNet-B0)
- [x] Model 3 — Track & Intensity Forecasting (LSTM)
- [x] Flask API with all prediction routes
- [x] GeoJSON map routes for Leaflet
- [ ] Frontend Dashboard (Leaflet + Plotly)
- [ ] Grad-CAM explainability overlay
- [ ] Uncertainty cone visualisation
- [ ] Cyclone Impact Forecasting engine
- [ ] AI Action Planner / Decision Assistant
- [ ] What-If / Scenario Simulator
- [ ] Storm DNA historical similarity search
- [ ] PostgreSQL + PostGIS database integration
- [ ] Docker Compose deployment

---

## ⚠️ Limitations

- Classification accuracy is 46% overall — VSCS performs best (F1=0.71); DD is weakest (F1=0.16)
- Track forecast error grows with horizon: ~130 km at +3h, ~391 km at +24h
- Model 3 requires an **existing tracked storm** — cannot predict genesis
- Trained only on North Indian Ocean data (INCYDE + IBTrACS NIO)
- Not suitable as a replacement for IMD / NHC / JTWC operational forecasts

---

## 📄 License

This project is developed for academic/research purposes under SIH 2026.
