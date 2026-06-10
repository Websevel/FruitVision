<div align="center">

# 🍎 FruitVision DIP System

**AI-powered fruit classification & ripeness detection using Digital Image Processing**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

*COMP-342L · Project 06 · Spring 2025 · Pak-Austria Fachhochschule, Haripur*

</div>

---

## 📌 Overview

FruitVision is a full-stack fruit analysis system that identifies **fruit type** and **ripeness stage** from a single image using 23 hand-crafted Digital Image Processing (DIP) features — no deep learning required.

| What it detects | How |
|---|---|
| 🍎 Apple · 🍌 Banana · 🥭 Mango · 🍊 Orange · 🍓 Strawberry | Color, texture & morphology |
| Unripe · Ripe · Overripe | Feature-based scoring engine |
| Nutrition, storage tips, botanical info | Knowledge base lookup |

---

## ✨ Features

- **23 DIP features** — HSV/LAB color stats, Sobel edges, Canny density, GLCM texture, morphological top-hat, Otsu thresholding, histogram kurtosis/skewness
- **FastAPI backend** — `/predict` endpoint accepts any image and returns JSON with fruit + stage + confidence scores
- **Streamlit frontend** — 4-tab dashboard: Analysis · Visualization · Knowledge · Feature Table
- **Interactive plots** — Plotly confidence gauges, radar charts, histogram overlays
- **CSV export** — download the full 23-feature report per prediction
- **Zero-config fallback** — runs without `scikit-image` using a numpy fallback for GLCM

---

## 🗂️ Project Structure

```
FruitVision/
├── app.py              # Streamlit frontend (4-tab dashboard)
├── main.py             # FastAPI backend (/health, /predict)
├── requirements.txt    # Python dependencies
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🚀 Quick Start

### 1 — Clone the repo

```bash
git clone https://github.com/<your-username>/FruitVision.git
cd FruitVision
```

### 2 — Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### 4 — Run the Streamlit app (standalone mode)

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

### 5 — (Optional) Run the FastAPI backend separately

```bash
uvicorn main:app --reload --port 8000
```

API docs available at **http://localhost:8000/docs**

---

## 🔌 API Reference

### `GET /health`
```json
{ "status": "ok" }
```

### `POST /predict`
Upload a fruit image (`multipart/form-data`, field name: `file`).

**Response:**
```json
{
  "fruit": "Banana",
  "stage": "Ripe",
  "fruit_confidence": 54.3,
  "stage_confidence": 48.7,
  "fruit_probabilities": { "Apple": 5.2, "Banana": 54.3, "Mango": 22.1, "Orange": 9.8, "Strawberry": 8.6 },
  "stage_probabilities": { "Unripe": 20.1, "Ripe": 48.7, "Overripe": 31.2 }
}
```

**cURL example:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -F "file=@banana.jpg"
```

---

## 🧠 DIP Feature Pipeline

```
Image Input (any size)
       │
       ▼
  Resize → 128×128
       │
   ┌───┴────────────────────────────────────┐
   │  Color Spaces: HSV, LAB, RGB            │
   │  Edge: Sobel, Canny                     │
   │  Texture: GLCM (contrast, homogeneity,  │
   │           energy, correlation)          │
   │  Morphology: Top-hat (spot detection)   │
   │  Segmentation: Otsu threshold           │
   │  Statistics: Histogram kurtosis/skew    │
   └───────────────────────────┬────────────┘
                               │
                     23-Feature Vector
                               │
                      Scoring Engine
                               │
                  Fruit Type + Ripeness Stage
```

### Lab Mapping

| Feature Group | Lab |
|---|---|
| Color (HSV / LAB means) | L02 |
| Color ratios (R/G, R/B, redness, greenness, yellowness) | L03 |
| Histogram statistics (kurtosis, skew, peak, saturation std) | L05 |
| Edge detection (Canny density, Sobel mean/std) | L07 |
| Morphological analysis (top-hat spot density) | L11 |
| Segmentation (Otsu threshold) | L12 |
| GLCM Texture | L05 |

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web frontend |
| `fastapi` + `uvicorn` | REST API backend |
| `opencv-python` | Image processing |
| `numpy` | Numerical operations |
| `scipy` | Statistical features |
| `scikit-image` | GLCM texture (optional, has fallback) |
| `pillow` | Image I/O |
| `plotly` | Interactive charts |
| `matplotlib` | Static plots |
| `pandas` | Feature table & CSV export |

---

## 👨‍💻 Authors

Built for **COMP-342L Digital Image Processing** — Project 06, Spring 2025  
**Pak-Austria Fachhochschule**, Haripur, Pakistan

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.
