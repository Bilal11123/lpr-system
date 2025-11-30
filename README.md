# License Plate Recognition (LPR) System  
**Real-time & Batch LPR using YOLO11 + PaddleOCR + Deep-SORT**

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-31011/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)](https://hub.docker.com/u/mbilal1446)
[![Streamlit](https://img.shields.io/badge/frontend-Streamlit-red)](http://localhost:8501)

A complete **License Plate Recognition** system supporting:
- Video file upload
- Live RTSP / HTTP / Webcam streaming
- High-accuracy OCR with **PaddleOCR**
- Vehicle tracking using **Deep-SORT**
- Beautiful dashboard with auto-refresh & CSV export

Trained License Plate Detector: [Roboflow Universe Dataset](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/11)

---

## Features

- Upload videos (MP4, AVI, MOV, MKV)
- Live stream from IP cameras / webcam (`0`)
- Confidence-based plate recognition
- Persistent SQLite database
- Real-time dashboard with auto-refresh
- Export results to CSV
- Fully containerized with Docker

---

## Quick Start (Recommended)

### Option 1: Run directly from Docker Hub (No clone needed!)

```bash
mkdir lpr-demo && cd lpr-demo
curl -L https://raw.githubusercontent.com/Bilal11123/lpr-system/main/docker-compose.yml -o docker-compose.yml
docker-compose up
```

Open → [http://localhost:8501](http://localhost:8501)

### Option 2: From source (with custom models)

```bash
git clone https://github.com/Bilal11123/lpr-system.git
cd lpr-system

# Place your models
# → yolo11n.pt (download from Ultralytics)
# → best.pt (your trained plate detector)

# Add SORT tracker
git clone https://github.com/abewley/sort.git backend/sort

# Run
docker-compose up --build
```

UI: [http://localhost:8501](http://localhost:8501)  
API Docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Docker Images (Public)

- Backend: `mbilal1446/lpr-backend:latest` → [Docker Hub](https://hub.docker.com/r/mbilal1446/lpr-backend)
- Frontend: `mbilal1446/lpr-frontend:latest` → [Docker Hub](https://hub.docker.com/r/mbilal1446/lpr-frontend)

Built with **Python 3.10.11-slim** for full compatibility.

---

## Local Development

```bash
# Backend
cd backend
uvicorn app:app --reload --port 8000

# Frontend (new terminal)
cd ui
streamlit run app.py
```

---

## Project Structure

```
lpr-system/
├── backend/           → FastAPI + YOLO + PaddleOCR + SORT
│   └── sort/          → Tracking logic
├── ui/                → Streamlit dashboard
├── uploads/           → Uploaded videos (persisted)
├── license_plates.db  → Detection results
└── docker-compose.yml
```

---

## Tech Stack

| Component         | Technology                      |
|-------------------|---------------------------------|
| Detection         | YOLO11 (Ultralytics)            |
| OCR               | PaddleOCR (high accuracy)       |
| Tracking          | SORT                            |
| Backend           | FastAPI                         |
| Frontend          | Streamlit                       |
| Deployment        | Docker Compose                  |
| Language          | Python 3.10.11                  |

---

**Deployed. Dockerized. Production-ready.**

Feel free to star ⭐ the repo if you like it!

Made with ❤️ by [Muhammad Bilal](https://github.com/Bilal11123)