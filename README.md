# License Plate Recognition (LPR) System

Real-time and batch license plate detection & recognition using:

- **YOLOv8** – Vehicle & plate detection  
- **EasyOCR** – OCR  
- **SORT** – Tracking  
- **FastAPI** – Backend  
- **Streamlit** – Dashboard  
- **Docker Compose** – Deployment  

Dataset used to train Lincese Plate Detection Model at [link](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/11).

---

## Features

- Upload video (MP4, AVI, etc.)
- Live RTSP / HTTP / Webcam streaming
- Confidence scoring
- SQLite persistence
- Auto-refresh UI
- CSV export
- GPU support (optional)

---

## Quick Start (Docker)

```bash
git clone https://github.com/Bilal11123/lpr-system.git
cd lpr-system

# Add models
mkdir -p models
# → yolov8n.pt
# → best.pt (your trained plate detector)

# Add SORT
git clone https://github.com/abewley/sort.git
# or copy sort/sort.py → project/sort/

# Run
docker-compose up --build
```

UI: [http://localhost:8501](http://localhost:8501)  
API: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Local Development (Python 3.10.11)

```bash
# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r backend/requirements.txt
pip install -r ui/requirements.txt

# Run backend
cd backend
uvicorn app:app --reload --port 8000

# Run frontend (new terminal)
cd ../ui
streamlit run app.py
```

---

## Docker Images (Python 3.10.11)

Both services use:

```dockerfile
FROM python:3.10.11-slim
```

Ensures **exact version compatibility**.

---

## Project Structure

```
project/
├── backend/           → FastAPI + processor
|   ├── sort/          → SORT tracker
├── ui/                → Streamlit UI
├── uploads/           → Uploaded videos
├── license_plates.db  → Results
└── docker-compose.yml
```