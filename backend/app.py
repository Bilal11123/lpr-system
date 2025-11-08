# backend/app.py
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pathlib import Path
import shutil
import sqlite3
from db import DB_PATH
from processor import process_video, process_stream
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="License Plate Recognition API")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

class ProcessResponse(BaseModel):
    message: str
    video_source: str

class StreamRequest(BaseModel):
    url: str
    name: str | None = None   # optional friendly name

class PlateRecord(BaseModel):
    car_id: int
    license_number: str
    score: float
    timestamp: str
    video_source: str

def get_plates_from_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM plates ORDER BY timestamp DESC")
    rows = cur.fetchall()
    conn.close()
    return [dict(row) for row in rows]

@app.post("/process-video/", response_model=ProcessResponse)
async def process_video_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Upload a video and start processing in the background.
    """
    if not file.filename.endswith(('.mp4', '.avi', '.mov')):
        raise HTTPException(status_code=400, detail="Invalid video format")

    video_path = UPLOAD_DIR / file.filename
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run processing in background
    background_tasks.add_task(process_video, str(video_path))

    return ProcessResponse(
        message="Video uploaded and processing started.",
        video_source=file.filename
    )

# @app.post("/process-path/")
# async def process_video_path(
#     background_tasks: BackgroundTasks,
#     video_path: str
# ):
#     """
#     Process a video from a local file path (for internal/testing use).
#     """
#     path = Path(video_path)
#     if not path.exists():
#         raise HTTPException(status_code=404, detail="Video file not found")

#     background_tasks.add_task(process_video, str(path))
#     return {"message": "Processing started", "video_source": path.name}

@app.post("/process-stream/")
async def process_stream_endpoint(
    background_tasks: BackgroundTasks,
    req: StreamRequest
):
    # simple sanity check – OpenCV will raise later if invalid
    if not req.url.lower().startswith(('http', 'rtsp', 'rtmp')):
        raise HTTPException(400, "URL must start with http(s)://, rtsp:// or rtmp://")
    name = req.name or req.url.split("/")[-1] or "stream"
    background_tasks.add_task(process_stream, req.url, name)
    return {"message": "Stream processing started", "video_source": name}

@app.get("/plates/", response_model=list[PlateRecord])
async def get_plates():
    """
    Retrieve all detected license plates.
    """
    plates = get_plates_from_db()
    return plates

@app.get("/health")
async def health_check():
    return {"status": "healthy"}