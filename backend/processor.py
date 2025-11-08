# backend/processor.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from ultralytics import YOLO
import cv2
import numpy as np
from sort.sort import Sort
from util import get_car, read_license_plate
from db import upsert_plate
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_video(video_path: str):
    """
    Process a single video file and store results in DB.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    VIDEO_SOURCE_NAME = video_path.name
    mot_tracker = Sort()
    seen_car_ids = set()

    # Load models
    coco_model = YOLO('yolov8n.pt')
    license_plate_detector = YOLO('./best.pt')

    cap = cv2.VideoCapture(str(video_path))
    frame_count = 0
    frame_skip = 5

    logger.info(f"Started processing {VIDEO_SOURCE_NAME}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count > 1000:  # Optional limit for testing
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        # Detect vehicles
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in [2, 3, 5, 7]:  # car, motorcycle, bus, truck
                detections_.append([x1, y1, x2, y2, score])

        # Track vehicles
        track_ids = mot_tracker.update(np.asarray(detections_))

        # Detect license plates
        license_plates = license_plate_detector(frame)[0]

        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # Assign plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            if car_id == -1:
                continue

            # Crop and OCR
            license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2)]
            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
            _, license_plate_crop_thresh = cv2.threshold(
                license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV
            )

            license_number, ocr_score = read_license_plate(license_plate_crop_thresh)
            if not license_number or len(license_number) < 4:
                continue

            logger.info(
                f"Frame {frame_count:04d} | Car ID: {car_id} | Plate: {license_number} | Score: {ocr_score:.3f}"
            )

            # Upsert into DB
            upsert_plate(
                car_id=int(car_id),
                license_number=license_number,
                score=ocr_score,
                video_source=VIDEO_SOURCE_NAME,
            )

    cap.release()
    logger.info(f"Finished processing {VIDEO_SOURCE_NAME}")

def process_stream(url: str, source_name: str):
    """
    Same pipeline as process_video but reads from a live URL.
    Stops after ~5 minutes of no new frames (timeout) or when user aborts.
    """
    import cv2
    from ultralytics import YOLO
    import numpy as np
    from sort.sort import Sort
    from util import get_car, read_license_plate
    from db import upsert_plate
    import logging, time

    logger = logging.getLogger(__name__)
    logger.info(f"Opening stream: {url}")

    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        logger.error(f"Cannot open stream {url}")
        return

    # Same models & tracker
    coco_model = YOLO('yolov8n.pt')
    license_plate_detector = YOLO('./best.pt')
    mot_tracker = Sort()

    frame_count = 0
    frame_skip = 5
    start_time = time.time()
    MAX_DURATION = 300  # 5 min safety net

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            logger.warning("No frame – retrying...")
            time.sleep(0.5)
            continue

        if time.time() - start_time > MAX_DURATION:
            logger.info("Stream timeout – stopping")
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        # ---- vehicle detection & tracking (identical to process_video) ----
        detections = coco_model(frame)[0]
        detections_ = []
        for d in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = d
            if int(class_id) in [2, 3, 5, 7]:
                detections_.append([x1, y1, x2, y2, score])

        track_ids = mot_tracker.update(np.asarray(detections_))

        # ---- plate detection & OCR ----
        license_plates = license_plate_detector(frame)[0]
        for lp in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, _ = lp
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(lp, track_ids)
            if car_id == -1:
                continue

            crop = frame[int(y1):int(y2), int(x1):int(x2)]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 64, 255, cv2.THRESH_BINARY_INV)

            text, ocr_score = read_license_plate(thresh)
            if not text:
                continue

            logger.info(f"Stream [{source_name}] Car {car_id} → {text} ({ocr_score:.3f})")
            upsert_plate(car_id=int(car_id), license_number=text,
                        score=ocr_score, video_source=source_name)

    cap.release()
    logger.info(f"Stream {source_name} finished")