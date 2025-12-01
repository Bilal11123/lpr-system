# backend/processor2.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["YOLO_VERBOSE"] = "False"
from ultralytics import YOLO
import cv2
import numpy as np
from sort.sort import Sort
from util import get_car_deep, read_license_plate
from db import upsert_plate
from pathlib import Path
import logging
import time
from deep_sort_realtime.deepsort_tracker import DeepSort

tracker = DeepSort(max_age=30,  # frames to keep track after disappearance
                    n_init=3,    # consecutive detections to confirm track
                    nn_budget=100,
                    override_track_class=None,
                    embedder="mobilenet",  # fast & lightweight
                    half=True,             # use FP16 for speed
                    bgr=True)

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
    coco_model = YOLO('yolo11n.pt')
    license_plate_detector = YOLO('./best_11_2.pt')

    cap = cv2.VideoCapture(str(video_path))
    frame_count = 0
    frame_skip = 5

    logger.info(f"Started processing {VIDEO_SOURCE_NAME}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # if frame_count > 1000:  # Optional limit for testing
        #     break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        # Detect vehicles
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.numpy():
            x1, y1, x2, y2, score, class_id = detection

            if int(class_id) in [2, 3, 5, 7]:  # car, motorcycle, bus, truck
                detections_.append(([x1, y1, x2 - x1, y2 - y1], score, class_id))  # ltrb format: left, top, width, height
                # detections_.append([x1, y1, x2, y2, score])

        # Track vehicles
        tracks = tracker.update_tracks(detections_, frame=frame)  # bgr image for embedder
        track_ids = [t for t in tracks if t.is_confirmed()]
        # track_ids = mot_tracker.update(np.asarray(detections_))

        # Detect license plates
        license_plates = license_plate_detector(frame)[0]

        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # Assign plate to car
            # xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)
            car_object = get_car_deep(license_plate, track_ids)
            if car_object == (-1, -1, -1, -1, -1):
                continue
            
            car_id = car_object.track_id

            if car_id == -1:
                continue

            # Crop and OCR
            license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2)]
            # gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
            # _, thresh = cv2.threshold(gray, 64, 255, cv2.THRESH_BINARY)
            # thresh_3ch = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            crop_rgb = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2RGB)

            license_number, ocr_score = read_license_plate(crop_rgb)
            if not license_number or len(license_number) < 7:
                continue

            logger.info(
                f"Frame {frame_count:04d} | Car ID: {car_id} | Plate: {license_number} | Score: {ocr_score:.3f}"
            )
            print(f"Frame {frame_count:04d} | Car ID: {car_id} | Plate: {license_number} | Score: {ocr_score:.3f}")

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

    logger = logging.getLogger(__name__)
    logger.info(f"Opening stream: {url}")

    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        logger.error(f"Cannot open stream {url}")
        return

    # Same models & tracker
    coco_model = YOLO('yolo11n.pt')
    license_plate_detector = YOLO('./best_11_2.pt')
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

        # vehicle detection & tracking (identical to process_video)
        detections = coco_model(frame)[0]
        detections_ = []
        for d in detections.boxes.data.numpy():
            x1, y1, x2, y2, score, class_id = d
            if int(class_id) in [2, 3, 5, 7]:
                detections_.append([x1, y1, x2, y2, score])

        # track_ids = mot_tracker.update(np.asarray(detections_))
        tracks = tracker.update_tracks(detections_, frame=frame)  # bgr image for embedder
        track_ids = [t for t in tracks if t.is_confirmed()]

        # plate detection & OCR
        license_plates = license_plate_detector(frame)[0]
        for lp in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, _ = lp
            car_object = get_car_deep(lp, track_ids)
            if car_object == (-1, -1, -1, -1, -1):
                continue
            
            car_id = car_object.track_id
            if car_id == -1:
                continue

            license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2)]
            # gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
            # _, thresh = cv2.threshold(gray, 64, 255, cv2.THRESH_BINARY)
            # thresh_3ch = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            crop_rgb = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2RGB)
            # _, thresh = cv2.threshold(gray, 64, 255, cv2.THRESH_BINARY_INV)

            text, ocr_score = read_license_plate(crop_rgb)
            if not text:
                continue

            logger.info(f"Stream [{source_name}] Car {car_id} → {text} ({ocr_score:.3f})")
            upsert_plate(car_id=int(car_id), license_number=text,
                        score=ocr_score, video_source=source_name)

    cap.release()
    logger.info(f"Stream {source_name} finished")

def process_video4(video_path: str, app):
    """
    Process a single video file and store results in DB.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    VIDEO_SOURCE_NAME = video_path.name

    # Load models
    coco_model = YOLO('yolo11n.pt')
    license_plate_detector = YOLO('./best_11_2.pt')

    cap = cv2.VideoCapture(str(video_path))
    frame_count = 0
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    height_threshold = frame_height * 0.55  # 50% of frame height
    

    logger.info(f"Started processing {VIDEO_SOURCE_NAME}")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Detect vehicles
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.numpy():
            x1, y1, x2, y2, score, class_id = detection

            if int(class_id) in [2, 3, 5, 7]:  # car, motorcycle, bus, truck
                detections_.append(([x1, y1, x2 - x1, y2 - y1], score, class_id))

        # Track vehicles
        tracks = tracker.update_tracks(detections_, frame=frame)
        track_ids = [t for t in tracks if t.is_confirmed()]

        # Detect license plates
        license_plates = license_plate_detector(frame)[0]

        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # Assign plate to car
            car_object = get_car_deep(license_plate, track_ids)
            if car_object == (-1, -1, -1, -1, -1):
                continue
            
            car_id = car_object.track_id

            if car_id == -1:
                continue
            
            # Get car bounding box from the track object
            car_bbox = car_object.to_ltrb()  # left, top, right, bottom
            # car_y_center = (car_bbox[1] + car_bbox[3]) / 2
            
            # Only process if vehicle bottom has passed 55% of frame height
            if car_bbox[3] < height_threshold:
                continue

            # Draw license plate bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # Crop and OCR
            license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2)]
            crop_rgb = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2RGB)

            license_number, ocr_score = read_license_plate(crop_rgb)
            if not license_number or len(license_number) < 7:
                continue

            logger.info(
                f"Frame {frame_count:04d} | Car ID: {car_id} | Plate: {license_number} | Score: {ocr_score:.3f}"
            )
            print(f"Frame {frame_count:04d} | Car ID: {car_id} | Plate: {license_number} | Score: {ocr_score:.3f}")

            # Draw text background for better visibility
            text = f"ID:{car_id} {license_number} {ocr_score:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 2
            
            # Get text size for background rectangle
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            
            # Position text above the license plate box
            text_x = int(x1)
            text_y = int(y1) - 10
            
            # Ensure text doesn't go off screen
            if text_y < text_height + 10:
                text_y = int(y2) + text_height + 10
            
            # Draw background rectangle
            cv2.rectangle(frame, 
                            (text_x, text_y - text_height - 5), 
                            (text_x + text_width + 5, text_y + baseline), 
                            (0, 255, 0), 
                            -1)
            
            # Draw text
            cv2.putText(frame, text, (text_x, text_y), 
                        font, font_scale, (0, 0, 0), thickness)

            # Upsert into DB (only once per car or when better score found)
            upsert_plate(
                car_id=int(car_id),
                license_number=license_number,
                score=ocr_score,
                video_source=VIDEO_SOURCE_NAME,
            )

        cv2.line(frame, (0, int(height_threshold)), (frame_width, int(height_threshold)), (0, 0, 255), 4, lineType=cv2.LINE_8, shift=0)
        # resized_img_fixed = cv2.resize(frame, (1080, 600))
    
    cap.release()
    logger.info(f"Finished processing {VIDEO_SOURCE_NAME}")