"""
frame_analyzer.py
-----------------
Detects and crops out presenter faces from extracted video frames.
Keeps screen share content, removes webcam overlays.
Uses OpenCV Haar Cascade for face detection.
"""

import cv2
import os
import numpy as np
from pathlib import Path


# Load Haar Cascade for face detection (ships with opencv-python)
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)


def detect_faces(frame: np.ndarray) -> list[tuple]:
    """
    Detect faces in a frame using Haar Cascade.
    Returns list of (x, y, w, h) bounding boxes.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )
    return list(faces) if len(faces) > 0 else []


def crop_face_from_frame(frame: np.ndarray, padding: int = 30) -> np.ndarray:
    """
    Detect faces and remove them from the frame.

    Strategy: If a face is detected in a corner region (typical webcam overlay),
    crop that entire corner out. Otherwise, blur the face region.
    """
    h, w = frame.shape[:2]
    faces = detect_faces(frame)

    if not faces:
        return frame

    result = frame.copy()

    for (fx, fy, fw, fh) in faces:
        # Expand the region around the face with padding
        x1 = max(0, fx - padding)
        y1 = max(0, fy - padding)
        x2 = min(w, fx + fw + padding)
        y2 = min(h, fy + fh + padding)

        # Check if face is in a corner (typical webcam overlay position)
        face_center_x = fx + fw // 2
        face_center_y = fy + fh // 2

        is_corner = (
            (face_center_x < w * 0.25 or face_center_x > w * 0.75) and
            (face_center_y < h * 0.25 or face_center_y > h * 0.75)
        )

        if is_corner:
            # Crop out the corner region entirely (fill with surrounding color)
            # Sample the background color from adjacent area
            bg_color = _sample_background(result, x1, y1, x2, y2)
            result[y1:y2, x1:x2] = bg_color
        else:
            # Blur the face region if it's not in a corner
            face_region = result[y1:y2, x1:x2]
            blurred = cv2.GaussianBlur(face_region, (99, 99), 30)
            result[y1:y2, x1:x2] = blurred

    return result


def _sample_background(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """Sample the dominant color around a region to use as fill."""
    h, w = frame.shape[:2]

    # Sample from the nearest edge
    samples = []

    # Sample above the region
    if y1 > 10:
        samples.append(frame[max(0, y1-10):y1, x1:x2])
    # Sample below
    if y2 < h - 10:
        samples.append(frame[y2:min(h, y2+10), x1:x2])
    # Sample left
    if x1 > 10:
        samples.append(frame[y1:y2, max(0, x1-10):x1])
    # Sample right
    if x2 < w - 10:
        samples.append(frame[y1:y2, x2:min(w, x2+10)])

    if samples:
        combined = np.concatenate([s.reshape(-1, 3) for s in samples if s.size > 0])
        if len(combined) > 0:
            return np.median(combined, axis=0).astype(np.uint8)

    return np.array([0, 0, 0], dtype=np.uint8)


def process_all_frames(input_dir: str, output_dir: str) -> list[str]:
    """
    Process all extracted frames: detect and crop faces.
    Returns list of cleaned frame paths.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cleaned_paths = []

    frame_files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    for filename in frame_files:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        frame = cv2.imread(input_path)
        if frame is None:
            continue

        cleaned = crop_face_from_frame(frame)
        cv2.imwrite(output_path, cleaned)
        cleaned_paths.append(output_path)

    return cleaned_paths


def filter_duplicate_frames(frame_paths: list[str], threshold: float = 0.95) -> list[str]:
    """
    Remove near-duplicate frames by comparing histograms.
    Returns filtered list of unique frame paths.
    """
    if not frame_paths:
        return []

    unique = [frame_paths[0]]
    prev_hist = _compute_histogram(frame_paths[0])

    for path in frame_paths[1:]:
        curr_hist = _compute_histogram(path)
        if curr_hist is None:
            continue

        similarity = cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_CORREL)

        if similarity < threshold:
            unique.append(path)
            prev_hist = curr_hist

    return unique


def _compute_histogram(frame_path: str):
    """Compute normalized grayscale histogram for a frame."""
    frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
    if frame is None:
        return None
    hist = cv2.calcHist([frame], [0], None, [256], [0, 256])
    cv2.normalize(hist, hist)
    return hist
