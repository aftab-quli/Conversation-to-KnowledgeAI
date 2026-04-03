"""
video_processor.py
------------------
Extracts audio and screenshots from uploaded MP4 videos.
Uses ffmpeg for audio extraction and OpenCV for scene detection + frame capture.
"""

import subprocess
import os
import cv2
import numpy as np
from pathlib import Path


def extract_audio(video_path: str, output_audio_path: str) -> bool:
    """Extract audio from MP4 as WAV using ffmpeg."""
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vn",                    # no video
        "-acodec", "pcm_s16le",   # PCM 16-bit
        "-ar", "16000",           # 16kHz (Whisper optimal)
        "-ac", "1",               # mono
        "-y",                     # overwrite
        output_audio_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed: {result.stderr}")
    return True


def detect_scene_changes(video_path: str, max_frames: int = 15, min_interval: float = 3.0) -> list[float]:
    """
    Detect scene changes in video using histogram comparison.
    Returns list of timestamps (seconds) where significant visual changes occur.
    Ensures minimum interval between frames to avoid duplicates.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    # Sample every 0.5 seconds instead of every frame (much faster)
    sample_interval = int(fps * 0.5) if fps > 0 else 15
    if sample_interval < 1:
        sample_interval = 1

    prev_hist = None
    diffs = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            cv2.normalize(hist, hist)

            if prev_hist is not None:
                diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
                timestamp = frame_idx / fps if fps > 0 else 0
                diffs.append((timestamp, diff))

            prev_hist = hist

        frame_idx += 1

    cap.release()

    if not diffs:
        # Fallback: evenly space frames across video
        return _evenly_spaced_timestamps(duration, max_frames)

    # Sort by diff score (highest = biggest visual change)
    diffs.sort(key=lambda x: x[1], reverse=True)

    # Filter: enforce minimum interval between selected timestamps
    selected = []
    for ts, score in diffs:
        if ts < 1.0:  # skip very beginning
            continue
        if all(abs(ts - s) >= min_interval for s in selected):
            selected.append(ts)
        if len(selected) >= max_frames:
            break

    # Sort chronologically
    selected.sort()

    # If we got too few, fill with evenly spaced
    if len(selected) < 5:
        selected = _evenly_spaced_timestamps(duration, max_frames)

    return selected


def _evenly_spaced_timestamps(duration: float, count: int) -> list[float]:
    """Generate evenly spaced timestamps across the video duration."""
    if duration <= 0:
        return [0.0]
    interval = duration / (count + 1)
    return [interval * (i + 1) for i in range(count)]


def extract_frames_at_timestamps(video_path: str, timestamps: list[float], output_dir: str) -> list[str]:
    """
    Extract PNG frames at specific timestamps using ffmpeg.
    Returns list of saved frame file paths.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    frame_paths = []

    for i, ts in enumerate(timestamps):
        output_path = os.path.join(output_dir, f"frame_{i:03d}.png")
        cmd = [
            "ffmpeg", "-ss", str(ts),
            "-i", video_path,
            "-frames:v", "1",
            "-y",
            output_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and os.path.exists(output_path):
            frame_paths.append(output_path)

    return frame_paths


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except (ValueError, AttributeError):
        return 0.0
