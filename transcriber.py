"""
transcriber.py
--------------
Handles audio transcription via local Whisper model (faster-whisper).
Extracts audio and transcribes locally without needing an API key.
"""

import os
import logging

logger = logging.getLogger(__name__)

# Lazy import — faster-whisper is heavy and may not be available on all deploys
_whisper_model = None


def _get_whisper_model():
    """Lazy-load the Whisper model only when actually needed."""
    global _whisper_model
    if _whisper_model is None:
        logger.info("Loading faster-whisper tiny model (first use)...")
        from faster_whisper import WhisperModel
        _whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")
        logger.info("Whisper model loaded.")
    return _whisper_model


def transcribe_audio(audio_path: str) -> dict:
    """
    Transcribe audio using local Whisper model (faster-whisper).
    Uses the tiny model for efficient inference.

    Returns dict with:
      - text: full transcript string
      - segments: list of {start, end, text} dicts with timestamps
    """
    model = _get_whisper_model()
    segments, info = model.transcribe(audio_path, beam_size=1)

    all_text = []
    all_segments = []

    for segment in segments:
        all_text.append(segment.text)
        all_segments.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text,
        })

    return {
        "text": " ".join(all_text),
        "segments": all_segments,
    }


def format_transcript_with_timestamps(segments: list[dict]) -> str:
    """Format transcript segments into readable text with timestamps."""
    lines = []
    for seg in segments:
        start_min = int(seg["start"] // 60)
        start_sec = int(seg["start"] % 60)
        timestamp = f"[{start_min:02d}:{start_sec:02d}]"
        lines.append(f"{timestamp} {seg['text'].strip()}")
    return "\n".join(lines)
