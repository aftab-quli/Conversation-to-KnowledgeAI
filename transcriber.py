"""
transcriber.py
--------------
Handles audio transcription via local Whisper model (faster-whisper).
Extracts audio and transcribes locally without needing an API key.
"""

import os
from faster_whisper import WhisperModel


def transcribe_audio(audio_path: str) -> dict:
    """
    Transcribe audio using local Whisper model (faster-whisper).
    Uses the tiny model for efficient inference (~512MB RAM).

    Returns dict with:
      - text: full transcript string
      - segments: list of {start, end, text} dicts with timestamps
    """
    model = WhisperModel("tiny", device="cpu", compute_type="int8")
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
