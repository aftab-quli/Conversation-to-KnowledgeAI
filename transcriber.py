"""
transcriber.py
--------------
Handles audio transcription via OpenAI Whisper API.
Sends extracted WAV audio and returns timestamped transcript.
"""

import os
from openai import OpenAI


def transcribe_audio(audio_path: str, api_key: str) -> dict:
    """
    Transcribe audio using OpenAI Whisper API.

    Returns dict with:
      - text: full transcript string
      - segments: list of {start, end, text} dicts with timestamps
    """
    client = OpenAI(api_key=api_key)

    # Check file size — Whisper API limit is 25MB
    file_size = os.path.getsize(audio_path)
    if file_size > 25 * 1024 * 1024:
        # Split and transcribe in chunks
        return _transcribe_large_file(audio_path, client)

    with open(audio_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="verbose_json",
            timestamp_granularities=["segment"]
        )

    segments = []
    if hasattr(response, 'segments') and response.segments:
        for seg in response.segments:
            segments.append({
                "start": seg.get("start", seg.start) if hasattr(seg, "start") else seg.get("start", 0),
                "end": seg.get("end", seg.end) if hasattr(seg, "end") else seg.get("end", 0),
                "text": seg.get("text", seg.text) if hasattr(seg, "text") else seg.get("text", ""),
            })

    return {
        "text": response.text if hasattr(response, 'text') else str(response),
        "segments": segments,
    }


def _transcribe_large_file(audio_path: str, client: OpenAI) -> dict:
    """
    Handle audio files larger than 25MB by splitting with ffmpeg.
    Splits into 10-minute chunks and transcribes each.
    """
    import subprocess
    import tempfile
    from pathlib import Path

    chunk_dir = tempfile.mkdtemp()
    chunk_pattern = os.path.join(chunk_dir, "chunk_%03d.wav")

    # Split into 10-minute chunks
    cmd = [
        "ffmpeg", "-i", audio_path,
        "-f", "segment",
        "-segment_time", "600",  # 10 minutes
        "-c", "copy",
        "-y",
        chunk_pattern
    ]
    subprocess.run(cmd, capture_output=True, text=True)

    # Transcribe each chunk
    all_text = []
    all_segments = []
    time_offset = 0.0

    chunk_files = sorted(Path(chunk_dir).glob("chunk_*.wav"))

    for chunk_path in chunk_files:
        with open(str(chunk_path), "rb") as f:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="verbose_json",
                timestamp_granularities=["segment"]
            )

        all_text.append(response.text if hasattr(response, 'text') else str(response))

        if hasattr(response, 'segments') and response.segments:
            for seg in response.segments:
                start = seg.start if hasattr(seg, 'start') else seg.get("start", 0)
                end = seg.end if hasattr(seg, 'end') else seg.get("end", 0)
                text = seg.text if hasattr(seg, 'text') else seg.get("text", "")
                all_segments.append({
                    "start": start + time_offset,
                    "end": end + time_offset,
                    "text": text,
                })

        # Get chunk duration for offset
        duration = _get_audio_duration(str(chunk_path))
        time_offset += duration

        # Cleanup chunk
        chunk_path.unlink()

    # Cleanup temp dir
    os.rmdir(chunk_dir)

    return {
        "text": " ".join(all_text),
        "segments": all_segments,
    }


def _get_audio_duration(audio_path: str) -> float:
    """Get audio file duration in seconds."""
    import subprocess
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        audio_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except (ValueError, AttributeError):
        return 600.0  # Default 10 min per chunk


def format_transcript_with_timestamps(segments: list[dict]) -> str:
    """Format transcript segments into readable text with timestamps."""
    lines = []
    for seg in segments:
        start_min = int(seg["start"] // 60)
        start_sec = int(seg["start"] % 60)
        timestamp = f"[{start_min:02d}:{start_sec:02d}]"
        lines.append(f"{timestamp} {seg['text'].strip()}")
    return "\n".join(lines)
