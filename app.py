"""
app.py
------
VicSherlock — Flask web server.
Upload a video recording → get a structured step-by-step guide with screenshots.

Run with: python app.py
Then open http://localhost:5000
"""

import os
import sys
import shutil
import threading
import uuid
import logging
from datetime import datetime
from pathlib import Path

from flask import Flask, render_template, request, jsonify, send_file
from dotenv import load_dotenv
from werkzeug.utils import secure_filename

load_dotenv()

from video_processor import extract_audio, detect_scene_changes, extract_frames_at_timestamps, get_video_duration
from frame_analyzer import process_all_frames, filter_duplicate_frames
from guide_generator import GuideGenerator
from doc_builder import generate_guide_document

from transcriber import transcribe_audio, format_transcript_with_timestamps

try:
    from slack_bot import create_slack_bot_scanner
except ImportError:
    create_slack_bot_scanner = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500MB max upload

# Directories
UPLOAD_DIR = Path("./uploads")
TEMP_DIR = Path("./temp")
OUTPUT_DIR = Path("./output")

for d in [UPLOAD_DIR, TEMP_DIR, OUTPUT_DIR]:
    d.mkdir(exist_ok=True)

# Job tracking
jobs = {}

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def run_video_job(job_id: str, video_path: str, instructions: str, doc_type: str, transcript: str = None):
    """Full pipeline: video → audio → transcript → frames → guide → docx.

    If transcript is provided, skip audio extraction and transcription.
    If transcript is None, extract audio and transcribe locally using faster-whisper.
    """
    job = jobs[job_id]
    job_temp = TEMP_DIR / job_id
    job_temp.mkdir(parents=True, exist_ok=True)

    try:
        # --- Step 1 & 2: Handle transcription ---
        if transcript:
            # Use provided transcript
            job["step"] = "Using provided transcript..."
            job["progress"] = 25
            logger.info(f"Job {job_id}: Using provided transcript")
            transcript_text = transcript
            segments = []
        else:
            # Extract audio and transcribe locally
            job["step"] = "Extracting audio from video..."
            job["progress"] = 10
            logger.info(f"Job {job_id}: Extracting audio")

            audio_path = str(job_temp / "audio.wav")
            extract_audio(video_path, audio_path)

            # Transcribe with local Whisper model
            job["step"] = "Transcribing audio with Whisper..."
            job["progress"] = 25
            logger.info(f"Job {job_id}: Transcribing")

            transcript_result = transcribe_audio(audio_path)
            transcript_text = transcript_result["text"]
            segments = transcript_result.get("segments", [])

        job["transcript_preview"] = transcript_text[:500] + "..." if len(transcript_text) > 500 else transcript_text

        # --- Step 3: Extract screenshots ---
        job["step"] = "Extracting screenshots from video..."
        job["progress"] = 45
        logger.info(f"Job {job_id}: Extracting frames")

        raw_frames_dir = str(job_temp / "frames_raw")
        timestamps = detect_scene_changes(video_path, max_frames=15)
        raw_frames = extract_frames_at_timestamps(video_path, timestamps, raw_frames_dir)

        # --- Step 4: Crop faces from screenshots ---
        job["step"] = "Processing screenshots (removing faces)..."
        job["progress"] = 55
        logger.info(f"Job {job_id}: Processing frames — cropping faces")

        clean_frames_dir = str(job_temp / "frames_clean")
        cleaned_frames = process_all_frames(raw_frames_dir, clean_frames_dir)

        # Remove near-duplicate frames
        unique_frames = filter_duplicate_frames(cleaned_frames)
        logger.info(f"Job {job_id}: {len(unique_frames)} unique frames after filtering")

        # --- Step 5: Generate guide with Claude ---
        job["step"] = "Generating guide with Claude..."
        job["progress"] = 70
        logger.info(f"Job {job_id}: Generating guide")

        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if not anthropic_key:
            raise RuntimeError("Missing ANTHROPIC_API_KEY in environment variables.")

        generator = GuideGenerator(anthropic_key)
        guide = generator.generate_guide(transcript_text, instructions, doc_type)

        # Map screenshots to steps
        frame_map = generator.map_screenshots_to_steps(guide["steps"], unique_frames)

        # --- Step 6: Generate .docx ---
        job["step"] = "Building document with screenshots..."
        job["progress"] = 90
        logger.info(f"Job {job_id}: Building docx")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        safe_title = "".join(c for c in guide["title"] if c.isalnum() or c in " _-")[:50].strip()
        filename = f"VicSherlock_{safe_title}_{timestamp}.docx"
        output_path = str(OUTPUT_DIR / filename)

        generate_guide_document(guide, frame_map, output_path)

        # --- Done ---
        job["status"] = "done"
        job["step"] = "Complete!"
        job["progress"] = 100
        job["file"] = {"name": guide["title"], "filename": filename}
        job["guide_title"] = guide["title"]
        job["step_count"] = len(guide["steps"])
        job["screenshot_count"] = len(unique_frames)

        logger.info(f"Job {job_id}: Complete — {filename}")

    except Exception as e:
        logger.error(f"Job {job_id}: Failed — {str(e)}")
        job["status"] = "error"
        job["error"] = str(e)

    finally:
        # Cleanup temp files
        try:
            shutil.rmtree(str(job_temp), ignore_errors=True)
            if os.path.exists(video_path):
                os.remove(video_path)
        except Exception:
            pass


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    # Validate file
    if "video" not in request.files:
        return jsonify({"error": "No video file provided."}), 400

    file = request.files["video"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Please upload an MP4, MOV, AVI, MKV, or WebM video file."}), 400

    instructions = request.form.get("instructions", "").strip()
    doc_type = request.form.get("doc_type", "step-by-step").strip()
    transcript = request.form.get("transcript", "").strip()

    if not instructions:
        return jsonify({"error": "Please provide instructions for the guide."}), 400

    # Save uploaded file
    job_id = str(uuid.uuid4())
    filename = secure_filename(file.filename)
    video_path = str(UPLOAD_DIR / f"{job_id}_{filename}")
    file.save(video_path)

    # Create job
    jobs[job_id] = {
        "status": "running",
        "step": "Starting...",
        "progress": 0,
        "file": None,
        "error": None,
    }

    # Run in background
    thread = threading.Thread(
        target=run_video_job,
        args=(job_id, video_path, instructions, doc_type, transcript if transcript else None)
    )
    thread.daemon = True
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found."}), 404
    return jsonify(job)


@app.route("/download/<path:filename>")
def download(filename):
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        return "File not found.", 404
    return send_file(str(file_path), as_attachment=True, download_name=filename)


@app.route("/scan-channels", methods=["POST"])
def scan_channels():
    """Scan Slack channels for documentation-worthy conversations."""
    try:
        if not create_slack_bot_scanner:
            return jsonify({"error": "Slack bot module not available."}), 400
        bot = create_slack_bot_scanner()
        if not bot:
            return jsonify({
                "error": "Slack bot not configured. Please set SLACK_BOT_TOKEN and ANTHROPIC_API_KEY."
            }), 400

        # Get optional limit parameter
        limit = request.args.get("limit", type=int)

        # Run scan in background
        job_id = str(uuid.uuid4())
        scan_job = {
            "status": "running",
            "step": "Initializing Slack channel scan...",
            "progress": 0,
            "results": None,
            "error": None,
        }
        jobs[job_id] = scan_job

        def run_slack_scan():
            try:
                scan_job["step"] = "Fetching channels..."
                scan_job["progress"] = 10

                scan_job["step"] = "Analyzing conversations with Claude..."
                scan_job["progress"] = 30

                results = bot.perform_full_scan_and_notify(limit_channels=limit)

                scan_job["step"] = "Complete!"
                scan_job["progress"] = 100
                scan_job["status"] = "done"
                scan_job["results"] = results

                logger.info(f"Slack scan complete: {results}")

            except Exception as e:
                logger.error(f"Error in Slack scan: {str(e)}")
                scan_job["status"] = "error"
                scan_job["error"] = str(e)

        thread = threading.Thread(target=run_slack_scan)
        thread.daemon = True
        thread.start()

        return jsonify({"job_id": job_id, "status": "scanning"})

    except Exception as e:
        logger.error(f"Error starting Slack scan: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/scan-status/<job_id>")
def scan_status(job_id):
    """Get status of a Slack scan job."""
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found."}), 404

    # Return scan-specific fields
    return jsonify({
        "status": job.get("status"),
        "step": job.get("step"),
        "progress": job.get("progress"),
        "results": job.get("results"),
        "error": job.get("error"),
    })


if __name__ == "__main__":
    print("\n🔍 VicSherlock — Conversation-to-Knowledge AI")
    print("   Open http://localhost:5000 in your browser\n")
    app.run(debug=False, port=5000, host="0.0.0.0")
