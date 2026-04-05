"""
app.py
------
VicSherlock — Conversation-to-Knowledge AI agent for Vic.ai.
Upload a video recording → get a structured step-by-step guide with screenshots.
Also handles Slack bot events for proactive documentation detection.

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

# Lazy imports — these modules pull in heavy deps (opencv, faster-whisper, numpy)
# Only import them when actually processing a video to stay under 512MB on Render free tier
video_processor = None
frame_analyzer = None
guide_generator = None
doc_builder = None
transcriber = None

def _lazy_import_video_modules():
    global video_processor, frame_analyzer, guide_generator, doc_builder, transcriber
    if video_processor is None:
        import video_processor as _vp
        import frame_analyzer as _fa
        import guide_generator as _gg
        import doc_builder as _db
        import transcriber as _tr
        video_processor = _vp
        frame_analyzer = _fa
        guide_generator = _gg
        doc_builder = _db
        transcriber = _tr

try:
    from gong_client import get_gong_findings
except ImportError:
    get_gong_findings = None

try:
    from slack_bot import create_slack_bot_scanner
except ImportError:
    create_slack_bot_scanner = None

# No Bolt — we handle all Slack events directly via Flask for full control
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

# Event log for diagnostics
_recent_events = []

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def run_video_job(job_id: str, video_path: str, instructions: str, doc_type: str, transcript: str = None):
    """Full pipeline: video → audio → transcript → frames → guide → docx.

    If transcript is provided, skip audio extraction and transcription.
    If transcript is None, extract audio and transcribe locally using faster-whisper.
    """
    # Lazy-load heavy modules only when actually processing video
    _lazy_import_video_modules()
    from video_processor import extract_audio, detect_scene_changes, extract_frames_at_timestamps, get_video_duration
    from frame_analyzer import process_all_frames, filter_duplicate_frames
    from guide_generator import GuideGenerator
    from doc_builder import generate_guide_document
    from transcriber import transcribe_audio, format_transcript_with_timestamps

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


@app.route("/generate-from-text", methods=["POST"])
def generate_from_text():
    """Generate a guide from pasted transcript text (no video needed)."""
    data = request.get_json(silent=True) or {}
    transcript = data.get("transcript", "").strip()
    instructions = data.get("instructions", "").strip()
    doc_type = data.get("doc_type", "step-by-step")

    if not transcript:
        return jsonify({"error": "Please paste a transcript."}), 400
    if not instructions:
        return jsonify({"error": "Please provide instructions."}), 400

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "running",
        "step": "Generating guide from transcript...",
        "progress": 10,
        "file": None,
        "error": None,
    }

    def run_text_job(_job_id=job_id, _transcript=transcript, _instructions=instructions, _doc_type=doc_type):
        try:
            from anthropic import Anthropic as _AC

            jobs[_job_id]["step"] = "Analyzing transcript with Claude..."
            jobs[_job_id]["progress"] = 30

            client = _AC(api_key=os.getenv("ANTHROPIC_API_KEY"))

            type_prompts = {
                "step-by-step": "Create a detailed step-by-step guide with numbered steps, clear actions, and expected outcomes for each step.",
                "faq": "Create an FAQ document with questions and answers extracted from the content.",
                "troubleshooting": "Create a troubleshooting guide with common issues, symptoms, and solutions.",
            }
            type_instruction = type_prompts.get(_doc_type, type_prompts["step-by-step"])

            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                system=f"You are an expert technical writer. {type_instruction} Format the output in clean Markdown.",
                messages=[{
                    "role": "user",
                    "content": f"Instructions: {_instructions}\n\nTranscript/Source Content:\n{_transcript[:15000]}"
                }],
            )
            guide_text = response.content[0].text

            jobs[_job_id]["step"] = "Building document..."
            jobs[_job_id]["progress"] = 70

            # Save as markdown file
            output_filename = f"guide_{_job_id[:8]}.md"
            output_path = OUTPUT_DIR / output_filename
            with open(output_path, "w") as f:
                f.write(guide_text)

            jobs[_job_id]["step"] = "Done!"
            jobs[_job_id]["progress"] = 100
            jobs[_job_id]["status"] = "done"
            jobs[_job_id]["file"] = {"name": f"Guide — {_instructions[:60]}", "filename": output_filename}
            jobs[_job_id]["step_count"] = guide_text.count("\n## ") + guide_text.count("\n### ") + guide_text.count("\n1.") + guide_text.count("\n- **Step")

        except Exception as e:
            logger.error(f"Error in text job: {e}")
            jobs[_job_id]["status"] = "error"
            jobs[_job_id]["error"] = str(e)

    thread = threading.Thread(target=run_text_job)
    thread.daemon = True
    thread.start()
    return jsonify({"job_id": job_id})


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


# --- Slack Events Endpoint ---
@app.route("/slack/events", methods=["POST"])
def slack_events():
    """Handle ALL Slack events directly — single handler for everything."""
    logger.info(">>> /slack/events HIT <<<")
    data = request.get_json(silent=True)
    if not data:
        logger.warning("No JSON data in request")
        return jsonify({"error": "no data"}), 400

    logger.info(f"Event payload type: {data.get('type')} | keys: {list(data.keys())}")

    # Handle Slack URL verification challenge
    if data.get("type") == "url_verification":
        logger.info("Received Slack URL verification challenge")
        return jsonify({"challenge": data.get("challenge", "")})

    # Handle event callbacks directly (bypass Bolt for reliability)
    if data.get("type") == "event_callback":
        event = data.get("event", {})
        event_type = event.get("type", "")

        logger.info(f"Received Slack event: {event_type} | Full event: {event}")

        # Store recent events for diagnostics
        _recent_events.append({"type": event_type, "event": event, "ts": str(datetime.now())})
        if len(_recent_events) > 20:
            _recent_events.pop(0)

        # Handle assistant_thread_started (from Agents & AI Apps feature)
        if event_type == "assistant_thread_started":
            thread_context = event.get("assistant_thread", {})
            channel = thread_context.get("channel_id", "")
            thread_ts = thread_context.get("thread_ts", "")
            if channel:
                try:
                    from slack_sdk import WebClient
                    slack = WebClient(token=os.getenv("SLACK_BOT_TOKEN"))
                    slack.chat_postMessage(
                        channel=channel,
                        thread_ts=thread_ts,
                        text=(
                            ":mag: *VicSherlock — Ready to investigate!*\n\n"
                            "I scan Slack channels for documentation-worthy conversations and help keep your team's knowledge up to date.\n\n"
                            "Try asking me:\n"
                            "• _\"What have you found recently?\"_\n"
                            "• _\"What did Katie Roy say about VicPay?\"_\n"
                            "• _\"Scan channels for updates\"_"
                        ),
                        mrkdwn=True
                    )
                except Exception as e:
                    logger.error(f"Error handling assistant_thread_started: {e}")
            return jsonify({"ok": True}), 200

        # Handle DM messages (both regular and threaded from Agents mode)
        if event_type == "message" and not event.get("bot_id") and not event.get("subtype"):
            user_text = event.get("text", "").strip()
            channel = event.get("channel", "")
            user_id = event.get("user", "")
            thread_ts = event.get("thread_ts", None)  # For Agents & AI Apps threaded mode

            # Handle "new chat" / "start fresh" commands
            if user_text and user_text.lower() in ("new chat", "start fresh", "clear", "reset"):
                try:
                    from slack_sdk import WebClient
                    slack = WebClient(token=os.getenv("SLACK_BOT_TOKEN"))
                    slack.chat_postMessage(
                        channel=channel,
                        text=(
                            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                            ":mag: *VicSherlock — New Session*\n"
                            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                            "Hey! I'm VicSherlock, your Conversation-to-Knowledge AI.\n\n"
                            "I scan Slack channels for documentation-worthy conversations and help keep your team's knowledge up to date.\n\n"
                            "Try asking me:\n"
                            "• _\"What have you found recently?\"_\n"
                            "• _\"What did Katie Roy say about VicPay?\"_\n"
                            "• _\"Scan channels for updates\"_\n\n"
                            "Or send me a doc/PDF and I'll help update it!"
                        ),
                        mrkdwn=True
                    )
                except Exception as e:
                    logger.error(f"Error sending new chat message: {e}")
                return jsonify({"ok": True}), 200

            if user_text and channel:
                # Send a fun "thinking" message immediately
                import random
                thinking_messages = [
                    ":mag: _The game is afoot! Scanning 221B channels of data..._",
                    ":smoking: _Elementary... give me a moment to consult my mind palace._",
                    ":male-detective: _Sherlock is on the case. Dusting for digital fingerprints..._",
                    ":mag_right: _Investigating the evidence... cross-referencing witness statements..._",
                    ":brain: _Retreating to my mind palace. It's bigger on the inside..._",
                    ":memo: _Ah, an interesting query! *adjusts deerstalker* Let me examine the clues..._",
                    ":sleuth_or_spy: _No mystery too great, no doc too small. Interrogating the data..._",
                    ":coffee: _Brewing deductions... this case requires extra caffeine._",
                    ":dart: _Zeroing in on the truth. Hold my magnifying glass..._",
                    ":books: _Flipping through the archives... I've seen this pattern before._",
                    ":bulb: _A-ha! Something's forming... let me piece it together._",
                    ":telescope: _Observing from a distance... the details are coming into focus._",
                    ":world_map: _Following the trail of breadcrumbs across your workspace..._",
                    ":old_key: _Unlocking the vault of knowledge. The cipher is almost cracked..._",
                    ":violin: _*plays violin pensively*... the answer is revealing itself._",
                    ":zap: _My neurons are firing! Correlating evidence across multiple sources..._",
                    ":hourglass_flowing_sand: _Patience, dear Watson. Genius takes a moment._",
                    ":smoking: _*puffs pipe thoughtfully*... I have a theory forming._",
                    ":newspaper: _Reading between the lines... literally. That's my whole job._",
                    ":crystal_ball: _I don't need a crystal ball, I have Gong transcripts and Slack messages._",
                    ":jigsaw: _Found a piece of the puzzle. Now finding where it fits..._",
                    ":microscope: _Examining the evidence at 400x magnification..._",
                    ":tophat: _*tips hat* One moment while I deduce brilliantly._",
                    ":fire: _Hot on the trail! The game is most definitely afoot._",
                    ":lock: _Cracking the case... and by case I mean your Gong calls._",
                    ":mag: _*twirls magnifying glass* Let me take a closer look at this..._",
                    ":pencil2: _Taking notes in my case file... this is getting interesting._",
                    ":eyes: _My keen eye has spotted something. Analyzing..._",
                    ":male-detective: _*strokes mustache contemplatively*... the truth is near._",
                    ":thought_balloon: _Watson would be impressed by this deduction. One moment..._",
                    ":spider_web: _Untangling the web of information..._",
                    ":mag_right: _The clues are falling into place. Almost there..._",
                    ":closed_lock_with_key: _Decrypting the mystery... just a few more seconds._",
                    ":performing_arts: _Behind every conversation lies a story worth documenting._",
                    ":trophy: _This might be my finest case yet. Let me verify..._",
                    ":rocket: _Mind palace processing at full speed..._",
                    ":game_die: _The odds are in our favor. Calculating..._",
                    ":bridge_at_night: _Connecting the dots between Slack and Gong..._",
                    ":stopwatch: _Faster than Scotland Yard, I promise. Almost done..._",
                    ":candle: _Burning the midnight oil on this investigation..._",
                    ":compass: _My internal compass is pointing to the answer..._",
                    ":scroll: _Consulting the ancient scrolls... I mean Slack messages._",
                    ":bomb: _This is going to be an explosive finding. Stand by..._",
                    ":shushing_face: _Shh... I'm eavesdropping on your channels. For science._",
                    ":nerd_face: _Applying advanced deductive algorithms... a.k.a. reading._",
                    ":moneybag: _This insight is going to be worth its weight in gold._",
                    ":test_tube: _Running the evidence through my analytical laboratory..._",
                    ":mag: _If this were a crime scene, I'd have it solved already._",
                    ":gear: _The gears of deduction are turning..._",
                    ":globe_with_meridians: _Scanning the entire knowledge universe for you..._",
                    ":chess_pawn: _Making my next move... checkmate incoming._",
                    ":sparkles: _Something magical is about to happen. Patience..._",
                    ":raised_eyebrow: _Hmm, now THIS is curious. Let me dig deeper._",
                    ":card_index: _Cross-referencing my files. I keep meticulous records._",
                    ":detective: _The mustache is tingling. That means I'm onto something._",
                    ":smoking: _*lights pipe* The answer is always in the details._",
                    ":bookmark: _Marking this as exhibit A. Analysis in progress..._",
                    ":label: _Labeling the evidence. Organization is key to deduction._",
                    ":round_pushpin: _Pinning clues to my evidence board..._",
                    ":link: _Connecting conversations to knowledge. It's what I do._",
                    ":bulb: _Watson, hand me my magnifying glass! I see something!_",
                    ":page_facing_up: _Reviewing the transcripts with a fine-toothed comb._",
                    ":male-detective: _No stone unturned, no message unread. Working on it..._",
                    ":watch: _Time is of the essence! But also quality. Give me a sec._",
                    ":key: _Found a key piece of evidence. Building the case now..._",
                    ":hammer_and_wrench: _Assembling the facts into something useful..._",
                    ":racing_car: _Deduction engine running at full throttle!_",
                    ":mountain: _We're about to reach the peak of understanding..._",
                    ":ocean: _Diving deep into the sea of data..._",
                    ":ringed_planet: _My deductive powers span galaxies. Give me a moment._",
                    ":dna: _Analyzing the DNA of this conversation..._",
                    ":satellite: _Intercepting relevant transmissions from your workspace..._",
                    ":owl: _Wise as an owl, fast as a... slightly slower owl. Working on it._",
                    ":fox_face: _Sly as a fox, sharp as a tack. Deducing..._",
                    ":eagle: _Eagle-eye view of your data. Swooping in on the answer._",
                    ":lion_face: _Bravely venturing into the data jungle..._",
                    ":100: _Going to give you a 100% thorough answer. Hold tight._",
                    ":crossed_swords: _Battling through the noise to find the signal..._",
                    ":shield: _Defending the truth with facts. Almost ready._",
                    ":bow_and_arrow: _Taking aim at the perfect answer..._",
                    ":broom: _Sweeping through the channels for hidden gems..._",
                    ":slot_machine: _Feeling lucky? My deductions always hit the jackpot._",
                    ":mag: _*polishes magnifying glass* This requires my A-game._",
                    ":clipboard: _Case file open. Evidence being reviewed..._",
                    ":file_cabinet: _Pulling records from the VicSherlock archives..._",
                    ":bar_chart: _The data doesn't lie. Let me read it for you._",
                    ":loudspeaker: _I've heard things in your channels. Let me tell you about them._",
                    ":bell: _*ding ding* That's the sound of a breakthrough coming._",
                    ":flashlight: _Shining a light on the dark corners of your data..._",
                    ":footprints: _Following the footprints... they lead somewhere interesting._",
                    ":thread: _Pulling on a thread. Let's see where it leads..._",
                    ":tada: _Spoiler: the answer is going to be good. Almost there._",
                    ":pray: _Even Sherlock needs a moment of focus. Concentrating..._",
                    ":brain: _221B neurons firing simultaneously. Stand by for brilliance._",
                    ":smoking: _*adjusts mustache, puffs pipe* The game... is afoot._",
                    ":male-detective: _They don't call me VicSherlock for nothing. Working..._",
                    ":mag_right: _Zooming in... enhancing... almost got it._",
                    ":star2: _A star detective on a star case. Results incoming._",
                    ":construction: _Building your answer brick by brick..._",
                    ":hourglass: _The sands of deduction are flowing. Nearly done._",
                    ":speaking_head_in_silhouette: _Listening to what your conversations are really saying..._",
                ]
                try:
                    from slack_sdk import WebClient as _WC
                    _thinking_slack = _WC(token=os.getenv("SLACK_BOT_TOKEN"))
                    thinking_kwargs = {"channel": channel, "text": random.choice(thinking_messages), "mrkdwn": True}
                    if thread_ts:
                        thinking_kwargs["thread_ts"] = thread_ts
                    thinking_msg = _thinking_slack.chat_postMessage(**thinking_kwargs)
                    thinking_ts = thinking_msg.get("ts")
                except Exception:
                    thinking_ts = None

                # Process in background thread so we respond to Slack within 3s
                def reply_with_claude(msg_text=user_text, msg_channel=channel, msg_user=user_id, _thinking_ts=thinking_ts, _thread_ts=thread_ts):
                    try:
                        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
                        slack_token = os.getenv("SLACK_BOT_TOKEN")
                        if not anthropic_key or not slack_token:
                            logger.error("Missing ANTHROPIC_API_KEY or SLACK_BOT_TOKEN")
                            return

                        from anthropic import Anthropic as AnthropicClient
                        from slack_sdk import WebClient

                        client = AnthropicClient(api_key=anthropic_key)
                        slack = WebClient(token=slack_token)

                        # --- Live Slack scan: fetch recent messages from channels ---
                        live_findings = ""
                        try:
                            import time

                            # Build a user cache with ONE bulk call instead of per-message lookups
                            _user_cache = {}
                            try:
                                users_resp = slack.users_list(limit=200)
                                for u in users_resp.get("members", []):
                                    _user_cache[u["id"]] = u.get("real_name") or u.get("name", u["id"])
                            except Exception:
                                pass  # If user list fails, we'll just use IDs

                            # Get channels the bot is in (single page is fine for most workspaces)
                            channels_to_scan = []
                            ch_result = slack.conversations_list(types="public_channel,private_channel", limit=200)
                            for ch in ch_result.get("channels", []):
                                if ch.get("is_member"):
                                    channels_to_scan.append({"id": ch["id"], "name": ch.get("name", "unknown")})

                            scan_snippets = []
                            one_week_ago = str(time.time() - 7 * 86400)

                            for ch_info in channels_to_scan:
                                try:
                                    history = slack.conversations_history(
                                        channel=ch_info["id"],
                                        oldest=one_week_ago,
                                        limit=10
                                    )
                                    for msg in history.get("messages", []):
                                        text = msg.get("text", "").strip()
                                        if len(text) > 50:
                                            uid = msg.get("user", "unknown")
                                            username = _user_cache.get(uid, uid)
                                            scan_snippets.append(
                                                f"[#{ch_info['name']}] {username}: {text[:300]}"
                                            )
                                except Exception:
                                    pass  # Skip channels we can't read

                            channel_names = [c["name"] for c in channels_to_scan]
                            logger.info(f"Scanned {len(channels_to_scan)} channels: {channel_names}")
                            logger.info(f"Found {len(scan_snippets)} message snippets")

                            if scan_snippets:
                                live_findings = "RECENT MESSAGES FROM SLACK CHANNELS (last 7 days):\n\n" + "\n\n".join(scan_snippets[:40])
                            elif channels_to_scan:
                                live_findings = f"Scanned {len(channels_to_scan)} channels ({', '.join(channel_names[:10])}) but found no substantial messages in the last 7 days. The bot may need to be invited to more active channels."
                            else:
                                live_findings = "The bot is not a member of any channels yet. It needs to be invited to channels (e.g. /invite @VicSherlock in Slack) to scan conversations."

                        except Exception as scan_err:
                            logger.error(f"Error during live scan: {scan_err}")
                            live_findings = "Could not perform live scan at this time."

                        # --- Gong integration: fetch recent call transcripts ---
                        gong_findings = ""
                        try:
                            if get_gong_findings:
                                gong_findings = get_gong_findings()
                        except Exception as gong_err:
                            logger.error(f"Error fetching Gong data: {gong_err}")

                        system_prompt = f"""You are VicSherlock, a Conversation-to-Knowledge AI bot for Vic.ai.
Your job is to monitor Slack conversations AND Gong calls for documentation-worthy content and help keep team knowledge up to date.

You actively scan Slack channels and Gong call recordings for tutorials, process changes, troubleshooting threads, and other documentation-worthy conversations, then alert the right people to update docs.
You can also convert video recordings into step-by-step implementation guides with screenshots at https://vicsherlock.onrender.com

{live_findings}

{gong_findings}

Based on all the data above (Slack messages AND Gong calls), identify documentation-worthy content: process changes, tutorials, troubleshooting guides, new procedures, training content, customer insights, or anything that should be captured in a knowledge base or Guru card.

When users ask about recent findings, analyze the real data above and share what you've found from BOTH Slack and Gong.
When users ask you to update documentation or Guru cards, acknowledge the request and explain what steps are needed.
When users ask about a specific person or topic, search through all the data above for relevant content.
When users ask about calls or meetings, use the Gong call data.

Be friendly, concise, and helpful. Use emoji sparingly."""

                        response = client.messages.create(
                            model="claude-sonnet-4-20250514",
                            max_tokens=1024,
                            system=system_prompt,
                            messages=[{"role": "user", "content": msg_text}],
                        )
                        reply = response.content[0].text

                        # Delete the thinking message and send the real reply
                        if _thinking_ts:
                            try:
                                slack.chat_delete(channel=msg_channel, ts=_thinking_ts)
                            except Exception:
                                pass  # If we can't delete it, no big deal

                        reply_kwargs = {"channel": msg_channel, "text": reply}
                        if _thread_ts:
                            reply_kwargs["thread_ts"] = _thread_ts
                        slack.chat_postMessage(**reply_kwargs)
                        logger.info(f"Replied to {msg_user} in {msg_channel}")

                    except Exception as e:
                        logger.error(f"Error replying to message: {e}")
                        try:
                            from slack_sdk import WebClient
                            slack = WebClient(token=os.getenv("SLACK_BOT_TOKEN"))
                            err_kwargs = {"channel": msg_channel, "text": "Oops, I hit a snag processing that. Try again in a moment!"}
                            if _thread_ts:
                                err_kwargs["thread_ts"] = _thread_ts
                            slack.chat_postMessage(**err_kwargs)
                        except Exception:
                            pass

                thread = threading.Thread(target=reply_with_claude)
                thread.daemon = True
                thread.start()

        # Handle app_home_opened
        elif event_type == "app_home_opened":
            user_id = event.get("user", "")
            try:
                from slack_sdk import WebClient
                slack = WebClient(token=os.getenv("SLACK_BOT_TOKEN"))
                slack.views_publish(
                    user_id=user_id,
                    view={
                        "type": "home",
                        "blocks": [
                            {"type": "header", "text": {"type": "plain_text", "text": "VicSherlock"}},
                            {"type": "section", "text": {"type": "mrkdwn", "text": "*Your Conversation-to-Knowledge AI Agent*\n\nI scan Slack channels for documentation-worthy conversations — tutorials, process changes, troubleshooting threads — and alert you when it's time to update your docs."}},
                            {"type": "divider"},
                            {"type": "section", "text": {"type": "mrkdwn", "text": "*What I can do:*\n• Scan channels for knowledge worth capturing\n• Alert you when Guru cards need updating\n• Convert video recordings into step-by-step guides\n• Help keep your team's documentation current"}},
                            {"type": "section", "text": {"type": "mrkdwn", "text": "Send me a message to get started!"}},
                        ]
                    }
                )
            except Exception as e:
                logger.error(f"Error publishing home tab: {e}")

        return jsonify({"ok": True}), 200

    return jsonify({"ok": True}), 200


# --- Background Channel Scanner ---
# Periodically scans Slack channels and posts findings to #vicsherlock

_last_scan_results = None
_scan_lock = threading.Lock()


def run_channel_scan():
    """Execute a single channel scan and DM findings."""
    global _last_scan_results
    try:
        if create_slack_bot_scanner is None:
            logger.warning("slack_bot module not available — skipping scan")
            return None

        scanner = create_slack_bot_scanner()
        if scanner is None:
            logger.warning("Could not create scanner (missing tokens) — skipping scan")
            return None

        logger.info("Starting channel scan...")
        results = scanner.perform_full_scan_and_notify(limit_channels=10)

        with _scan_lock:
            _last_scan_results = {
                "timestamp": datetime.now().isoformat(),
                "results": results,
            }

        logger.info(
            f"Scan complete: {results.get('channels_scanned', 0)} channels scanned, "
            f"{len(results.get('documentation_worthy', []))} findings, "
            f"{results.get('notifications', {}).get('sent', 0)} DMs sent"
        )
        return results

    except Exception as e:
        logger.error(f"Channel scan failed: {e}")
        return None


@app.route("/scan-now", methods=["POST", "GET"])
def trigger_scan():
    """Manually trigger a channel scan (POST or GET /scan-now)."""
    def do_scan():
        run_channel_scan()

    thread = threading.Thread(target=do_scan, daemon=True)
    thread.start()
    return jsonify({"status": "scan_started", "message": "Channel scan triggered — findings will be DM'd to you"}), 200


@app.route("/scanner-status", methods=["GET"])
def scanner_status():
    """Get the last scan results."""
    with _scan_lock:
        if _last_scan_results:
            return jsonify(_last_scan_results), 200
        else:
            return jsonify({"status": "no_scans_yet"}), 200


@app.route("/debug-events")
def debug_events():
    """Show recent Slack events for debugging."""
    return jsonify({"recent_events": _recent_events, "count": len(_recent_events)})


@app.route("/debug-gong")
def debug_gong():
    """Test Gong API connection and show diagnostics."""
    if not get_gong_findings:
        return jsonify({"error": "gong_client module not imported", "hint": "Check if gong_client.py exists"})

    from gong_client import GongClient
    client = GongClient()

    diag = {
        "is_configured": client.is_configured,
        "access_key_set": bool(client.access_key),
        "access_key_preview": client.access_key[:4] + "..." if client.access_key else "(empty)",
        "secret_set": bool(client.access_key_secret),
        "base_url": client.base_url,
        "env_base_url": os.getenv("GONG_BASE_URL", "(not set)"),
    }

    if client.is_configured:
        import requests as _req
        try:
            # Try a simple API call to test auth
            resp = _req.post(
                f"{client.base_url}/calls/extensive",
                auth=client._auth(),
                json={
                    "filter": {
                        "fromDateTime": "2026-01-01T00:00:00Z",
                        "toDateTime": "2026-04-05T00:00:00Z",
                    },
                    "contentSelector": {"exposedFields": {"parties": True}}
                },
                timeout=15,
            )
            diag["api_status_code"] = resp.status_code
            diag["api_response_preview"] = str(resp.text[:500])
            if resp.status_code == 200:
                data = resp.json()
                diag["calls_found"] = len(data.get("calls", []))
                diag["has_more"] = data.get("records", {}).get("cursor") is not None
            else:
                diag["api_error"] = resp.text[:300]
        except Exception as e:
            diag["connection_error"] = str(e)
    else:
        diag["hint"] = "Set GONG_ACCESS_KEY and GONG_ACCESS_KEY_SECRET env vars. These are API keys from Gong Settings > API, NOT your login username/password."

    return jsonify(diag)


@app.route("/send-finding", methods=["POST", "GET"])
def send_finding():
    """Send a test finding message as VicSherlock bot to the DM channel."""
    slack_token = os.getenv("SLACK_BOT_TOKEN")
    if not slack_token:
        return jsonify({"error": "no SLACK_BOT_TOKEN"}), 500

    data = request.get_json() or {}
    channel_name = data.get("channel_name", "#customer_stonewall-kitchen")
    author = data.get("author", "Anthony Margaritondo")
    topic = data.get("topic", "Approval Flow Behavior When Invoice Fields Are Modified")
    preview = data.get("preview", "Looking into how approvers change if someone modifies an invoice and changes a field that affects the approval flow...")
    target_channel = data.get("target_channel", os.getenv("NOTIFY_CHANNEL_ID", "D09JP0H2DSN"))

    try:
        from slack_sdk import WebClient
        slack = WebClient(token=slack_token)

        message = (
            f"Hey! I just scanned your Slack channels and noticed something in *{channel_name}* from *{author}* that looks like it should be documented:\n\n"
            f"*{topic}*\n\n"
            f"> {preview}\n\n"
            "Would you like me to:\n"
            "• *Create a new doc* from this conversation?\n"
            "• *Update an existing doc* — just send me the current doc/PDF and I'll revise it!\n\n"
            "Just reply here and let me know!"
        )

        result = slack.chat_postMessage(channel=target_channel, text=message, mrkdwn=True)
        return jsonify({"ok": True, "ts": result["ts"]}), 200

    except Exception as e:
        logger.error(f"Error sending finding: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("\n🔍 VicSherlock — Conversation-to-Knowledge AI")
    print("   Open http://localhost:5000 in your browser\n")
    app.run(debug=False, port=5000, host="0.0.0.0")
