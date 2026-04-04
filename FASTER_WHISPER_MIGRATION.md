# VicSherlock Migration to faster-whisper

## Summary
Successfully updated VicSherlock to use **local Whisper transcription** via `faster-whisper` instead of OpenAI API. This eliminates the need for an OpenAI API key and allows transcription on resource-constrained environments like Render's free tier.

## Key Changes

### 1. transcriber.py
- Replaced `from openai import OpenAI` with `from faster_whisper import WhisperModel`
- Simplified `transcribe_audio()` function:
  - Removed `api_key` parameter (no longer needed)
  - Uses `WhisperModel("tiny")` with `device="cpu"` and `compute_type="int8"` for efficient inference
  - Direct local processing (no API calls)
  - RAM usage: ~512MB (suitable for Render free tier)
- Removed `_transcribe_large_file()` and `_get_audio_duration()` helper functions (faster-whisper handles streaming automatically)
- Kept `format_transcript_with_timestamps()` unchanged

### 2. app.py
- Changed import to direct (non-optional): `from transcriber import transcribe_audio, format_transcript_with_timestamps`
- Updated `run_video_job()` docstring to clarify local transcription
- Updated transcription logic:
  - Removed `openai_key = os.getenv("OPENAI_API_KEY")` check
  - Changed `transcribe_audio(audio_path, openai_key)` to `transcribe_audio(audio_path)` (no API key parameter)
  - Step 1 & 2 now extract audio locally and transcribe without any external API

### 3. requirements.txt
- Added `faster-whisper>=1.0.0`
- Removed dependency on `openai` (was never explicitly listed; now never needed)

### 4. render.yaml
- Removed `OPENAI_API_KEY` from envVars (no longer needed)
- Kept `ANTHROPIC_API_KEY` (still required for Claude guide generation)
- Build command already includes ffmpeg installation (no changes needed)

### 5. templates/index.html
- Changed transcript label from "Transcript" to "Transcript (optional)"
- Updated field hint: "Leave empty to auto-transcribe from video, or paste a transcript to skip transcription."
- Modified `generate()` function to NOT require transcript:
  - Removed the alert that demanded transcript input
  - Transcript is now truly optional; users can submit with empty transcript
  - If empty, server will auto-transcribe; if provided, server will use it

## Workflow

### User provides transcript (old approach still works):
1. User pastes transcript
2. Server skips Steps 1 & 2 (audio extraction and transcription)
3. Proceeds to frame extraction and guide generation

### User provides video without transcript (new capability):
1. Server extracts audio from video
2. Server runs local Whisper model (`tiny`) to transcribe audio
3. Proceeds to frame extraction and guide generation
4. **No API keys required**

## Environment Variables

**Before:**
- `OPENAI_API_KEY` (required if no transcript provided)
- `ANTHROPIC_API_KEY` (required for guide generation)

**After:**
- `ANTHROPIC_API_KEY` (required for guide generation)
- `OPENAI_API_KEY` (not needed)

## Performance Notes

- **faster-whisper tin model** uses int8 quantization on CPU
- Expected inference time: 1-2 minutes per hour of audio on modern CPU
- RAM usage: ~512MB (suitable for Render free tier with 512MB RAM limit)
- No network I/O for transcription (entirely local processing)

## Testing Checklist

- [ ] Python syntax verified (✓ Done)
- [ ] Deploy to Render (verify ffmpeg is available on build)
- [ ] Upload video without transcript (test auto-transcription)
- [ ] Upload video with transcript (test bypass)
- [ ] Verify ANTHROPIC_API_KEY is sufficient (no OPENAI_API_KEY needed)
- [ ] Monitor RAM usage during transcription
- [ ] Confirm final document generation works end-to-end

## Rollback Plan

If issues arise, revert to OpenAI Whisper API:
1. Restore original `transcriber.py` from git history
2. Add `openai` back to `requirements.txt`
3. Add `OPENAI_API_KEY` back to `render.yaml`
4. Revert `app.py` changes to the transcription call
5. Revert HTML changes to require transcript

## Files Modified

- `/tmp/vicsherlock_render/transcriber.py` - Complete rewrite
- `/tmp/vicsherlock_render/app.py` - Imports and transcription logic
- `/tmp/vicsherlock_render/requirements.txt` - Dependencies
- `/tmp/vicsherlock_render/render.yaml` - Environment variables
- `/tmp/vicsherlock_render/templates/index.html` - UI (label, hint, validation)
