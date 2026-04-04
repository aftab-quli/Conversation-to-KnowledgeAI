# VicSherlock Browser-Based Transcription Update

## Overview
The VicSherlock web app has been updated to remove the dependency on OpenAI Whisper API and instead use browser-based transcription via the Web Speech API. Users can now transcribe video audio directly in their browser, with an option to manually enter transcripts.

## Changes Made

### 1. `requirements.txt`
- **Removed**: `openai>=1.3.0` package (no longer needed for Whisper)
- All other dependencies remain unchanged

### 2. `app.py`

#### Modified `run_video_job()` function:
- Added new parameter: `transcript: str = None`
- Added logic to handle three scenarios:
  1. **Transcript provided**: Skip Steps 1-2 (audio extraction & Whisper), use provided transcript directly
  2. **No transcript but OPENAI_API_KEY exists**: Fall back to Whisper for transcription (backwards compatible)
  3. **No transcript and no OPENAI_API_KEY**: Raise error asking user to provide transcript
- Updated progress reporting accordingly

#### Updated `/upload` route:
- Added extraction of `transcript` from form data: `request.form.get("transcript")`
- Passes transcript to `run_video_job()` as optional parameter

### 3. `templates/index.html`

#### New UI Section - "Video Transcript":
After a video file is dropped, a new section appears with:
- **"Transcribe in Browser"** button (disabled if Speech API not supported)
- **"Stop"** button (hidden by default, shows during transcription)
- **Status indicator** with real-time feedback
- **Textarea** for transcript entry/editing (allows manual entry too)

#### Browser Transcription Features:
- Uses `window.SpeechRecognition || window.webkitSpeechRecognition` API
- Creates a hidden `<video>` element to play video audio
- Captures audio in real-time from the video playback
- Shows live progress: "Listening... 45 words captured"
- Handles three completion states:
  - Success: "Transcription complete. Review and edit as needed."
  - Failure: "No speech detected. Please try again or enter manually."
  - Browser support: Graceful fallback message for unsupported browsers

#### Updated `generate()` function:
- Checks for transcript in textarea
- Includes transcript in form data if available: `formData.append('transcript', transcript)`

#### New Functions:
- `startBrowserTranscription()`: Initiates Speech Recognition and video playback
- `stopBrowserTranscription()`: Stops ongoing transcription
- `showTranscriptionError()`: Displays error messages in consistent styling

## User Workflow

1. **Drop Video** → Transcription section appears
2. **Click "Transcribe in Browser"** → Video plays audio, speech recognition captures words
3. **Review/Edit Transcript** → User can edit the generated or manually-entered transcript
4. **Click "Generate Guide"** → Video + transcript + instructions sent to backend
5. **Backend processes** → Uses provided transcript (skips audio extraction & Whisper)

## Backward Compatibility

- If no transcript is provided AND `OPENAI_API_KEY` is set, the system falls back to Whisper
- Existing deployments with OPENAI_API_KEY continue to work without changes
- Browser transcription is optional; users can always manually enter transcripts

## Browser Support

- **Chrome/Edge**: Full support via `window.SpeechRecognition`
- **Firefox**: Full support via `window.SpeechRecognition`
- **Safari**: Full support via `window.webkitSpeechRecognition`
- **Unsupported**: Users shown friendly message; manual transcript entry still available

## Styling & Dark Theme

All new UI elements follow the existing dark theme:
- Primary button: Purple (#7c3aed) with gradient
- Status indicators: Color-coded (info, success, error)
- Textarea: Matches existing form styling
- Consistent with the VicSherlock design system

## Notes

- No API keys required for browser transcription (all processing happens locally)
- Speech Recognition is continuous with interim results for smooth UX
- Transcript textarea allows both automatic and manual editing
- Status messages provide clear feedback throughout the process
