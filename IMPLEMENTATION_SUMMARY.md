# Hook Checker - Implementation Summary

## Project Overview
A Streamlit-based web application that analyzes short-form video hooks (first 3 seconds of TikTok/Reels/YouTube Shorts) to evaluate their attention-grabbing effectiveness.

## Architecture

### 1. **protone.py** (Main Streamlit Application)
The core application with the following key functions:

- **`load_whisper_model()`**: Caches OpenAI Whisper "base" model for audio transcription
- **`evaluate_hook(hook, api_key, motion_score=None)`**: Uses Google Gemini API to evaluate hook effectiveness
  - Receives an optional OpenCV motion signal as supplementary context
  - Returns JSON with: score (1-10), text metrics, visual perspective, strengths, weaknesses, and improvement suggestion
- **`show_evaluation(result)`**: Renders evaluation results in Streamlit UI
- **`process_video(uploaded_file, api_key)`**: Main pipeline that:
  1. Computes an optional grayscale motion signal from up to 10 frames per second in the first 3 seconds
  2. Extracts audio from video using FFmpeg
  3. Transcribes audio using Whisper
  4. Extracts hook (first 3 seconds of speech)
  5. Sends the hook and motion signal to Gemini for evaluation
- **`render_feedback()`**: Shows a feedback form only after a successful analysis and saves category, message, optional email, and UTC timestamp to Google Sheets.
- **`main()`**: Streamlit UI with:
  - File uploader (MP4/MOV, max 500MB)
  - API key validation
  - Error handling and user feedback

### 2. **video_to_mp3.py** (Audio Extraction)
Converts video files to MP3 using FFmpeg:
- Validates input file exists
- Checks FFmpeg availability
- Extracts audio track with quality settings
- Error handling for timeout and invalid conversions

### 3. **extract_hook.py** (Hook Extraction)
Extracts speech from the first N seconds (default 3):
- Parses Whisper transcription segments
- Collects text until time limit reached
- Strips and returns clean text

## Dependencies
- `streamlit` - Web UI framework
- `openai-whisper` - Audio transcription
- `google-genai` - Gemini AI integration
- `python-dotenv` - Environment variable loading
- `gspread` - Google Sheets feedback storage
- `google-auth` - Google service account authentication
- `opencv-python-headless` - Grayscale frame sampling for visual motion context
- `ffmpeg` - Audio extraction (system dependency)

## Setup & Running

### Prerequisites
1. FFmpeg installed and in PATH
2. Python 3.14+
3. Gemini API key in `.env` file

### Installation
```bash
pip install -r requirements.txt
```

### Running the Application
```bash
streamlit run protone.py
```

## Environment Configuration
Create `.env` file with:
```
GEMINI_API_KEY=your_gemini_api_key_here
```

## Workflow
1. User uploads MP4 or MOV video (max 500MB)
2. System extracts audio → MP3
3. Whisper transcribes audio
4. Hook extracted (first 3 seconds)
5. OpenCV computes an optional motion signal from sampled grayscale frames
6. Gemini evaluates hook effectiveness and provides a visual-perspective assessment
7. Results displayed in UI with score, metrics, visual perspective, strengths, weaknesses, and suggestions
8. Feedback form appears after a successful response and stores submissions in Google Sheets

## Implementation Status
✅ Complete and fully functional
✅ All dependencies installed
✅ Environment configured
✅ Ready for deployment
