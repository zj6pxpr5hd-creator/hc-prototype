# Hook Checker - Implementation Summary

## Project Overview
A Streamlit-based web application that analyzes short-form video hooks (first 3 seconds of TikTok/Reels/YouTube Shorts) to evaluate their attention-grabbing effectiveness.

## Architecture

### 1. **protone.py** (Main Streamlit Application)
The core application with the following key functions:

- **`load_whisper_model()`**: Caches OpenAI Whisper "base" model for audio transcription
- **`evaluate_hook(hook, api_key)`**: Uses Google Gemini API to evaluate hook effectiveness
  - Returns JSON with: score (1-10), strengths, weaknesses, and improvement suggestion
- **`show_evaluation(result)`**: Renders evaluation results in Streamlit UI
- **`process_video(uploaded_file, api_key)`**: Main pipeline that:
  1. Extracts audio from video using FFmpeg
  2. Transcribes audio using Whisper
  3. Extracts hook (first 3 seconds of speech)
  4. Sends to Gemini for evaluation
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
5. Gemini evaluates hook effectiveness
6. Results displayed in UI with score, strengths, weaknesses, and suggestions
7. Feedback form appears after a successful response and stores submissions in Google Sheets

## Implementation Status
✅ Complete and fully functional
✅ All dependencies installed
✅ Environment configured
✅ Ready for deployment
