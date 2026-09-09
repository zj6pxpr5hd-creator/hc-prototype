# Hook Checker

Hook Checker analyzes the first three seconds of an uploaded video and evaluates the hook with Whisper and Gemini.

## Requirements

- Python 3.14+
- A Gemini API key
- A Google Sheet for feedback storage
- OpenCV for visual motion analysis

## Install and run

```bash
./venv/bin/pip install -r requirements.txt
./venv/bin/pip install -r packages.txt
./venv/bin/streamlit run protone.py
```

Create a `.env` file with:

```text
GEMINI_API_KEY=your_gemini_api_key
```

## Configure Google Sheets feedback

1. Open your Google Sheet, then go to Extensions → Apps Script. A code editor linked to that specific sheet will open.

2. Write the function that receives the data
Delete the example code and paste a doPost(e) function that reads the received data (in JSON format) and adds it as a new row in the sheet with SpreadsheetApp.getActiveSheet().appendRow(...).

3. Publish as a Web App
Click "Deploy" → "New Deployment" → "Web Application" type. Set "Who has access" to "Anyone" (this is necessary so your Python script can call it externally). Click Deploy.

4. Copy the generated URL
After deployment, Google will provide you with a unique URL (such as https://script.google.com/macros/s/XXXXX/exec). This is the address your Python script will send the data to.

5. Call the URL from Python with request.post()
In your Streamlit script, simply use the request library to send a POST with the feedback data to that URL—no complex authentication required.

## Feedback flow

The feedback form appears only after a video has been analyzed successfully. It stores the UTC timestamp, category, message, and optional email in the configured worksheet. Videos, audio files, transcriptions, hook text, and API keys are not stored with feedback.

The visual analysis samples up to 10 grayscale frames per second from the first three seconds. Gemini receives this motion signal as supplementary context and returns a visual-perspective assessment. If the video cannot be read by OpenCV, the regular audio and text analysis can still continue.
