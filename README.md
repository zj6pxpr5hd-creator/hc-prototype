# Hook Checker

Hook Checker analyzes the first three seconds of an uploaded video and evaluates the hook with Whisper and Gemini.

## Requirements

- Python 3.14+
- FFmpeg available in `PATH`
- A Gemini API key
- A Google Sheet and service account for feedback storage

## Install and run

```bash
./venv/bin/pip install -r requirements.txt
./venv/bin/streamlit run protone.py
```

Create a `.env` file with:

```text
GEMINI_API_KEY=your_gemini_api_key
GOOGLE_SHEET_NAME=the_exact_google_sheet_name
GOOGLE_WORKSHEET_NAME=Feedback
```

## Configure Google Sheets feedback

1. Create a Google Cloud project and enable Google Sheets API and Google Drive API.
2. Create a service account and download its JSON credentials.
3. Share the target Google Sheet with the service account email as an editor.
4. Configure the credentials without committing them.

For local development, set the complete JSON document as one environment variable:

```bash
export GOOGLE_SERVICE_ACCOUNT_JSON='{"type":"service_account", ...}'
```

For Streamlit deployment, add the credentials to `.streamlit/secrets.toml` instead:

```toml
GOOGLE_SHEET_NAME = "the_exact_google_sheet_name"
GOOGLE_WORKSHEET_NAME = "Feedback"

[google_service_account]
type = "service_account"
project_id = "your-project-id"
private_key_id = "your-key-id"
private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
client_email = "service-account@your-project.iam.gserviceaccount.com"
client_id = "your-client-id"
```

Never commit `.env`, `.streamlit/secrets.toml`, or service-account JSON files.

## Feedback flow

The feedback form appears only after a video has been analyzed successfully. It stores the UTC timestamp, category, message, and optional email in the configured worksheet. Videos, audio files, transcriptions, hook text, and API keys are not stored with feedback.
