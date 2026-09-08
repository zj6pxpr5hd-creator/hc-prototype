import json
import os
import re
from datetime import datetime, timezone


GOOGLE_SHEETS_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.file",
]


def _secret_value(name, default=None):
    try:
        import streamlit as st

        return st.secrets.get(name, default)
    except Exception:
        return default


def _service_account_info():
    secret_credentials = _secret_value("google_service_account")
    if secret_credentials:
        return dict(secret_credentials)

    credentials_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not credentials_json:
        raise RuntimeError(
            "Google Sheets is not configured. Add google_service_account to "
            "Streamlit secrets or set GOOGLE_SERVICE_ACCOUNT_JSON."
        )

    try:
        credentials = json.loads(credentials_json)
    except json.JSONDecodeError as error:
        raise RuntimeError("GOOGLE_SERVICE_ACCOUNT_JSON is not valid JSON.") from error

    if not isinstance(credentials, dict):
        raise RuntimeError("Google service account credentials must be a JSON object.")

    return credentials


def _setting(name, default=None):
    return os.getenv(name) or _secret_value(name, default)


def save_feedback(category, message, email=""):
    message = message.strip()
    email = email.strip()
    if not message:
        raise ValueError("Feedback message cannot be empty.")
    if email and not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
        raise ValueError("Please enter a valid email address or leave it empty.")

    import requests

    url = _setting("GOOGLE_SHEET_URL")
    if not url:
        raise RuntimeError("GOOGLE_SHEET_URL is not configured.")
    if not re.fullmatch(r"https://[^\s]+", url):
        raise RuntimeError("GOOGLE_SHEET_URL must be a valid HTTPS URL.")

    payload = {
        "message": message,
        "email": email,
        "category": category,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
    except requests.RequestException as error:
        raise RuntimeError("The feedback service could not be reached.") from error