# Import json so credentials stored as text can be converted into a Python dictionary.
import json
# Import os so environment variables can be read.
import os
# Import regular-expression tools for validating URLs and email addresses.
import re
# Import timezone-aware timestamps for feedback records.
from datetime import datetime, timezone


# List the Google permissions needed by a service account that uses Sheets and Drive.
GOOGLE_SHEETS_SCOPES = [
    # Permit spreadsheet access.
    "https://www.googleapis.com/auth/spreadsheets",
    # Permit access to files created by this application.
    "https://www.googleapis.com/auth/drive.file",
]


# Read a named secret from Streamlit, returning a fallback when Streamlit is unavailable.
def _secret_value(name, default=None):
    try:
        # Import Streamlit only when this helper is called.
        import streamlit as st

        # Return the requested Streamlit secret or the supplied default value.
        return st.secrets.get(name, default)
    except Exception:
        # Treat missing Streamlit configuration as an absent secret.
        return default


# Load Google service-account credentials from Streamlit secrets or an environment variable.
def _service_account_info():
    # First look for credentials configured in Streamlit's secrets system.
    secret_credentials = _secret_value("google_service_account")
    # Use those credentials when they exist.
    if secret_credentials:
        # Copy the secret into a regular dictionary before returning it.
        return dict(secret_credentials)

    # Otherwise read the credentials as a JSON string from the operating system environment.
    credentials_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    # Stop when neither supported credentials source is configured.
    if not credentials_json:
        # Explain both supported configuration options to the user.
        raise RuntimeError(
            "Google Sheets is not configured. Add google_service_account to "
            "Streamlit secrets or set GOOGLE_SERVICE_ACCOUNT_JSON."
        )

    try:
        # Convert the JSON text into Python data.
        credentials = json.loads(credentials_json)
    except json.JSONDecodeError as error:
        # Replace the lower-level parsing error with a clear configuration message.
        raise RuntimeError("GOOGLE_SERVICE_ACCOUNT_JSON is not valid JSON.") from error

    # A service-account configuration must be an object represented by a dictionary.
    if not isinstance(credentials, dict):
        # Reject lists, strings, numbers, and other JSON types.
        raise RuntimeError("Google service account credentials must be a JSON object.")

    # Return the validated credentials dictionary.
    return credentials


# Read a setting from the environment first, then fall back to Streamlit secrets.
def _setting(name, default=None):
    # The environment value takes priority; the secret is used when it is absent.
    return os.getenv(name) or _secret_value(name, default)


# Validate feedback and send it to the configured Google Sheets service.
def save_feedback(category, message, email=""):
    # Remove accidental spaces around the user's feedback message.
    message = message.strip()
    # Remove accidental spaces around the optional email address.
    email = email.strip()
    # Do not send a record when the message contains no actual text.
    if not message:
        # Tell the caller why an empty message is not accepted.
        raise ValueError("Feedback message cannot be empty.")
    # Validate an email only when the user supplied one.
    if email and not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
        # Reject email text that does not have a basic address structure.
        raise ValueError("Please enter a valid email address or leave it empty.")

    # Import requests only when feedback is actually being submitted.
    import requests

    # Read the web address of the service that receives feedback.
    url = _setting("GOOGLE_SHEET_URL")
    # Stop when no destination URL has been configured.
    if not url:
        # Explain which setting is missing.
        raise RuntimeError("GOOGLE_SHEET_URL is not configured.")
    # Require the destination to be an HTTPS URL with no whitespace.
    if not re.fullmatch(r"https://[^\s]+", url):
        # Reject unsafe or malformed destination values.
        raise RuntimeError("GOOGLE_SHEET_URL must be a valid HTTPS URL.")

    # Build the JSON data that will be sent to the feedback service.
    payload = {
        # Include the feedback message written by the user.
        "message": message,
        # Include the optional email address.
        "email": email,
        # Include the category selected by the user.
        "category": category,
        # Record the current time in UTC using an unambiguous ISO format.
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    try:
        # Send the feedback data as a JSON POST request and wait up to ten seconds.
        response = requests.post(url, json=payload, timeout=10)
        # Turn HTTP error responses into Python exceptions.
        response.raise_for_status()
    except requests.RequestException as error:
        # Hide low-level network details behind a clear application error.
        raise RuntimeError("The feedback service could not be reached.") from error