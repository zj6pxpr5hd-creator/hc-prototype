import json
import os
import tempfile
from pathlib import Path

import streamlit as st
import whisper
from dotenv import load_dotenv

from extract_hook import extract_hook
from feedback import save_feedback
from video_to_mp3 import video_to_mp3


MAX_UPLOAD_SIZE = 500 * 1024 * 1024


@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")


def evaluate_hook(hook, api_key):
    from google import genai

    client = genai.Client(api_key=api_key)
    prompt = f"""
CONTEXT:
You are evaluating short-form video hooks (the first few seconds of a TikTok/Reels/YouTube Short) for how well they grab attention.

TASK:
Review the following hook and judge how effective it is at stopping someone from scrolling.

OUTPUT FORMAT:
Return ONLY valid JSON in this exact structure:
{{
  "score": <number from 1 to 10>,
  "strengths": ["...", "..."],
  "weaknesses": ["...", "..."],
  "suggestion": "one improved version of the hook"
}}

RULES:
- Be honest and critical, don't just give high scores by default.
- Do not include any text outside the JSON.

HOOK TO REVIEW:
"{hook}"
"""

    interaction = client.interactions.create(
        model="models/gemini-3.7-flash",
        system_instruction="You are a strict, expert short-form video content critic.",
        input=prompt,
        generation_config={
            "temperature": 1,
            "max_output_tokens": 1000,
            "top_p": 0.95,
            "thinking_level": "low",
        },
    )
    return interaction.output_text


def show_evaluation(result):
    try:
        evaluation = json.loads(result)
    except json.JSONDecodeError:
        st.warning("Gemini returned an unexpected response format.")
        st.write(result)
        return

    score = evaluation.get("score")
    if score is not None:
        st.metric("Hook score", f"{score}/10")

    strengths = evaluation.get("strengths", [])
    if strengths:
        st.subheader("Strengths")
        for strength in strengths:
            st.write(f"- {strength}")

    weaknesses = evaluation.get("weaknesses", [])
    if weaknesses:
        st.subheader("Weaknesses")
        for weakness in weaknesses:
            st.write(f"- {weakness}")

    suggestion = evaluation.get("suggestion")
    if suggestion:
        st.subheader("Suggested improvement")
        st.write(suggestion)


def render_feedback():
    st.divider()
    st.subheader("Share your feedback")
    st.write("Tell us what worked, what did not, or which feature you would like to see next.")

    with st.form("feedback_form", clear_on_submit=True):
        category = st.selectbox(
            "Feedback type",
            ["Bug", "Feature request", "General feedback"],
        )
        message = st.text_area("Your feedback", placeholder="Write your feedback here...")
        email = st.text_input("Email (optional)", placeholder="you@example.com")
        submitted = st.form_submit_button("Send feedback")

    if submitted:
        if not message.strip():
            st.warning("Please enter some feedback before sending.")
            return

        try:
            save_feedback(category, message, email)
        except Exception as error:
            st.error(f"Feedback could not be sent: {error}")
        else:
            st.success("Thanks for your feedback!")


def process_video(uploaded_file, api_key):
    suffix = Path(uploaded_file.name).suffix.lower()

    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary_path = Path(temporary_directory)
        video_path = temporary_path / f"input{suffix}"
        audio_path = temporary_path / "audio.mp3"
        video_path.write_bytes(uploaded_file.getbuffer())

        with st.status("Processing video...", expanded=True) as status:
            st.write("Extracting audio")
            video_to_mp3(video_path, audio_path)

            st.write("Transcribing audio")
            model = load_whisper_model()
            transcription = model.transcribe(str(audio_path))
            hook = extract_hook(transcription, max_seconds=3)

            if not hook:
                status.update(label="No speech found", state="error")
                raise ValueError("No speech was detected in the first three seconds.")

            st.write("Evaluating hook")
            evaluation = evaluate_hook(hook, api_key)
            status.update(label="Analysis complete", state="complete")

    return evaluation


def main():
    load_dotenv()
    st.set_page_config(page_title="Hook Checker", page_icon="🎬")
    st.title("Hook Checker")
    st.write("Upload a video to extract and evaluate its opening hook.")

    uploaded_file = st.file_uploader(
        "Choose a video",
        type=["mp4", "mov"],
        help="Supported formats: MP4 and MOV. Maximum size: 500 MB.",
    )

    if uploaded_file is None:
        st.session_state.pop("evaluation", None)
        st.session_state.pop("uploaded_signature", None)
        st.info("Select an MP4 or MOV video to get started.")
        return

    uploaded_signature = (uploaded_file.name, uploaded_file.size)
    if st.session_state.get("uploaded_signature") != uploaded_signature:
        st.session_state.pop("evaluation", None)
        st.session_state["uploaded_signature"] = uploaded_signature

    if uploaded_file.size > MAX_UPLOAD_SIZE:
        st.error("The video is larger than the 500 MB limit.")
        return

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("GEMINI_API_KEY is missing from the environment or .env file.")
        return

    if st.button("Analyze video", type="primary"):
        try:
            st.session_state["evaluation"] = process_video(uploaded_file, api_key)
        except Exception as error:
            st.session_state.pop("evaluation", None)
            st.error(str(error))

    evaluation = st.session_state.get("evaluation")
    if evaluation:
        show_evaluation(evaluation)
        render_feedback()


if __name__ == "__main__":
    main()