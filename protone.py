import json
import os
import tempfile
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
import whisper
from dotenv import load_dotenv

from extract_hook import extract_hook
from feedback import save_feedback
from motion_score import calculate_motion_profile
from video_to_mp3 import video_to_mp3


MAX_UPLOAD_SIZE = 500 * 1024 * 1024


@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")


def evaluate_hook(hook, api_key, motion_profile=None):
    from google import genai

    client = genai.Client(api_key=api_key)
    prompt = f"""
CONTEXT:
You are evaluating short-form video hooks (the first few seconds of a TikTok/Reels/YouTube Short) for how well they grab attention.

TASK:
Review the following hook and judge how effective it is at stopping someone from scrolling. Provide both an overall score and a breakdown of objective sub-metrics that explain WHY the score is what it is.

OUTPUT FORMAT:
Return ONLY valid JSON in this exact structure:
{{
  "score": <number from 1 to 10>,
  "metrics": {{
    "curiosity_gap": {{
      "score": <1-10>,
      "explanation": "..."
    }},
    "pacing": {{
      "score": <1-10>,
      "explanation": "is the first idea delivered fast enough, or is it buried under filler words/slow setup?"
    }},
    "specificity": {{
      "score": <1-10>,
      "explanation": "is the hook concrete and specific, or vague/generic?"
    }},
    "emotional_trigger": {{
      "type": "e.g. surprise, FOMO, fear, humor, controversy, none",
      "strength": <1-10>
    }},
    "hook_pattern": "e.g. shocking statistic, provocative question, pattern interrupt, bold claim, in-medias-res story, none detected",
    "payoff_clarity": {{
      "score": <1-10>,
      "explanation": "does the hook clearly imply what the viewer will get if they keep watching?"
    }}
  }},
    "visual_perspective": {{
        "score": <number from 1 to 10, or null if visual context is unavailable>,
        "explanation": "..."
    }},
  "strengths": ["...", "..."],
  "weaknesses": ["...", "..."],
  "suggestion": "one improved version of the hook"
}}

RULES:
- Be honest and critical, don't just give high scores by default.
- Each metric score must be justified in the explanation field, referencing specific words/phrases from the hook when possible.
- The overall "score" should reflect a weighted judgment across the metrics, not just an average — explain implicitly through strengths/weaknesses if one metric dominates.
- Treat "visual_perspective.score" as an independent 1-10 judgment of visual hook quality, not as a conversion of the motion profile. Never copy, average, normalize, or threshold the profile into that score.
- The motion profile measures normalized mean absolute grayscale pixel differences between sampled frames. It measures temporal pixel change only; it cannot establish composition, clarity, readability, semantic relevance, visual appeal, or production quality.
- High change can come from cuts, camera shake, exposure changes, compression artifacts, or noise. Low change can be intentional and effective. Use the average, p95, peak, and high-change ratio to distinguish sustained movement from isolated changes.
- If the motion profile is null, set "visual_perspective.score" to null and explain that visual analysis is unavailable. If it is present, mention its limitations rather than treating it as a quality verdict.
- Do not include any text outside the JSON.

VISUAL MOTION PROFILE (JSON MEASUREMENTS ONLY):
{json.dumps(motion_profile, ensure_ascii=True, allow_nan=False, separators=(",", ":"))}

HOOK TO REVIEW:
"{hook}"
"""

    interaction = client.interactions.create(
        model="models/gemini-3.6-flash",
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
    metrics = evaluation.get("metrics", {})
    if score is not None:
        with st.container(border=True):
            st.subheader("Overall performance")
            st.metric("Hook score", f"{score}/10")

    scored_metrics = [
        ("Curiosity gap", "curiosity_gap"),
        ("Pacing", "pacing"),
        ("Specificity", "specificity"),
        ("Payoff clarity", "payoff_clarity"),
    ]
    available_metrics = [
        (label, metrics.get(key, {}))
        for label, key in scored_metrics
        if metrics.get(key)
    ]
    if available_metrics:
        st.divider()
        st.subheader("Metric breakdown")
        metric_columns = st.columns(len(available_metrics))
        for column, (label, metric) in zip(metric_columns, available_metrics):
            with column:
                metric_score = metric.get("score")
                if metric_score is not None:
                    st.metric(label, f"{metric_score}/10")
                explanation = metric.get("explanation")
                if explanation:
                    st.caption(explanation)

    emotional_trigger = metrics.get("emotional_trigger", {})
    trigger_type = emotional_trigger.get("type")
    trigger_strength = emotional_trigger.get("strength")
    hook_pattern = metrics.get("hook_pattern")
    if trigger_type or trigger_strength is not None or hook_pattern:
        st.divider()
        st.subheader("Hook profile")
        profile_columns = st.columns(2)
        with profile_columns[0]:
            if trigger_type or trigger_strength is not None:
                st.markdown("**Emotional trigger**")
                if trigger_type:
                    st.write(trigger_type)
                if trigger_strength is not None:
                    st.metric("Trigger strength", f"{trigger_strength}/10")
        with profile_columns[1]:
            if hook_pattern:
                st.markdown("**Hook pattern**")
                st.write(hook_pattern)

    visual_perspective = evaluation.get("visual_perspective", {})
    if visual_perspective:
        visual_score = visual_perspective.get("score")
        visual_explanation = visual_perspective.get("explanation")
        if visual_score is not None or visual_explanation:
            st.divider()
            with st.container(border=True):
                st.subheader("Visual perspective")
                if visual_score is not None:
                    st.metric("Visual hook score", f"{visual_score}/10")
                if visual_explanation:
                    st.write(visual_explanation)

    strengths = evaluation.get("strengths", [])
    weaknesses = evaluation.get("weaknesses", [])
    if strengths or weaknesses:
        st.divider()
        st.subheader("What is working and what to refine")
        insight_columns = st.columns(2)
        with insight_columns[0]:
            if strengths:
                with st.container(border=True):
                    st.markdown("**Strengths**")
                    for strength in strengths:
                        st.write(f"- {strength}")
        with insight_columns[1]:
            if weaknesses:
                with st.container(border=True):
                    st.markdown("**Weaknesses**")
                    for weakness in weaknesses:
                        st.write(f"- {weakness}")

    suggestion = evaluation.get("suggestion")
    if suggestion:
        st.divider()
        with st.container(border=True):
            st.subheader("Suggested improvement")
            suggestion_columns = st.columns([5, 1])
            with suggestion_columns[0]:
                st.write(suggestion)
            with suggestion_columns[1]:
                suggestion_json = json.dumps(suggestion).replace("<", "\\u003c")
                components.html(
                    f"""
                    <button
                        onclick='copySuggestion()'
                        style="width:100%; padding:0.5rem; cursor:pointer;"
                    >
                        Copy
                    </button>
                    <script>
                        const suggestion = {suggestion_json};

                        async function copySuggestion() {{
                            const button = document.querySelector("button");
                            try {{
                                await navigator.clipboard.writeText(suggestion);
                                button.textContent = "Copied";
                            }} catch (error) {{
                                button.textContent = "Copy failed";
                            }}
                        }}
                    </script>
                    """,
                    height=48,
                )


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
        motion_profile = calculate_motion_profile(
            video_path,
            max_seconds=3,
            sample_rate=10,
        )

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
            evaluation = evaluate_hook(hook, api_key, motion_profile)
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
        help="Supported formats: MP4 and MOV. Maximum size: 200 MB.",
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