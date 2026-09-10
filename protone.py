# Import json for converting evaluation data between Python objects and JSON text.
import json
# Import os for reading the Gemini API key from environment variables.
import os
# Import tempfile for creating a temporary folder that is automatically cleaned up.
import tempfile
# Import Path for safe, cross-platform file-path operations.
from pathlib import Path

# Import Streamlit, which provides the web application interface.
import streamlit as st
# Import Streamlit's component helper for displaying custom HTML and JavaScript.
import streamlit.components.v1 as components
# Import Whisper, which transcribes spoken audio into text.
import whisper
# Import the helper that loads values from a .env file into the environment.
from dotenv import load_dotenv

# Import the function that keeps only the opening words of a transcription.
from extract_hook import extract_hook
# Import the function that sends user feedback to the configured service.
from feedback import save_feedback
# Import the function that measures visual changes between video frames.
from motion_score import calculate_motion_profile
# Import the function that extracts a video's audio into an MP3 file.
from video_to_mp3 import video_to_mp3

# Import Pydantic tools for describing and validating Gemini's structured response.
from pydantic import BaseModel, Field, ValidationError

from card_generator import generate_dm_summary_card

# Allow uploaded videos to be at most 500 megabytes in size.
MAX_UPLOAD_SIZE = 500 * 1024 * 1024

# Describe one scored part of the hook evaluation.
class ScoredMetric(BaseModel):
    # Require a whole-number score from 1 through 10.
    score: int = Field(ge=1, le=10, description="Score from 1 to 10.")
    # Require a short explanation tied to specific words from the hook.
    explanation: str = Field(
        description="Concise 1-2 sentence explanation referencing specific words/phrases from the hook."
    )

# Describe the emotion that the hook is expected to create.
class EmotionalTrigger(BaseModel):
    # Store the name of the main emotion, such as curiosity or humor.
    type: str = Field(
        description="Primary emotion triggered, e.g., 'surprise', 'FOMO', 'fear', 'humor', 'controversy', 'curiosity', or 'none'."
    )
    # Require an emotional-strength score from 1 through 10.
    strength: int = Field(
        ge=1, le=10, 
        description="Intensity of the emotional pull from 1 to 10."
    )

# Describe the evaluation of the video's visual movement.
class VisualPerspective(BaseModel):
    # Allow the visual score to be missing when no motion data was available.
    score: int | None = Field(
        default=None, ge=1, le=10, 
        description="Visual quality/potential score (1-10), or null if no visual data was provided."
    )
    # Require an explanation of the video's visual movement or composition.
    explanation: str = Field(
        description="Brief critique of visual movement, composition, or potential framing based on motion data."
    )

# Describe all of the individual measurements returned for a hook.
class HookMetrics(BaseModel):
    # Measure whether the hook makes the viewer want an answer.
    curiosity_gap: ScoredMetric = Field(
        description="Evaluates how effectively the hook leaves a compelling unanswered question without being cheap clickbait."
    )
    # Measure speaking speed, word density, and unnecessary filler words.
    pacing: ScoredMetric = Field(
        description="Evaluates speech speed, word density, and removal of filler words in the first 3 seconds."
    )
    # Measure whether the hook uses concrete details instead of vague claims.
    specificity: ScoredMetric = Field(
        description="Evaluates whether the hook uses concrete numbers/details vs generic statements."
    )
    # Store the type and strength of the emotional trigger.
    emotional_trigger: EmotionalTrigger
    # Store the structural pattern used by the hook.
    hook_pattern: str = Field(
        description="The structural formula used, e.g., 'shocking statistic', 'provocative question', 'pattern interrupt', 'bold claim'."
    )
    # Measure how clearly the viewer understands the promised benefit.
    payoff_clarity: ScoredMetric = Field(
        description="Evaluates how clearly the viewer understands what value they will gain by staying."
    )

# Describe the complete evaluation returned by the Gemini model.
class HookEvaluation(BaseModel):
    # Require one overall score from 1 through 10.
    score: int = Field(
        ge=1, le=10, 
        description="Overall holistic hook score from 1 to 10. Reflects weighted impact, not a strict average."
    )
    # Include the detailed text-based measurements.
    metrics: HookMetrics
    # Include the separate visual evaluation.
    visual_perspective: VisualPerspective
    # Store two or three short positive observations.
    strengths: list[str] = Field(
        description="2 to 3 short bullet points highlighting what worked. Max 15 words per point."
    )
    # Store two or three short areas for improvement.
    weaknesses: list[str] = Field(
        description="2 to 3 short bullet points highlighting what dragged or failed. Max 15 words per point."
    )
    # Store only the spoken words of the improved hook.
    suggestion: str = Field(
        description="CRITICAL: Output ONLY the exact raw script text for the revised hook. Do NOT add preamble, intro text, quotation marks, or explanations like 'I chose this because...'. Just the spoken words."
    )


# Cache the Whisper model so repeated analyses can reuse it instead of loading it each time.
@st.cache_resource
def load_whisper_model():
    # Load Whisper's base-sized speech-recognition model.
    return whisper.load_model("base")


# Ask Gemini to evaluate the extracted hook and return validated structured data.
def evaluate_hook(hook, api_key, motion_profile=None):
    # Import Gemini libraries only when an evaluation is requested.
    from google import genai
    from google.genai import types

    # Create a Gemini client authenticated with the application's API key.
    client = genai.Client(api_key=api_key)
    # Build instructions that give Gemini the hook, motion data, and scoring rules.
    prompt = f"""
        ROLE & GOAL:
        You are an expert short-form content director (TikTok, Reels, Shorts). 
        Your task is to evaluate the provided video hook and provide a rigorous, honest critique.

        INPUT DATA:
        - Text Hook: "{hook}"
        - Visual Motion Profile: {json.dumps(motion_profile, ensure_ascii=True, allow_nan=False, separators=(",", ":"))}

        EVALUATION GUIDELINES:
        1. Honest Scoring: Do not give default high scores. A score of 5/10 is average. 8+ must be exceptional.
        2. Grounded Explanations: Reference specific words or phrases from the text hook in your metric explanations.
        3. Weighted Overall Score: The overall score is a holistic judgment, not a strict mathematical average of sub-metrics.
        4. Visual Perspective Evaluation:
        - If Visual Motion Profile is null: Set visual_perspective.score to null and explain that visual data was not provided.
        - If Visual Motion Profile is present: The score must reflect your independent qualitative judgment of visual potential/quality. Do NOT copy or threshold the raw motion metrics.
        - Understand motion metrics limitations: High pixel changes can stem from camera shake, lighting shifts, or edits; low motion can be intentional. Mention these nuances in the explanation.
    """

    # Ask Gemini to generate an evaluation using the requested model and schema.
    response = client.models.generate_content(
        # Select the Gemini model used for the evaluation.
        model="gemini-3.5-flash-lite",
        # Send the detailed evaluation instructions as the model's input.
        contents=prompt,
        # Configure the response format and generation limits.
        config=types.GenerateContentConfig(
            # Reinforce the role Gemini should follow while judging the hook.
            system_instruction=(
                "You are a strict, expert short-form video content critic."
            ),
            # Ask Gemini to return JSON rather than free-form prose.
            response_mime_type="application/json",
            # Tell Gemini to shape the JSON according to the Pydantic model.
            response_schema=HookEvaluation,
            # Allow creative but controlled wording in the evaluation.
            temperature=1,
            # Limit how many output tokens Gemini may produce.
            max_output_tokens=2000,
            # Prefer likely words while still allowing some variety.
            top_p=0.95,
        ),
    )

    try:
        # Parse Gemini's JSON and validate every field against HookEvaluation.
        evaluation = HookEvaluation.model_validate_json(response.text)
        
    except ValidationError as e:
        # Show the validation problem in the Streamlit interface.
        st.error(f"Error validating evaluation: {e}")
        # Return no evaluation because Gemini's response did not match the required shape.
        return None

    # Convert the validated Pydantic object back into JSON for storage in session state.
    return evaluation.model_dump_json()


# Display a saved JSON evaluation as readable Streamlit sections.
def show_evaluation(result):
    try:
        # Convert the JSON string into a Python dictionary.
        evaluation = json.loads(result)
    except json.JSONDecodeError:
        # Warn when the stored result is not valid JSON.
        st.warning("Gemini returned an unexpected response format.")
        # Show the raw result so the unexpected response can be inspected.
        st.write(result)
        # Stop because the remaining display code expects a dictionary.
        return

    # Read the overall score from the evaluation dictionary.
    score = evaluation.get("score")
    # Read the detailed metrics, or use an empty dictionary when they are absent.
    metrics = evaluation.get("metrics", {})
    # Display the overall score only when Gemini supplied one.
    if score is not None:
        # Put the overall score inside a bordered Streamlit container.
        with st.container(border=True):
            # Add a heading above the overall number.
            st.subheader("Overall performance")
            # Display the score in Streamlit's metric component.
            st.metric("Hook score", f"{score}/10")

    # Pair each human-readable metric label with its dictionary key.
    scored_metrics = [
        ("Curiosity gap", "curiosity_gap"),
        ("Pacing", "pacing"),
        ("Specificity", "specificity"),
        ("Payoff clarity", "payoff_clarity"),
    ]
    # Keep only metrics that actually exist in the model response.
    available_metrics = [
        # Store the display label together with the metric dictionary.
        (label, metrics.get(key, {}))
        # Examine every expected metric name.
        for label, key in scored_metrics
        # Exclude missing or empty metric values.
        if metrics.get(key)
    ]
    # Draw the metric section only when at least one metric is available.
    if available_metrics:
        # Add a horizontal separator before the next section.
        st.divider()
        # Add a heading for the individual measurements.
        st.subheader("Metric breakdown")
        # Create one equal-width column for each available metric.
        metric_columns = st.columns(len(available_metrics))
        # Pair each display column with its corresponding metric.
        for column, (label, metric) in zip(metric_columns, available_metrics):
            # Make the following Streamlit elements appear inside this column.
            with column:
                # Read the metric's numeric score.
                metric_score = metric.get("score")
                # Display the numeric score when it exists.
                if metric_score is not None:
                    st.metric(label, f"{metric_score}/10")
                # Read the explanation written by Gemini.
                explanation = metric.get("explanation")
                # Display the explanation in smaller caption text when it exists.
                if explanation:
                    st.caption(explanation)


    # Read the emotional-trigger details from the metrics dictionary.
    emotional_trigger = metrics.get("emotional_trigger", {})
    # Read the emotion's name.
    trigger_type = emotional_trigger.get("type")
    # Read the emotion's strength score.
    trigger_strength = emotional_trigger.get("strength")
    # Read the structural pattern used by the hook.
    hook_pattern = metrics.get("hook_pattern")
    # Show the profile section only when at least one profile value exists.
    if trigger_type or trigger_strength is not None or hook_pattern:
        # Separate this section from the previous section.
        st.divider()
        # Add a heading for the hook's emotional and structural profile.
        st.subheader("Hook profile")
        # Create two columns: one for emotion and one for pattern.
        profile_columns = st.columns(2)
        # Put emotional information in the first column.
        with profile_columns[0]:
            # Display the emotional subsection when either emotional value exists.
            if trigger_type or trigger_strength is not None:
                # Label the emotional subsection.
                st.markdown("**Emotional trigger**")
                # Display the emotion name when it exists.
                if trigger_type:
                    st.write(trigger_type)
                # Display the emotional strength when it exists.
                if trigger_strength is not None:
                    st.metric("Trigger strength", f"{trigger_strength}/10")
        # Put the structural pattern in the second column.
        with profile_columns[1]:
            # Display the pattern only when Gemini supplied it.
            if hook_pattern:
                # Label the pattern subsection.
                st.markdown("**Hook pattern**")
                # Display the pattern text.
                st.write(hook_pattern)

    # Read the optional visual evaluation object.
    visual_perspective = evaluation.get("visual_perspective", {})
    # Display visual information when the object exists.
    if visual_perspective:
        # Read the visual score, which may intentionally be null.
        visual_score = visual_perspective.get("score")
        # Read Gemini's explanation of the visual data.
        visual_explanation = visual_perspective.get("explanation")
        # Avoid displaying an empty visual section.
        if visual_score is not None or visual_explanation:
            # Separate this section from the hook profile.
            st.divider()
            # Put the visual results inside a bordered container.
            with st.container(border=True):
                # Add the section heading.
                st.subheader("Visual perspective")
                # Display the visual score when it was available.
                if visual_score is not None:
                    st.metric("Visual hook score", f"{visual_score}/10")
                # Display the visual explanation when it was available.
                if visual_explanation:
                    st.write(visual_explanation)

    # Read the list of positive observations, defaulting to an empty list.
    strengths = evaluation.get("strengths", [])
    # Read the list of problems, defaulting to an empty list.
    weaknesses = evaluation.get("weaknesses", [])
    # Show the insights section when either list contains text.
    if strengths or weaknesses:
        # Separate the insights from the visual section.
        st.divider()
        # Add a heading describing the two types of insights.
        st.subheader("What is working and what to refine")
        # Create one column for strengths and one for weaknesses.
        insight_columns = st.columns(2)
        # Put strengths in the first column.
        with insight_columns[0]:
            # Draw the strengths box only when there are strengths to show.
            if strengths:
                # Put the strengths in a bordered container.
                with st.container(border=True):
                    # Label the first insight list.
                    st.markdown("**Strengths**")
                    # Display each positive observation as a separate line.
                    for strength in strengths:
                        st.write(f"- {strength}")
        # Put weaknesses in the second column.
        with insight_columns[1]:
            # Draw the weaknesses box only when there are weaknesses to show.
            if weaknesses:
                # Put the weaknesses in a bordered container.
                with st.container(border=True):
                    # Label the second insight list.
                    st.markdown("**Weaknesses**")
                    # Display each area for improvement as a separate line.
                    for weakness in weaknesses:
                        st.write(f"- {weakness}")

    # Read the suggested rewritten hook.
    suggestion = evaluation.get("suggestion")
    # Show the suggestion section only when rewritten text exists.
    if suggestion:
        # Separate the suggestion from the insights.
        st.divider()
        # Put the rewritten hook inside a bordered container.
        with st.container(border=True):
            # Add a heading for the rewritten hook.
            st.subheader("Suggested improvement")
            # Give the hook most of the row and reserve a narrow column for Copy.
            suggestion_columns = st.columns([5, 1])
            # Put the rewritten words in the wide column.
            with suggestion_columns[0]:
                st.write(suggestion)
            # Put the copy button in the narrow column.
            with suggestion_columns[1]:
                # Encode the suggestion as safe JSON for JavaScript.
                suggestion_json = json.dumps(suggestion).replace("<", "\\u003c")
                # Render a small HTML button that copies the suggestion to the clipboard.
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
    # Separate the feedback form from the evaluation above it.
    st.divider()
    # Add the feedback form's heading.
    st.subheader("Share your feedback")
    # Explain what kinds of feedback the form accepts.
    st.write("Tell us what worked, what did not, or which feature you would like to see next.")

    # Group the feedback controls into a form so they submit together.
    with st.form("feedback_form", clear_on_submit=True):
        # Let the user choose what kind of feedback they are sending.
        category = st.selectbox(
            "Feedback type",
            ["Bug", "Feature request", "General feedback"],
        )
        # Provide a multi-line field for the feedback message.
        message = st.text_area("Your feedback", placeholder="Write your feedback here...")
        # Provide an optional field for a reply email address.
        email = st.text_input("Email (optional)", placeholder="you@example.com")
        # Add the button that submits the completed form.
        submitted = st.form_submit_button("Send feedback")

    # Process validation and saving only after the form was submitted.
    if submitted:
        # Reject whitespace-only feedback before contacting the external service.
        if not message.strip():
            # Tell the user what needs to be corrected.
            st.warning("Please enter some feedback before sending.")
            # Stop this function before attempting to save invalid input.
            return

        try:
            # Send the category, message, and email to the feedback helper.
            save_feedback(category, message, email)
        except Exception as error:
            # Show a user-friendly error when saving fails.
            st.error(f"Feedback could not be sent: {error}")
        else:
            # Confirm successful submission when no exception occurred.
            st.success("Thanks for your feedback!")


# Convert the uploaded video, transcribe its opening, and evaluate the hook.
def process_video(uploaded_file, api_key):
    # Keep the uploaded file's extension so temporary conversion tools recognize its type.
    suffix = Path(uploaded_file.name).suffix.lower()

    # Create a temporary folder that is deleted automatically after processing finishes.
    with tempfile.TemporaryDirectory() as temporary_directory:
        # Convert the temporary-folder name into a Path object.
        temporary_path = Path(temporary_directory)
        # Choose the temporary path where the uploaded video will be stored.
        video_path = temporary_path / f"input{suffix}"
        # Choose the temporary path where extracted audio will be stored.
        audio_path = temporary_path / "audio.mp3"
        # Write the uploaded video bytes to the temporary video file.
        video_path.write_bytes(uploaded_file.getbuffer())
        # Measure motion in the first three seconds using ten samples per second.
        motion_profile = calculate_motion_profile(
            video_path,
            max_seconds=3,
            sample_rate=10,
        )

        # Show a progress area while each processing stage runs.
        with st.status("Processing video...", expanded=True) as status:
            # Tell the user that audio extraction is beginning.
            st.write("Extracting audio")
            # Use FFmpeg to create an MP3 from the video's audio track.
            video_to_mp3(video_path, audio_path)

            # Tell the user that speech transcription is beginning.
            st.write("Transcribing audio")
            # Load the cached Whisper speech-recognition model.
            model = load_whisper_model()
            # Transcribe the extracted MP3 and return segment timestamps and text.
            transcription = model.transcribe(str(audio_path))
            # Keep only speech segments that begin during the first three seconds.
            hook = extract_hook(transcription, max_seconds=3)

            # Stop processing when the opening contains no recognized speech.
            if not hook:
                # Mark the visible progress status as failed.
                status.update(label="No speech found", state="error")
                # Raise an error so the caller can display the reason to the user.
                raise ValueError("No speech was detected in the first three seconds.")

            # Tell the user that the language-model evaluation is beginning.
            st.write("Evaluating hook")
            # Ask Gemini to evaluate the spoken hook and the visual motion profile.
            evaluation = evaluate_hook(hook, api_key, motion_profile)
            # Mark all processing steps as complete.
            status.update(label="Analysis complete", state="complete")

    # Return the JSON evaluation after the temporary files have been cleaned up.
    return evaluation


# Configure the Streamlit page and coordinate the complete user workflow.
def main():
    # Load variables from a local .env file into the process environment.
    load_dotenv()
    # Set the browser-tab title and the small page icon.
    st.set_page_config(page_title="Hook Checker", page_icon="🎬")
    # Display the app's main heading.
    st.title("Hook Checker")
    # Explain the app's purpose below the heading.
    st.write("Upload a video to extract and evaluate its opening hook.")

    # Display a file picker restricted to MP4 and MOV videos.
    uploaded_file = st.file_uploader(
        "Choose a video",
        type=["mp4", "mov"],
        help="Supported formats: MP4 and MOV. Maximum size: 200 MB.",
    )

    # Clear old results and stop when no video is currently selected.
    if uploaded_file is None:
        # Remove any previous evaluation from Streamlit's session state.
        st.session_state.pop("evaluation", None)
        # Remove the previous file identity from Streamlit's session state.
        st.session_state.pop("uploaded_signature", None)
        # Tell the user what kind of action starts the workflow.
        st.info("Select an MP4 or MOV video to get started.")
        # Leave main because there is no file to analyze yet.
        return

    # Identify the selected upload by both its filename and its byte size.
    uploaded_signature = (uploaded_file.name, uploaded_file.size)
    # Reset old results when the selected upload is different from the previous one.
    if st.session_state.get("uploaded_signature") != uploaded_signature:
        # Remove the evaluation belonging to the previous upload.
        st.session_state.pop("evaluation", None)
        # Remember the current upload for the next rerun of the Streamlit script.
        st.session_state["uploaded_signature"] = uploaded_signature

    # Reject files larger than the application's configured maximum.
    if uploaded_file.size > MAX_UPLOAD_SIZE:
        # Explain why the selected video cannot be processed.
        st.error("The video is larger than the 500 MB limit.")
        # Stop before reading or processing an oversized file.
        return

    # Read the Gemini API key from the environment or .env file.
    api_key = os.getenv("GEMINI_API_KEY")
    # Stop when Gemini cannot be authenticated.
    if not api_key:
        # Explain where the missing key must be configured.
        st.error("GEMINI_API_KEY is missing from the environment or .env file.")
        # Stop before displaying an unusable Analyze button workflow.
        return

    # Begin analysis only after the user presses the primary action button.
    if st.button("Analyze video", type="primary"):
        try:
            # Process the uploaded file and save the JSON result in session state.
            st.session_state["evaluation"] = process_video(uploaded_file, api_key)
        except Exception as error:
            # Remove any incomplete result when processing fails.
            st.session_state.pop("evaluation", None)
            # Show the error message in the app.
            st.error(str(error))

    # Read the most recent successful evaluation from session state.
    evaluation = st.session_state.get("evaluation")
    # Display the evaluation and feedback form when an evaluation exists.
    if evaluation:
        # Render all evaluation sections.
        show_evaluation(evaluation)
        # Render the form that lets the user submit feedback.
        render_feedback()


# Run the Streamlit entry point only when this file is executed directly.
if __name__ == "__main__":
    # Start the application workflow.
    main()