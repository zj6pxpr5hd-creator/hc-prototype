# 1 extract text

print("Script started")

import whisper
print("Whisper imported successfully")

model = whisper.load_model("base")
print("Model loaded")

transcription = model.transcribe("test.mp3")
print("Transcription complete")

# 2 extract hook

from extract_hook import extract_hook

hook = extract_hook(transcription, max_seconds=3)
print(f"Extracted hook: {hook}")

# 3 use gemini to evaluate hook

import os
from dotenv import load_dotenv

print("Loading environment variables from .env file")

load_dotenv()  # reads .env and loads variables into the environment

api_key = os.environ["GEMINI_API_KEY"]

from google import genai

client = genai.Client(
    api_key=os.environ.get("GEMINI_API_KEY"),
)

print("Client initialized successfully")

transcription_text = transcription["text"]

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

generation_config = {
    'temperature': 1,
    'max_output_tokens': 1000,
    'top_p': 0.95,
    'thinking_level': 'low',
}

interaction = client.interactions.create(
    model='models/gemini-3-flash-preview',
    system_instruction="You are a strict, expert short-form video content critic.",
    input=prompt,
    generation_config=generation_config,
)

print(interaction.output_text)