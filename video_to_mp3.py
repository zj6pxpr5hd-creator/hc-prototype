from pathlib import Path
import shutil
import subprocess

def video_to_mp3(video_path, output_path):
    input_path = Path(video_path)
    output_path = Path(output_path)

    if not input_path.is_file():
        raise FileNotFoundError(f"Video file not found: {input_path}")

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError(
            "FFmpeg is not installed or is not available in the system PATH."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(input_path),
        "-map", "0:a:0",
        "-vn",
        "-c:a", "libmp3lame",
        "-q:a", "2",
        "-y",
        str(output_path),
    ]

    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("FFmpeg timed out while extracting the audio.") from error

    if result.returncode != 0:
        details = result.stderr.strip() or "Unknown FFmpeg error."
        raise RuntimeError(f"Could not extract audio from the video: {details}")

    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError("FFmpeg completed without creating a valid MP3 file.")

    return output_path
