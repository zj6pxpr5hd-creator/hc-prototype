# Import Path so file paths work consistently on macOS and other operating systems.
from pathlib import Path
# Import shutil so the program can search for the FFmpeg executable.
import shutil
# Import subprocess so Python can run FFmpeg as an external command-line program.
import subprocess

# Convert the audio track from a video file into an MP3 file.
def video_to_mp3(video_path, output_path):
    # Turn the input value into a Path object so file checks are easy to perform.
    input_path = Path(video_path)
    # Turn the output value into a Path object so its parent directory can be created.
    output_path = Path(output_path)

    # Refuse to continue when the requested input video does not exist as a file.
    if not input_path.is_file():
        # Explain which missing file caused the problem.
        raise FileNotFoundError(f"Video file not found: {input_path}")

    # Find the FFmpeg program by searching the folders listed in the system PATH.
    ffmpeg_path = shutil.which("ffmpeg")
    # Stop with a useful message when FFmpeg cannot be found.
    if ffmpeg_path is None:
        # Tell the user that FFmpeg must be installed and discoverable by the system.
        raise RuntimeError(
            "FFmpeg is not installed or is not available in the system PATH."
        )

    # Create the output folder when it does not already exist.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Build the FFmpeg command as a list, with one list item for each command argument.
    command = [
        # Run the FFmpeg executable that was found above.
        ffmpeg_path,
        # Hide FFmpeg's decorative startup banner.
        "-hide_banner",
        # Show only actual errors instead of normal progress messages.
        "-loglevel", "error",
        # Tell FFmpeg which video file should be read.
        "-i", str(input_path),
        # Select the first audio stream from the input video.
        "-map", "0:a:0",
        # Do not copy video because the requested output contains audio only.
        "-vn",
        # Encode the audio using the MP3 encoder.
        "-c:a", "libmp3lame",
        # Ask the MP3 encoder for a high-quality variable-bitrate result.
        "-q:a", "2",
        # Replace an existing output file without asking for confirmation.
        "-y",
        # Tell FFmpeg where to write the converted audio file.
        str(output_path),
    ]

    try:
        # Run FFmpeg and wait for it to finish, while collecting its output.
        result = subprocess.run(
            # Pass the command and its arguments to the operating system.
            command,
            # Let this function inspect the exit code instead of raising automatically.
            check=False,
            # Capture normal output and error output so they can be handled below.
            capture_output=True,
            # Decode FFmpeg's output into text instead of returning raw bytes.
            text=True,
            # Prevent a broken or stalled conversion from running forever.
            timeout=300,
        )
    except subprocess.TimeoutExpired as error:
        # Convert Python's timeout exception into a simpler application-level message.
        raise RuntimeError("FFmpeg timed out while extracting the audio.") from error

    # FFmpeg uses a non-zero return code when conversion fails.
    if result.returncode != 0:
        # Prefer FFmpeg's detailed error message, but provide a fallback if it was empty.
        details = result.stderr.strip() or "Unknown FFmpeg error."
        # Raise an error that includes the reason conversion failed.
        raise RuntimeError(f"Could not extract audio from the video: {details}")

    # Confirm that FFmpeg created a non-empty output file.
    if not output_path.is_file() or output_path.stat().st_size == 0:
        # Report a failure when the file is missing or empty.
        raise RuntimeError("FFmpeg completed without creating a valid MP3 file.")

    # Give the caller the completed output path so it can use the MP3 file.
    return output_path
