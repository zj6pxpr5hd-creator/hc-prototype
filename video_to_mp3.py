import subprocess
import os

def video_to_mp3(video_path, output_path="audio.mp3"):
    command = [
        "ffmpeg",
        "-i", video_path,      # input file
        "-vn",                  # no video
        "-acodec", "libmp3lame",  # mp3 encoder
        "-y",                    # overwrite output if it exists
        output_path
    ]
    subprocess.run(command, check=True)
    return output_path
