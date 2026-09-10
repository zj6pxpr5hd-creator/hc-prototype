# Read the spoken-text segments and return only the words spoken near the start of the video.
def extract_hook(transcription, max_seconds=3):
    # Start with an empty string so text can be added to it one segment at a time.
    hook_text = ""
    # Check each transcript segment in the order in which it was spoken.
    for segment in transcription["segments"]:
        # Keep this segment when it starts before the allowed opening time.
        if segment["start"] < max_seconds:
            # Add the segment's spoken words to the growing hook text.
            hook_text += segment["text"]
        else:
            # Stop checking because later segments start even farther into the video.
            break
    # Remove extra spaces or newline characters from the beginning and end of the result.
    return hook_text.strip()

