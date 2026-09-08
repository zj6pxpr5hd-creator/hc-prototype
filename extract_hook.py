def extract_hook(transcription, max_seconds=3):
    hook_text = ""
    for segment in transcription["segments"]:
        if segment["start"] < max_seconds:
            hook_text += segment["text"]
        else:
            break
    return hook_text.strip()

