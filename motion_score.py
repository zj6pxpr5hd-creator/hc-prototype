import math
from pathlib import Path

DEFAULT_MAX_SECONDS = 3
DEFAULT_SAMPLE_RATE = 10
MAX_FRAME_WIDTH = 320
HIGH_CHANGE_THRESHOLD = 0.10


def calculate_motion_profile(
    video_path: str | Path,
    max_seconds: float = DEFAULT_MAX_SECONDS,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> dict[str, float | int | str] | None:
    """Return a JSON-friendly profile of grayscale changes in the opening video."""
    try:
        import cv2
    except ImportError:
        return None

    if (
        not math.isfinite(max_seconds)
        or max_seconds <= 0
        or sample_rate <= 0
    ):
        return None

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        return None

    frames_per_second = capture.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(frames_per_second) or frames_per_second <= 0:
        frames_per_second = 30
    sample_interval = max(1, math.ceil(frames_per_second / sample_rate))
    frame_index = 0
    previous_frame = None
    differences = []

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            frame_time = frame_index / frames_per_second
            if frame_time >= max_seconds:
                break
            if frame_index % sample_interval != 0:
                frame_index += 1
                continue

            grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if grayscale.shape[1] > MAX_FRAME_WIDTH:
                target_height = round(
                    grayscale.shape[0] * MAX_FRAME_WIDTH / grayscale.shape[1]
                )
                grayscale = cv2.resize(
                    grayscale,
                    (MAX_FRAME_WIDTH, target_height),
                    interpolation=cv2.INTER_AREA,
                )

            if previous_frame is not None:
                difference = cv2.absdiff(previous_frame, grayscale)
                differences.append(float(difference.mean()) / 255)

            previous_frame = grayscale
            frame_index += 1
    except cv2.error:
        return None
    finally:
        capture.release()

    if not differences:
        return None

    ordered_differences = sorted(differences)
    percentile_index = (len(ordered_differences) - 1) * 0.95
    lower_index = math.floor(percentile_index)
    upper_index = math.ceil(percentile_index)
    fraction = percentile_index - lower_index
    p95_change = ordered_differences[lower_index] + (
        ordered_differences[upper_index] - ordered_differences[lower_index]
    ) * fraction
    high_change_count = sum(
        difference >= HIGH_CHANGE_THRESHOLD for difference in differences
    )
    comparison_count = len(differences)
    high_change_ratio = high_change_count / comparison_count
    average_change = sum(differences) / comparison_count
    peak_change = max(differences)

    if high_change_ratio >= 0.2:
        interpretation = "recurring high-change transitions"
    elif peak_change >= HIGH_CHANGE_THRESHOLD:
        interpretation = "mostly steady with isolated high-change transitions"
    else:
        interpretation = "mostly steady frame-to-frame change"

    return {
        "average_change": round(average_change, 6),
        "peak_change": round(peak_change, 6),
        "p95_change": round(p95_change, 6),
        "high_change_count": high_change_count,
        "comparison_count": comparison_count,
        "high_change_ratio": round(high_change_ratio, 6),
        "interpretation": interpretation,
    }


def calculate_motion_score(
    video_path: str | Path,
    max_seconds: float = DEFAULT_MAX_SECONDS,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> float | None:
    """Return the profile's average change for compatibility with old callers."""
    profile = calculate_motion_profile(video_path, max_seconds, sample_rate)
    if profile is None:
        return None
    return float(profile["average_change"])
