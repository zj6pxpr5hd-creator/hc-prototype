# Import math for number validation, rounding, and percentile calculation.
import math
# Import Path so callers may provide either strings or Path objects.
from pathlib import Path

# Analyze at most the first three seconds unless the caller chooses another value.
DEFAULT_MAX_SECONDS = 3
# Examine about ten frames per second unless the caller chooses another rate.
DEFAULT_SAMPLE_RATE = 10
# Resize wide frames to this width to reduce the amount of work OpenCV must do.
MAX_FRAME_WIDTH = 320
# Treat a normalized frame difference at or above this value as a large change.
HIGH_CHANGE_THRESHOLD = 0.10


def calculate_motion_profile(
    video_path: str | Path,
    max_seconds: float = DEFAULT_MAX_SECONDS,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> dict[str, float | int | str] | None:
    """Return a JSON-friendly profile of grayscale changes in the opening video."""
    # Import OpenCV only when this function is used, so the rest of the app can run without it.
    try:
        import cv2
    except ImportError:
        # Return no profile when the optional OpenCV package is unavailable.
        return None

    # Reject impossible timing or sampling settings before opening the video.
    if (
        not math.isfinite(max_seconds)
        or max_seconds <= 0
        or sample_rate <= 0
    ):
        # Tell the caller that no valid motion analysis could be calculated.
        return None

    # Ask OpenCV to open the supplied video file.
    capture = cv2.VideoCapture(str(video_path))
    # Check that OpenCV successfully opened the file.
    if not capture.isOpened():
        # Release any resources OpenCV allocated before returning.
        capture.release()
        # Tell the caller that the video could not be read.
        return None

    # Read the video's frames-per-second value from the video metadata.
    frames_per_second = capture.get(cv2.CAP_PROP_FPS)
    # Use a reasonable fallback when the metadata is missing or invalid.
    if not math.isfinite(frames_per_second) or frames_per_second <= 0:
        frames_per_second = 30
    # Calculate how many frames to skip between sampled frames.
    sample_interval = max(1, math.ceil(frames_per_second / sample_rate))
    # Track the number of the frame currently being processed.
    frame_index = 0
    # Store the previous sampled grayscale frame for comparison with the next one.
    previous_frame = None
    # Store each measured change between two sampled frames.
    differences = []

    try:
        # Continue reading frames until the file ends or the requested time is reached.
        while True:
            # Read one frame and receive both success status and image data.
            success, frame = capture.read()
            # Stop when OpenCV cannot read another frame.
            if not success:
                break

            # Convert the frame number into the video's elapsed time in seconds.
            frame_time = frame_index / frames_per_second
            # Stop after the requested opening portion of the video.
            if frame_time >= max_seconds:
                break
            # Skip frames that are not part of the requested sampling interval.
            if frame_index % sample_interval != 0:
                # Advance the frame counter before trying the next frame.
                frame_index += 1
                continue

            # Convert the color image to grayscale so brightness changes can be compared.
            grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Reduce very wide images to save processing time while preserving proportions.
            if grayscale.shape[1] > MAX_FRAME_WIDTH:
                # Calculate the new height that keeps the original image proportions.
                target_height = round(
                    grayscale.shape[0] * MAX_FRAME_WIDTH / grayscale.shape[1]
                )
                # Resize the grayscale frame using area averaging for good reduction quality.
                grayscale = cv2.resize(
                    grayscale,
                    (MAX_FRAME_WIDTH, target_height),
                    interpolation=cv2.INTER_AREA,
                )

            # A difference can be measured only after a previous frame exists.
            if previous_frame is not None:
                # Find the absolute brightness difference at every pixel.
                difference = cv2.absdiff(previous_frame, grayscale)
                # Store the average difference normalized from 0.0 to 1.0.
                differences.append(float(difference.mean()) / 255)

            # Keep this frame so it can be compared with the next sampled frame.
            previous_frame = grayscale
            # Advance the frame counter before reading the next frame.
            frame_index += 1
    except cv2.error:
        # Return no result if OpenCV reports a processing error.
        return None
    finally:
        # Always close the video resource, including when an error occurs.
        capture.release()

    # A profile is impossible when fewer than two usable frames were sampled.
    if not differences:
        return None

    # Sort the changes so percentile values can be calculated by position.
    ordered_differences = sorted(differences)
    # Locate the 95th-percentile position, which represents a high but typical change.
    percentile_index = (len(ordered_differences) - 1) * 0.95
    # Find the lower whole-number index around that percentile position.
    lower_index = math.floor(percentile_index)
    # Find the upper whole-number index around that percentile position.
    upper_index = math.ceil(percentile_index)
    # Calculate how far the percentile lies between the lower and upper values.
    fraction = percentile_index - lower_index
    # Interpolate between the two neighboring values to get a smooth percentile estimate.
    p95_change = ordered_differences[lower_index] + (
        ordered_differences[upper_index] - ordered_differences[lower_index]
    ) * fraction
    # Count how many frame comparisons meet the high-change threshold.
    high_change_count = sum(
        difference >= HIGH_CHANGE_THRESHOLD for difference in differences
    )
    # Count the total number of frame comparisons used in the analysis.
    comparison_count = len(differences)
    # Convert the high-change count into a fraction of all comparisons.
    high_change_ratio = high_change_count / comparison_count
    # Calculate the average normalized change across all comparisons.
    average_change = sum(differences) / comparison_count
    # Find the single largest normalized change.
    peak_change = max(differences)

    # Describe whether large changes happen often, once, or hardly at all.
    if high_change_ratio >= 0.2:
        # At least one fifth of comparisons show large movement changes.
        interpretation = "recurring high-change transitions"
    elif peak_change >= HIGH_CHANGE_THRESHOLD:
        # At least one comparison is large, but large changes are not recurring.
        interpretation = "mostly steady with isolated high-change transitions"
    else:
        # No comparison reaches the high-change threshold.
        interpretation = "mostly steady frame-to-frame change"

    # Return numeric results rounded enough to be easy to read and send as JSON.
    return {
        # Report the average amount of image change.
        "average_change": round(average_change, 6),
        # Report the largest amount of image change.
        "peak_change": round(peak_change, 6),
        # Report the 95th-percentile amount of image change.
        "p95_change": round(p95_change, 6),
        # Report the number of comparisons that were unusually different.
        "high_change_count": high_change_count,
        # Report how many comparisons were performed in total.
        "comparison_count": comparison_count,
        # Report the fraction of comparisons with unusually large changes.
        "high_change_ratio": round(high_change_ratio, 6),
        # Report the plain-English interpretation selected above.
        "interpretation": interpretation,
    }


def calculate_motion_score(
    video_path: str | Path,
    max_seconds: float = DEFAULT_MAX_SECONDS,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> float | None:
    """Return the profile's average change for compatibility with old callers."""
    # Calculate the complete motion profile using the same settings.
    profile = calculate_motion_profile(video_path, max_seconds, sample_rate)
    # Preserve the old function's behavior when no profile can be calculated.
    if profile is None:
        return None
    # Return only the average change expected by older code.
    return float(profile["average_change"])
