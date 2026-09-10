# Import Pillow classes used to create an image and draw text and shapes on it.
from PIL import Image, ImageDraw, ImageFont
# Import BytesIO so the finished image can be kept in memory instead of saved to disk.
import io

# Build a shareable image summarizing the video's analysis results.
def generate_dm_summary_card(motion_score: int, pacing_score: str, rewrite_text: str) -> io.BytesIO:
    """
    Generates a 1200x630 dark-mode summary card optimized for DM attachments.
    Returns a BytesIO buffer suitable for st.download_button.
    """
    # Set the image size to 1200 by 630 pixels, a common landscape sharing size.
    width, height = 1200, 630
    # Create a dark RGB image that will act as the card's background.
    card = Image.new("RGB", (width, height), color="#0E1117")
    # Create a drawing tool that can place text and shapes on the image.
    draw = ImageDraw.Draw(card)

    # Try to load readable font files for the title and body text.
    try:
        # Load a bold font for the largest title text.
        font_title = ImageFont.truetype("DejaVuSans-Bold.ttf", 42)
        # Load a smaller bold font for section headings.
        font_header = ImageFont.truetype("DejaVuSans-Bold.ttf", 28)
        # Load a normal font for labels and explanations.
        font_body = ImageFont.truetype("DejaVuSans.ttf", 24)
    except IOError:
        # Use Pillow's built-in font when the requested font files are unavailable.
        font_title = font_header = font_body = ImageFont.load_default()

    # Draw the card title near the upper-left corner in bright green.
    draw.text((60, 50), "⚡ HOOK ANALYSIS BREAKDOWN", fill="#00FFA3", font=font_title)
    # Draw a horizontal line beneath the title to separate the header from the metrics.
    draw.line([(60, 110), (1140, 110)], fill="#333333", width=2)

    # Draw the left box that will contain the visual-motion score.
    draw.rectangle([(60, 140), (570, 260)], fill="#1E222A", outline="#333333", width=2)
    # Label the left box so readers know what its number represents.
    draw.text((80, 160), "VISUAL MOTION (OpenCV)", fill="#888888", font=font_body)
    
    # Choose green for a strong score, red for a weak score, and orange otherwise.
    motion_color = "#00FFA3" if motion_score >= 7 else "#FF4B4B" if motion_score <= 4 else "#FFAA00"
    # Draw the motion score using the color selected above.
    draw.text((80, 195), f"{motion_score}/10", fill=motion_color, font=font_title)

    # Draw the right box that will contain the speech-pacing score.
    draw.rectangle([(630, 140), (1140, 260)], fill="#1E222A", outline="#333333", width=2)
    # Label the right box so readers know where this score came from.
    draw.text((650, 160), "SPEECH PACING (Whisper)", fill="#888888", font=font_body)
    # Draw the pacing value as text, whether it is a number or a descriptive label.
    draw.text((650, 195), str(pacing_score), fill="#00E5FF", font=font_title)

    # Draw the large box that will contain the suggested rewritten hook.
    draw.rectangle([(60, 290), (1140, 520)], fill="#1E222A", outline="#00FFA3", width=2)
    # Label the rewrite box and identify the source of the suggestion.
    draw.text((90, 315), "💡 SUGGESTED HOOK REWRITE (Gemini AI)", fill="#00FFA3", font=font_header)

    # Split the rewrite into individual words so long text can be wrapped onto lines.
    words = rewrite_text.split()
    # Store the finished lines that will be drawn in the rewrite box.
    lines = []
    # Store words for the line currently being assembled.
    current_line = []
    
    # Process each word in the rewritten hook.
    for word in words:
        # Temporarily add the next word to the current line.
        current_line.append(word)
        # Check whether the line is now longer than the chosen character limit.
        if len(" ".join(current_line)) > 55:
            # Remove the word that made the line too long.
            current_line.pop()
            # Save the shorter line as a completed line of text.
            lines.append(" ".join(current_line))
            # Start the next line with the word that did not fit.
            current_line = [word]
    # Save the final line when the rewrite contained any remaining words.
    if current_line:
        lines.append(" ".join(current_line))

    # Begin drawing rewrite text at a position below its heading.
    y_text = 370
    # Draw at most three lines so the text remains inside the card's rewrite box.
    for line in lines[:3]:  # Max 3 lines to fit in box
        # Draw quotation marks around each displayed line of the suggested hook.
        draw.text((90, y_text), f'"{line}"', fill="#FFFFFF", font=font_body)
        # Move down before drawing the next line.
        y_text += 38

    # Draw a small footer identifying the tool that created this card.
    draw.text((60, 560), "Analyzed with HookAnalyzer Prototype • Free Tool", fill="#666666", font=font_body)

    # Create an in-memory binary buffer to hold the PNG image bytes.
    buffer = io.BytesIO()
    # Encode the card as a PNG and write those bytes into the buffer.
    card.save(buffer, format="PNG")
    # Move the buffer cursor back to the beginning so the next reader sees the whole image.
    buffer.seek(0)
    # Return the in-memory PNG to the caller.
    return buffer