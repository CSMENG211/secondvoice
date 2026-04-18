import re
from typing import Literal

from constants import DEFAULT_QUESTION_START_PATTERN

QuestionTriggerMode = Literal["phrase", "smart", "always"]


def extract_question_prompt(
    transcript: str,
    start_pattern: str = DEFAULT_QUESTION_START_PATTERN,
    mode: QuestionTriggerMode = "phrase",
) -> str | None:
    """Extract a prompt when the transcript matches the selected trigger mode."""
    if mode == "always":
        return transcript.strip()

    prompt = extract_after_start_pattern(transcript, start_pattern)
    if prompt is not None:
        return prompt

    if mode == "smart" and is_question(transcript):
        return transcript.strip(" .,:;-")

    return None


def extract_after_start_pattern(
    transcript: str,
    start_pattern: str,
) -> str | None:
    """Return transcript text that appears after a configured start regex."""
    normalized_transcript = normalize(transcript)
    match = re.search(start_pattern.lower(), normalized_transcript)
    if not match:
        return None

    return transcript_after_normalized_index(transcript, normalized_transcript, match.end())


def is_question(transcript: str) -> bool:
    """Return whether a transcript appears to be a direct question."""
    return transcript.strip().endswith("?")


def normalize(text: str) -> str:
    """Lowercase text, remove punctuation, and normalize common trigger words."""
    words = re.sub(r"[^a-z0-9]+", " ", text.lower()).split()
    normalized_words = ["ok" if word == "okay" else word for word in words]
    return " ".join(normalized_words)


def transcript_after_normalized_index(
    original: str,
    normalized_original: str,
    normalized_index: int,
) -> str:
    """Map a normalized text index back to the corresponding original-word suffix."""
    if normalized_index >= len(normalized_original):
        return ""

    normalized_suffix = normalized_original[normalized_index:].strip()
    if not normalized_suffix:
        return ""

    suffix_words = normalized_suffix.split()
    original_words = re.findall(r"\S+", original)

    if len(suffix_words) > len(original_words):
        return ""

    return " ".join(original_words[-len(suffix_words):]).strip(" .,:;-")
