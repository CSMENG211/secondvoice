from array import array
from collections import deque
import math

from constants import (
    AUDIO_CHUNK_SECONDS,
    AUDIO_PRE_ROLL_SECONDS,
    AUDIO_SAMPLE_RATE,
)


def audio_blocksize() -> int:
    """Return the number of samples in one audio chunk."""
    return int(AUDIO_SAMPLE_RATE * AUDIO_CHUNK_SECONDS)


def block_count_for_seconds(seconds: float) -> int:
    """Return the number of audio chunks needed to cover a duration."""
    return max(1, math.ceil(seconds / AUDIO_CHUNK_SECONDS))


def create_pre_roll_buffer() -> deque[bytes]:
    """Return a buffer that keeps audio from just before speech starts."""
    return deque(maxlen=block_count_for_seconds(AUDIO_PRE_ROLL_SECONDS))


def chunk_is_speech(chunk: bytes, threshold: int) -> bool:
    """Return whether an audio chunk crosses the configured speech threshold."""
    return rms_level(chunk) >= threshold


def rms_level(chunk: bytes) -> float:
    """Calculate the root-mean-square amplitude for an int16 audio chunk."""
    if not chunk:
        return 0.0

    samples = array("h")
    samples.frombytes(chunk)
    if not samples:
        return 0.0

    square_sum = sum(sample * sample for sample in samples)
    return math.sqrt(square_sum / len(samples))
