from collections.abc import Iterable
from pathlib import Path
import wave

from constants import (
    AUDIO_CHANNELS,
    AUDIO_SAMPLE_RATE,
    AUDIO_SAMPLE_WIDTH_BYTES,
)


def open_wav_writer(output_path: Path) -> wave.Wave_write:
    """Open a mono int16 WAV file using the app's audio settings."""
    wav_file = wave.open(str(output_path), "wb")
    wav_file.setnchannels(AUDIO_CHANNELS)
    wav_file.setsampwidth(AUDIO_SAMPLE_WIDTH_BYTES)
    wav_file.setframerate(AUDIO_SAMPLE_RATE)
    return wav_file


def write_wav_file(output_path: Path, chunks: Iterable[bytes]) -> None:
    """Write a complete mono int16 WAV file using the app's audio settings."""
    with open_wav_writer(output_path) as wav_file:
        write_chunks(wav_file, chunks)


def write_chunks(wav_file: wave.Wave_write, chunks: Iterable[bytes]) -> None:
    """Write a sequence of raw int16 audio chunks into a WAV file."""
    for chunk in chunks:
        wav_file.writeframes(chunk)
