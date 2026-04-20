from dataclasses import dataclass
from pathlib import Path
import time
import wave

from loguru import logger
import numpy as np

from audio.constants import (
    AUDIO_CHANNELS,
    AUDIO_SAMPLE_RATE,
    AUDIO_SAMPLE_WIDTH_BYTES,
)
from audio.wav import open_wav_writer

@dataclass(frozen=True)
class AudioEnhancementConfig:
    """Configuration for optional transcription-time audio enhancement."""

    enabled: bool = True
    prop_decrease: float = 0.85
    n_fft: int = 512
    win_length: int = 512
    hop_length: int = 128
    time_mask_smooth_ms: int = 50
    freq_mask_smooth_hz: int = 500


def enhance_wav(
    input_path: Path,
    output_path: Path,
    config: AudioEnhancementConfig,
) -> Path:
    """Write an enhanced WAV for transcription, returning raw input on fallback."""
    if not config.enabled:
        return input_path

    started = time.perf_counter()
    try:
        samples, sample_rate = load_wav_as_float32(input_path)
        enhanced = reduce_noise(samples, sample_rate, config)
        write_float32_as_wav(output_path, enhanced)
    except Exception as error:
        output_path.unlink(missing_ok=True)
        logger.warning("Audio enhancement failed for {}: {}", input_path, error)
        logger.warning("Falling back to raw audio for transcription.")
        return input_path

    duration_ms = (time.perf_counter() - started) * 1000
    logger.debug(
        "Enhanced audio for transcription in {:.0f} ms: {}",
        duration_ms,
        output_path,
    )
    return output_path


def load_wav_as_float32(audio_path: Path) -> tuple[np.ndarray, int]:
    """Load a mono 16-bit PCM WAV as float32 samples in [-1, 1]."""
    with wave.open(str(audio_path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())

    if channels != AUDIO_CHANNELS:
        raise ValueError(f"Expected {AUDIO_CHANNELS} audio channel, got {channels}.")
    if sample_width != AUDIO_SAMPLE_WIDTH_BYTES:
        raise ValueError(
            f"Expected {AUDIO_SAMPLE_WIDTH_BYTES * 8}-bit PCM WAV, "
            f"got {sample_width * 8}-bit audio."
        )
    if sample_rate != AUDIO_SAMPLE_RATE:
        raise ValueError(f"Expected {AUDIO_SAMPLE_RATE} Hz audio, got {sample_rate} Hz.")

    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
    return samples / np.float32(32768.0), sample_rate


def reduce_noise(
    samples: np.ndarray,
    sample_rate: int,
    config: AudioEnhancementConfig,
) -> np.ndarray:
    """Run non-stationary spectral gating through noisereduce."""
    import noisereduce as nr

    return nr.reduce_noise(
        y=samples,
        sr=sample_rate,
        stationary=False,
        prop_decrease=config.prop_decrease,
        n_fft=config.n_fft,
        win_length=config.win_length,
        hop_length=config.hop_length,
        time_mask_smooth_ms=config.time_mask_smooth_ms,
        freq_mask_smooth_hz=config.freq_mask_smooth_hz,
    ).astype(np.float32, copy=False)


def write_float32_as_wav(output_path: Path, samples: np.ndarray) -> None:
    """Write float samples as a mono 16-bit PCM WAV, clipping safely."""
    clipped = np.clip(samples, -1.0, 1.0)
    int_samples = np.round(clipped * 32767.0).astype(np.int16)
    with open_wav_writer(output_path) as wav_file:
        wav_file.writeframes(int_samples.tobytes())
