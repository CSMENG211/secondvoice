from pathlib import Path
import threading
from typing import Protocol
import wave

from loguru import logger

from constants import (
    DEFAULT_ENDPOINT_TRANSCRIPTION_MODEL,
    DEFAULT_FINAL_TRANSCRIPTION_BACKEND,
    DEFAULT_FINAL_TRANSCRIPTION_MODEL,
    TRANSCRIPTION_BACKEND_FASTER_WHISPER,
    TRANSCRIPTION_BACKEND_MLX_WHISPER,
    TRANSCRIPTION_INITIAL_PROMPT,
)

DEFAULT_BACKEND = DEFAULT_FINAL_TRANSCRIPTION_BACKEND
DEFAULT_MODEL = DEFAULT_FINAL_TRANSCRIPTION_MODEL


class Transcriber(Protocol):
    """Common interface for local speech-to-text backends."""

    def transcribe(self, audio_path: Path, *, log_progress: bool = True) -> str:
        """Transcribe a WAV file and return plain text."""
        ...


def create_transcriber(
    backend: str = DEFAULT_BACKEND,
    model: str = DEFAULT_MODEL,
) -> Transcriber:
    """Create a local transcriber for the configured backend."""
    if backend == TRANSCRIPTION_BACKEND_FASTER_WHISPER:
        return FasterWhisperTranscriber(model)
    if backend == TRANSCRIPTION_BACKEND_MLX_WHISPER:
        return MlxWhisperTranscriber(model)

    raise ValueError(f"Unsupported transcription backend: {backend!r}")


class FasterWhisperTranscriber:
    """Reusable wrapper around a local faster-whisper model."""

    def __init__(self, model: str = DEFAULT_ENDPOINT_TRANSCRIPTION_MODEL) -> None:
        """Load the configured faster-whisper model into memory."""
        from faster_whisper import WhisperModel

        logger.info(
            "Loading faster-whisper model {}. The first run may download model files...",
            model,
        )
        self.model = WhisperModel(model, device="auto", compute_type="auto")
        self._lock = threading.Lock()
        logger.info("faster-whisper model loaded: {}", model)

    def transcribe(self, audio_path: Path, *, log_progress: bool = True) -> str:
        """Transcribe a WAV file and return plain text."""
        with self._lock:
            if log_progress:
                logger.info("Decoding audio with faster-whisper...")
            segments, _ = self.model.transcribe(
                str(audio_path),
                beam_size=5,
                initial_prompt=TRANSCRIPTION_INITIAL_PROMPT,
            )

            transcript_parts = []
            for segment in segments:
                text = segment.text.strip()
                if text:
                    transcript_parts.append(text)
                    if log_progress:
                        logger.debug("Transcribed segment: {}", text)

            if log_progress:
                logger.debug("Done decoding audio.")
            return " ".join(transcript_parts).strip()


class MlxWhisperTranscriber:
    """Reusable wrapper around an MLX Whisper model on Apple Silicon."""

    def __init__(self, model: str = DEFAULT_FINAL_TRANSCRIPTION_MODEL) -> None:
        """Store the configured MLX model name for lazy transcription."""
        self.model = model
        self._lock = threading.Lock()
        logger.info(
            "Configured mlx-whisper model {}. The first transcription may download model files.",
            model,
        )

    def transcribe(self, audio_path: Path, *, log_progress: bool = True) -> str:
        """Transcribe a WAV file with mlx-whisper and return plain text."""
        try:
            import mlx_whisper
        except ImportError as exc:
            raise RuntimeError(
                "mlx-whisper is not installed. Install dependencies with "
                "`pip install -r requirements.txt`."
            ) from exc

        with self._lock:
            if log_progress:
                logger.info("Decoding audio with mlx-whisper...")

            result = self._transcribe_with_prompt(mlx_whisper, audio_path)
            text = str(result.get("text", "")).strip()

            if log_progress:
                logger.debug("Done decoding audio.")
            return text

    def _transcribe_with_prompt(self, mlx_whisper, audio_path: Path) -> dict:
        """Call mlx-whisper with the app prompt, retrying for older signatures."""
        audio = load_wav_as_float32(audio_path)
        try:
            return mlx_whisper.transcribe(
                audio,
                path_or_hf_repo=self.model,
                initial_prompt=TRANSCRIPTION_INITIAL_PROMPT,
            )
        except TypeError:
            logger.debug("mlx-whisper did not accept initial_prompt; retrying without it.")
            return mlx_whisper.transcribe(
                audio,
                path_or_hf_repo=self.model,
            )


def load_wav_as_float32(audio_path: Path):
    """Load a PCM WAV file as mono float32 samples for mlx-whisper."""
    import numpy as np

    with wave.open(str(audio_path), "rb") as wav_file:
        sample_width = wav_file.getsampwidth()
        channels = wav_file.getnchannels()
        frames = wav_file.readframes(wav_file.getnframes())

    if sample_width != 2:
        raise RuntimeError(
            f"Expected 16-bit PCM WAV for mlx-whisper, got {sample_width * 8}-bit audio."
        )

    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples


LocalTranscriber = FasterWhisperTranscriber
