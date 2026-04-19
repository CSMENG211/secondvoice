from pathlib import Path
import threading

from loguru import logger

from constants import DEFAULT_TRANSCRIPTION_MODEL

DEFAULT_MODEL = DEFAULT_TRANSCRIPTION_MODEL


class LocalTranscriber:
    """Reusable wrapper around a local faster-whisper model."""

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        """Load the configured Whisper model into memory."""
        from faster_whisper import WhisperModel

        logger.info("Loading local Whisper model. The first run may download model files...")
        self.model = WhisperModel(model, device="auto", compute_type="auto")
        self._lock = threading.Lock()
        logger.info("Model loaded.")

    def transcribe(self, audio_path: Path, *, log_progress: bool = True) -> str:
        """Transcribe a WAV file and return plain text."""
        with self._lock:
            if log_progress:
                logger.info("Decoding audio...")
            segments, _ = self.model.transcribe(str(audio_path), beam_size=5)

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
