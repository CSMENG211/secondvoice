import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

from constants import (
    AUDIO_SAMPLE_RATE,
    SPEAKER_MATCH_THRESHOLD,
    SPEAKER_MODEL_DIR,
    SPEAKER_MODEL_SOURCE,
    SPEAKER_PROFILE_EMBEDDING_PATH,
    SPEAKER_PROFILE_METADATA_PATH,
)


@dataclass(frozen=True)
class SpeakerHint:
    """Confidence hint for whether an audio segment sounds like the interviewee."""

    role_hint: str
    confidence: float | None
    similarity: float | None
    profile_available: bool

    def prompt_value(self) -> str:
        """Return the confidence value for downstream LLM prompts."""
        if self.confidence is None:
            return "unavailable"
        return f"{self.confidence:.2f}"

    def log_value(self) -> str:
        """Return a concise human-readable speaker hint."""
        if self.confidence is None:
            return "unavailable"
        similarity = "unknown" if self.similarity is None else f"{self.similarity:.3f}"
        return f"{self.role_hint} confidence={self.confidence:.2f} similarity={similarity}"


class SpeakerIdentifier:
    """Persistent interviewee voice enrollment and per-segment matching."""

    def __init__(self, log_missing_profile: bool = True) -> None:
        self.classifier = None
        try:
            self.profile = self._load_profile()
        except RuntimeError as error:
            logger.warning("Could not load interviewee voice profile: {}", error)
            self.profile = None

        if self.profile is not None:
            logger.info("Loaded interviewee voice profile.")
        elif log_missing_profile:
            logger.info(
                "No interviewee voice profile found. Run with --enroll to enable voice hints."
            )

    @property
    def has_profile(self) -> bool:
        """Return whether a persisted interviewee profile is available."""
        return self.profile is not None

    def enroll_from_clips(self, audio_paths: list[Path]) -> None:
        """Persist an averaged interviewee embedding from enrollment clips."""
        if not audio_paths:
            raise ValueError("At least one enrollment clip is required.")

        torch = self._torch()
        embeddings = [self._encode(audio_path) for audio_path in audio_paths]
        profile = torch.stack(embeddings).mean(dim=0)
        profile = torch.nn.functional.normalize(profile, dim=-1)

        SPEAKER_PROFILE_EMBEDDING_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(profile.cpu(), SPEAKER_PROFILE_EMBEDDING_PATH)
        self._save_metadata(len(audio_paths))
        self.profile = profile.cpu()

    def match(self, audio_path: Path) -> SpeakerHint:
        """Return an interviewee voice-match hint for one audio segment."""
        if self.profile is None:
            return SpeakerHint(
                role_hint="unknown",
                confidence=None,
                similarity=None,
                profile_available=False,
            )

        torch = self._torch()
        embedding = self._encode(audio_path)
        profile = self.profile.to(embedding.device)
        similarity = torch.nn.functional.cosine_similarity(embedding, profile, dim=-1).item()
        confidence = max(0.0, min(1.0, (similarity + 1.0) / 2.0))
        role_hint = "interviewee" if confidence >= SPEAKER_MATCH_THRESHOLD else "interviewer"
        return SpeakerHint(
            role_hint=role_hint,
            confidence=confidence,
            similarity=similarity,
            profile_available=True,
        )

    def _encode(self, audio_path: Path):
        """Return a normalized speaker embedding for a WAV file."""
        torch = self._torch()
        classifier = self._classifier()
        signal = classifier.load_audio(str(audio_path))
        embedding = classifier.encode_batch(signal).squeeze()
        return torch.nn.functional.normalize(embedding, dim=-1)

    def _classifier(self):
        if self.classifier is None:
            try:
                from speechbrain.inference.speaker import EncoderClassifier
            except ImportError:
                try:
                    from speechbrain.pretrained import EncoderClassifier
                except ImportError as error:
                    raise RuntimeError(
                        "Voice matching requires speechbrain. Install project requirements, "
                        "then run enrollment again."
                    ) from error

            self.classifier = EncoderClassifier.from_hparams(
                source=SPEAKER_MODEL_SOURCE,
                savedir=str(SPEAKER_MODEL_DIR),
            )
        return self.classifier

    def _load_profile(self):
        if not SPEAKER_PROFILE_EMBEDDING_PATH.exists():
            return None

        torch = self._torch()
        return torch.load(SPEAKER_PROFILE_EMBEDDING_PATH, map_location="cpu")

    def _save_metadata(self, clip_count: int) -> None:
        metadata = {
            "created_at": datetime.now(UTC).isoformat(),
            "model": SPEAKER_MODEL_SOURCE,
            "sample_rate": AUDIO_SAMPLE_RATE,
            "clip_count": clip_count,
            "embedding_strategy": "mean",
            "match_threshold": SPEAKER_MATCH_THRESHOLD,
        }
        SPEAKER_PROFILE_METADATA_PATH.write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )

    def _torch(self):
        try:
            import torch
        except ImportError as error:
            raise RuntimeError(
                "Voice enrollment requires torch, torchaudio, and speechbrain. "
                "Install project requirements, then run enrollment again."
            ) from error

        return torch
