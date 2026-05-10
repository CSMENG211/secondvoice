from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol


class StreamingTranscriber(Protocol):
    """Common interface for the recorder's single streaming ASR path."""

    def transcribe(self, audio_path: Path, *, log_progress: bool = True) -> str:
        """Transcribe a WAV file and return plain text."""
        ...


@dataclass(frozen=True)
class CompletedStreamSegment:
    """Recorded stream segment plus the trigger that ended it."""

    path: Path
    completion_reason: str
    transcript: str = ""
    locked_transcript: str = ""
    locked_chunk_index: int = 0


@dataclass(frozen=True)
class TranscriptionJob:
    """Snapshot of an in-progress segment to transcribe off the recorder thread."""

    segment_index: int
    pause_index: int
    start_chunk_index: int
    end_chunk_index: int
    chunks: list[bytes]


@dataclass(frozen=True)
class TranscriptionResult:
    """Transcript result for one queued draft segment snapshot."""

    transcript: str = ""
    is_rejected: bool = False
    segment_index: int = -1
    pause_index: int = -1
    start_chunk_index: int = 0
    end_chunk_index: int = 0


@dataclass(frozen=True)
class SemanticEndpointJob:
    """Stabilized transcript submitted for semantic completion classification."""

    segment_index: int
    pause_index: int
    transcript: str
    transcript_key: str = ""


@dataclass(frozen=True)
class SemanticEndpointResult:
    """Semantic completion result for one stabilized transcript."""

    is_complete: bool
    transcript: str = ""
    is_rejected: bool = False
    segment_index: int = -1
    pause_index: int = -1
    transcript_key: str = ""


SemanticEndpointDetector = Callable[[str], SemanticEndpointResult]
