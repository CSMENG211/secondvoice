import queue
import sys
import tempfile
import threading
from pathlib import Path
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import app
import speech.transcription as transcription
from audio import (
    CompletedStreamSegment,
    SemanticEndpointJob,
    SemanticEndpointResult,
    TranscriptionJob,
    TranscriptionResult,
    run_semantic_endpoint_worker,
    run_transcription_worker,
)
from audio.levels import audio_blocksize
from audio.wav import write_wav_file


def silent_chunk() -> bytes:
    """Return one block of silence matching the app audio chunk size."""
    return b"\0" * audio_blocksize() * 2


class RecordingTranscriber:
    def __init__(self) -> None:
        self.path: Path | None = None

    def transcribe(self, audio_path: Path, *, log_progress: bool = True) -> str:
        self.path = audio_path
        assert audio_path.exists()
        return "hello world"


class FinalizingTranscriber:
    def __init__(self, transcript: str, *, raise_error: bool = False) -> None:
        self.transcript = transcript
        self.raise_error = raise_error
        self.path: Path | None = None
        self.frame_count: int | None = None

    def transcribe(self, audio_path: Path, *, log_progress: bool = True) -> str:
        self.path = audio_path
        assert audio_path.exists()
        with wave.open(str(audio_path), "rb") as wav_file:
            self.frame_count = wav_file.getnframes()
        if self.raise_error:
            raise RuntimeError("boom")
        return self.transcript


def test_process_stream_segment_uses_streamed_transcript_and_raw_audio() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        raw_path = Path(temp_dir) / "stream-segment-0001.wav"
        write_wav_file(raw_path, [silent_chunk()])
        options = app.RuntimeOptions(ask_chatgpt=False)

        submitted = app.process_stream_segment(
            CompletedStreamSegment(raw_path, "test", transcript="hello world"),
            options,
            include_mode_prompt=True,
            photo_tracker=app.PhotoUploadTracker(),
            round_context="",
            behavior_state=app.BehaviorState(),
        )

        assert not submitted
        assert not raw_path.exists()


def test_process_stream_segment_prefers_finalized_transcript_when_available() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        raw_path = Path(temp_dir) / "stream-segment-0001.wav"
        write_wav_file(raw_path, [silent_chunk() for _ in range(5)])
        options = app.RuntimeOptions(ask_chatgpt=False)
        final_transcriber = FinalizingTranscriber("hash map because lookup is constant time")

        submitted = app.process_stream_segment(
            CompletedStreamSegment(
                raw_path,
                "test",
                transcript="draft transcript",
                locked_transcript="I would use a hash map",
                locked_chunk_index=4,
            ),
            options,
            include_mode_prompt=True,
            photo_tracker=app.PhotoUploadTracker(),
            round_context="",
            behavior_state=app.BehaviorState(),
            final_transcriber=final_transcriber,
        )

        assert not submitted
        assert final_transcriber.path is not None
        assert final_transcriber.path != raw_path
        assert final_transcriber.frame_count == audio_blocksize() * 3
        assert not raw_path.exists()
        assert not final_transcriber.path.exists()


def test_finalize_segment_transcript_falls_back_to_streamed_text_on_error() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        raw_path = Path(temp_dir) / "stream-segment-0001.wav"
        write_wav_file(raw_path, [silent_chunk()])
        final_transcriber = FinalizingTranscriber("", raise_error=True)

        transcript = app.finalize_segment_transcript(
            raw_path,
            "draft transcript",
            locked_transcript="",
            locked_chunk_index=0,
            final_transcriber=final_transcriber,
        )

        assert transcript == "draft transcript"


def test_combine_locked_and_tail_transcript_deduplicates_overlap() -> None:
    transcript = app.combine_locked_and_tail_transcript(
        "I would use a hash map",
        "hash map because lookup is constant time",
    )

    assert transcript == "I would use a hash map because lookup is constant time"


def test_mlx_transcribers_share_process_lock() -> None:
    first = transcription.MlxWhisperTranscriber("dummy-model")
    second = transcription.MlxWhisperTranscriber("dummy-model")

    assert not hasattr(first, "_lock")
    assert not hasattr(second, "_lock")
    assert isinstance(transcription._MLX_TRANSCRIBE_LOCK, type(threading.Lock()))


def test_transcription_worker_uses_snapshot_audio_and_cleans_temp_files() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)
        job_queue: queue.Queue[TranscriptionJob | None] = queue.Queue()
        result_queue: queue.Queue[TranscriptionResult] = queue.Queue()
        transcriber = RecordingTranscriber()

        worker = threading.Thread(
            target=run_transcription_worker,
            args=(
                output_dir,
                job_queue,
                result_queue,
                transcriber,
            ),
        )
        worker.start()
        job_queue.put(
            TranscriptionJob(
                segment_index=1,
                pause_index=2,
                start_chunk_index=0,
                end_chunk_index=1,
                chunks=[silent_chunk()],
            )
        )
        job_queue.put(None)
        worker.join(timeout=3)

        assert not worker.is_alive()
        assert transcriber.path == output_dir / "stream-segment-0001-transcription-check-0002.wav"
        assert result_queue.get_nowait().transcript == "hello world"
        assert not transcriber.path.exists()


def test_semantic_worker_classifies_stabilized_text() -> None:
    job_queue: queue.Queue[SemanticEndpointJob | None] = queue.Queue()
    result_queue: queue.Queue[SemanticEndpointResult] = queue.Queue()

    def detector(transcript: str) -> SemanticEndpointResult:
        assert transcript == "stable answer"
        return SemanticEndpointResult(is_complete=True, transcript=transcript)

    worker = threading.Thread(
        target=run_semantic_endpoint_worker,
        args=(
            job_queue,
            result_queue,
            detector,
        ),
    )
    worker.start()
    job_queue.put(
        SemanticEndpointJob(
            segment_index=1,
            pause_index=2,
            transcript="stable answer",
            transcript_key="stable answer",
        )
    )
    job_queue.put(None)
    worker.join(timeout=3)

    result = result_queue.get_nowait()
    assert result.is_complete
    assert result.transcript == "stable answer"


def main() -> None:
    test_process_stream_segment_uses_streamed_transcript_and_raw_audio()
    test_process_stream_segment_prefers_finalized_transcript_when_available()
    test_finalize_segment_transcript_falls_back_to_streamed_text_on_error()
    test_combine_locked_and_tail_transcript_deduplicates_overlap()
    test_mlx_transcribers_share_process_lock()
    test_transcription_worker_uses_snapshot_audio_and_cleans_temp_files()
    test_semantic_worker_classifies_stabilized_text()
    print("streaming pipeline tests passed")


if __name__ == "__main__":
    main()
