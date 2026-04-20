import queue
import shutil
import sys
import tempfile
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import app
import audio.segmenter as segmenter
from audio import AudioEnhancementConfig, CompletedStreamSegment, SemanticEndpointJob
from audio.levels import audio_blocksize
from audio.wav import write_wav_file
from speech import SpeakerHint


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


class RecordingSpeakerIdentifier:
    def __init__(self) -> None:
        self.path: Path | None = None

    def match(self, audio_path: Path) -> SpeakerHint:
        self.path = audio_path
        assert audio_path.exists()
        return SpeakerHint(
            role_hint="unknown",
            confidence=None,
            similarity=None,
            profile_available=False,
        )


def copy_enhancement(input_path: Path, output_path: Path, _config: AudioEnhancementConfig) -> Path:
    shutil.copyfile(input_path, output_path)
    return output_path


def test_final_transcription_uses_enhanced_audio_and_speaker_uses_raw() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        raw_path = Path(temp_dir) / "stream-segment-0001.wav"
        write_wav_file(raw_path, [silent_chunk()])
        transcriber = RecordingTranscriber()
        speaker_identifier = RecordingSpeakerIdentifier()
        options = app.RuntimeOptions(ask_chatgpt=False)

        original_enhance_wav = app.enhance_wav
        try:
            app.enhance_wav = copy_enhancement
            submitted = app.process_stream_segment(
                CompletedStreamSegment(raw_path, "test"),
                transcriber,
                speaker_identifier,
                options,
                include_mode_prompt=True,
                photo_tracker=app.PhotoUploadTracker(),
            )
        finally:
            app.enhance_wav = original_enhance_wav

        assert not submitted
        assert speaker_identifier.path == raw_path
        assert transcriber.path == raw_path.with_name("stream-segment-0001-enhanced.wav")
        assert not raw_path.exists()
        assert not transcriber.path.exists()


def test_semantic_worker_uses_raw_audio_and_cleans_temp_files() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)
        job_queue: queue.Queue[SemanticEndpointJob | None] = queue.Queue()
        result_queue: queue.Queue[segmenter.SemanticEndpointResult] = queue.Queue()
        seen_path: list[Path] = []

        def detector(audio_path: Path) -> segmenter.SemanticEndpointResult:
            seen_path.append(audio_path)
            assert audio_path.exists()
            return segmenter.SemanticEndpointResult(is_complete=False, transcript="hello")

        worker = threading.Thread(
            target=segmenter.run_semantic_endpoint_worker,
            args=(
                output_dir,
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
                chunks=[silent_chunk()],
                purpose="semantic",
            )
        )
        job_queue.put(None)
        worker.join(timeout=3)

        assert not worker.is_alive()
        assert seen_path == [output_dir / "stream-segment-0001-semantic-check-0002.wav"]
        assert result_queue.get_nowait().transcript == "hello"
        assert not seen_path[0].exists()


def test_semantic_worker_cleans_temp_files_after_detector_exception() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)
        job_queue: queue.Queue[SemanticEndpointJob | None] = queue.Queue()
        result_queue: queue.Queue[segmenter.SemanticEndpointResult] = queue.Queue()

        def detector(_audio_path: Path) -> segmenter.SemanticEndpointResult:
            raise RuntimeError("boom")

        worker = threading.Thread(
            target=segmenter.run_semantic_endpoint_worker,
            args=(
                output_dir,
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
                chunks=[silent_chunk()],
                purpose="fallback",
            )
        )
        job_queue.put(None)
        worker.join(timeout=3)

        result = result_queue.get_nowait()
        assert not result.is_complete
        assert not (output_dir / "stream-segment-0001-fallback-check-0002.wav").exists()


def main() -> None:
    test_final_transcription_uses_enhanced_audio_and_speaker_uses_raw()
    test_semantic_worker_uses_raw_audio_and_cleans_temp_files()
    test_semantic_worker_cleans_temp_files_after_detector_exception()
    print("audio enhancement pipeline tests passed")


if __name__ == "__main__":
    main()
