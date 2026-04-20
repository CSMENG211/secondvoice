import queue
import tempfile
import threading
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audio import SemanticEndpointResult, StreamSegmenter
from audio.levels import audio_blocksize
from audio.segmenter import (
    StreamSpeechDetector,
    is_repetitive_transcript,
    new_transcript_words,
    trim_repetitive_transcript_suffix,
)


def silent_chunk() -> bytes:
    """Return one block of silence matching the app audio chunk size."""
    return b"\0" * audio_blocksize() * 2


def constant_chunk(level: int) -> bytes:
    """Return one block whose int16 samples all have the requested level."""
    sample = level.to_bytes(2, byteorder=sys.byteorder, signed=True)
    return sample * audio_blocksize()


def marked_chunk(marker: int) -> bytes:
    """Return one distinguishable chunk for queue assertions."""
    sample = marker.to_bytes(2, byteorder=sys.byteorder, signed=True)
    return sample * audio_blocksize()


def attach_endpoint_queues(segmenter: StreamSegmenter) -> None:
    """Attach in-memory endpoint queues without starting a worker thread."""
    segmenter.semantic_job_queue = queue.Queue()
    segmenter.semantic_result_queue = queue.Queue()


def complete_fallback_check(
    segmenter: StreamSegmenter,
    transcript: str,
    is_rejected: bool = False,
) -> bool:
    """Publish one fallback result for the queued fallback job."""
    assert segmenter.semantic_job_queue is not None
    assert segmenter.semantic_result_queue is not None
    job = segmenter.semantic_job_queue.get_nowait()
    assert job.purpose == "fallback"
    segmenter.semantic_result_queue.put(
        SemanticEndpointResult(
            is_complete=False,
            transcript=transcript,
            is_rejected=is_rejected,
            segment_index=job.segment_index,
            pause_index=job.pause_index,
            purpose=job.purpose,
        )
    )
    return segmenter.handle_semantic_endpoint_results()


def test_stream_speech_detector_hysteresis_and_hangover() -> None:
    detector = StreamSpeechDetector(
        start_threshold=500,
        continue_threshold_ratio=0.6,
        hangover_seconds=0.3,
    )

    assert not detector.is_speech(constant_chunk(350))
    assert detector.is_speech(constant_chunk(500))
    assert detector.is_speech(constant_chunk(350))
    assert detector.is_speech(silent_chunk())
    assert detector.is_speech(silent_chunk())
    assert detector.is_speech(silent_chunk())
    assert not detector.is_speech(silent_chunk())


def test_new_transcript_words() -> None:
    assert new_transcript_words("hello world", "hello world") == []
    assert new_transcript_words("Hello, WORLD!", "hello world") == []
    assert new_transcript_words("hello world", "hello brave") == ["brave"]
    assert new_transcript_words("hello world", "hello world from google") == [
        "from",
        "google",
    ]
    assert new_transcript_words(
        "return indices of the two numbers",
        "return indices of the two numbers that add up",
    ) == ["that", "add", "up"]


def test_repetitive_transcript_detection() -> None:
    repeated_transcript = "Some new ones, " + "maybe new ones, " * 24
    decayed_transcript = (
        "Given an array of integers, I would use a hash map. "
        + "maybe new ones, " * 8
    )
    variant_decayed_transcript = (
        "Given an array of integers, I would use a hash map. "
        "maybe new ones maybe new one maybe new ones maybe the new ones"
    )
    bigram_decayed_transcript = (
        "Given an array of integers, I would use a hash map. "
        + "left pointer " * 8
    )
    normal_transcript = (
        "Given an array of integers and a target, return indices of two "
        "numbers that add up to the target."
    )

    assert is_repetitive_transcript(repeated_transcript)
    assert is_repetitive_transcript(decayed_transcript)
    assert is_repetitive_transcript(variant_decayed_transcript)
    assert is_repetitive_transcript(bigram_decayed_transcript)
    assert not is_repetitive_transcript(normal_transcript)
    assert not is_repetitive_transcript("yes yes yes")
    assert new_transcript_words("hello world", repeated_transcript) == []
    assert new_transcript_words("hello world", decayed_transcript) == []
    assert (
        trim_repetitive_transcript_suffix(decayed_transcript)
        == "Given an array of integers, I would use a hash map"
    )
    assert (
        trim_repetitive_transcript_suffix(variant_decayed_transcript)
        == "Given an array of integers, I would use a hash map"
    )
    assert (
        trim_repetitive_transcript_suffix(bigram_decayed_transcript)
        == "Given an array of integers, I would use a hash map"
    )
    assert trim_repetitive_transcript_suffix(normal_transcript) == normal_transcript
    assert trim_repetitive_transcript_suffix(repeated_transcript) == ""


def test_endpoint_draft_chunks_are_capped_to_recent_audio() -> None:
    segment_queue: queue.Queue[object] = queue.Queue()

    with tempfile.TemporaryDirectory() as temp_dir:
        segmenter = StreamSegmenter(
            output_dir=Path(temp_dir),
            segment_queue=segment_queue,
            stop_event=threading.Event(),
            hard_silence_seconds=0.2,
            semantic_endpoint_detector=lambda _audio_path: SemanticEndpointResult(
                is_complete=False,
                transcript="",
            ),
            endpoint_draft_seconds=0.2,
        )
        attach_endpoint_queues(segmenter)
        segmenter.start_segment()
        chunks = [marked_chunk(index) for index in range(1, 6)]
        segmenter.write_segment_chunks(chunks)

        segmenter.queue_semantic_endpoint_check()

        assert segmenter.semantic_job_queue is not None
        job = segmenter.semantic_job_queue.get_nowait()
        assert job.purpose == "semantic"
        assert job.chunks == chunks[-2:]

        segmenter.silent_blocks = segmenter.hard_silence_blocks_needed
        assert segmenter.queue_fallback_endpoint_check()
        fallback_job = segmenter.semantic_job_queue.get_nowait()
        assert fallback_job.purpose == "fallback"
        assert fallback_job.chunks == chunks[-2:]


def test_fallback_delays_when_endpoint_transcript_gains_words() -> None:
    segment_queue: queue.Queue[object] = queue.Queue()

    def detector(_audio_path: Path) -> SemanticEndpointResult:
        return SemanticEndpointResult(
            is_complete=False,
            transcript="hello world from google today",
        )

    with tempfile.TemporaryDirectory() as temp_dir:
        segmenter = StreamSegmenter(
            output_dir=Path(temp_dir),
            segment_queue=segment_queue,
            stop_event=threading.Event(),
            hard_silence_seconds=0.2,
            semantic_endpoint_detector=detector,
        )
        segmenter.start_segment()
        segmenter.write_segment_chunks([silent_chunk()])
        attach_endpoint_queues(segmenter)
        segmenter.latest_semantic_transcript = "hello world"
        segmenter.silent_blocks = segmenter.hard_silence_blocks_needed

        segmenter.handle_hard_silence_fallback()

        assert segment_queue.empty()
        assert segmenter.recording_started
        assert segmenter.fallback_check_queued_this_pause

        completed = complete_fallback_check(segmenter, "hello world from google today")

        assert not completed
        assert segment_queue.empty()
        assert segmenter.recording_started
        assert segmenter.silent_blocks == 0
        assert segmenter.semantic_pause_index == 1
        assert segmenter.fallback_delay_count == 1


def test_fallback_accepts_after_max_delay_count() -> None:
    segment_queue: queue.Queue[object] = queue.Queue()

    with tempfile.TemporaryDirectory() as temp_dir:
        segmenter = StreamSegmenter(
            output_dir=Path(temp_dir),
            segment_queue=segment_queue,
            stop_event=threading.Event(),
            hard_silence_seconds=0.2,
            semantic_endpoint_detector=lambda _audio_path: SemanticEndpointResult(
                is_complete=False,
                transcript="hello world from google today",
            ),
            max_fallback_delays=1,
        )
        segmenter.start_segment()
        segmenter.write_segment_chunks([silent_chunk()])
        attach_endpoint_queues(segmenter)
        segmenter.latest_semantic_transcript = "hello world"
        segmenter.silent_blocks = segmenter.hard_silence_blocks_needed
        segmenter.fallback_delay_count = 1

        segmenter.handle_hard_silence_fallback()

        completed_result = complete_fallback_check(segmenter, "hello world from google today")

        assert completed_result
        completed = segment_queue.get_nowait()
        assert completed.completion_reason == "0.2s silence fallback"
        assert not segmenter.recording_started


def test_fallback_accepts_after_max_segment_duration() -> None:
    segment_queue: queue.Queue[object] = queue.Queue()

    with tempfile.TemporaryDirectory() as temp_dir:
        segmenter = StreamSegmenter(
            output_dir=Path(temp_dir),
            segment_queue=segment_queue,
            stop_event=threading.Event(),
            hard_silence_seconds=0.2,
            semantic_endpoint_detector=lambda _audio_path: SemanticEndpointResult(
                is_complete=False,
                transcript="hello world from google today",
            ),
            max_segment_seconds=0.1,
        )
        segmenter.start_segment()
        segmenter.write_segment_chunks([silent_chunk()])
        attach_endpoint_queues(segmenter)
        segmenter.latest_semantic_transcript = "hello world"
        segmenter.silent_blocks = segmenter.hard_silence_blocks_needed

        segmenter.handle_hard_silence_fallback()

        completed_result = complete_fallback_check(segmenter, "hello world from google today")

        assert completed_result
        completed = segment_queue.get_nowait()
        assert completed.completion_reason == "0.2s silence fallback"
        assert not segmenter.recording_started


def test_fallback_accepts_when_endpoint_transcript_has_no_new_words() -> None:
    segment_queue: queue.Queue[object] = queue.Queue()

    def detector(_audio_path: Path) -> SemanticEndpointResult:
        return SemanticEndpointResult(
            is_complete=False,
            transcript="hello world",
        )

    with tempfile.TemporaryDirectory() as temp_dir:
        segmenter = StreamSegmenter(
            output_dir=Path(temp_dir),
            segment_queue=segment_queue,
            stop_event=threading.Event(),
            hard_silence_seconds=0.2,
            semantic_endpoint_detector=detector,
        )
        segmenter.start_segment()
        segmenter.write_segment_chunks([silent_chunk()])
        attach_endpoint_queues(segmenter)
        segmenter.latest_semantic_transcript = "hello world"
        segmenter.silent_blocks = segmenter.hard_silence_blocks_needed

        segmenter.handle_hard_silence_fallback()

        assert segment_queue.empty()
        assert segmenter.recording_started

        completed_result = complete_fallback_check(segmenter, "hello world")

        assert completed_result
        completed = segment_queue.get_nowait()
        assert completed.completion_reason == "0.2s silence fallback"
        assert not segmenter.recording_started


def test_fallback_accepts_one_new_hallucinated_word() -> None:
    segment_queue: queue.Queue[object] = queue.Queue()

    def detector(_audio_path: Path) -> SemanticEndpointResult:
        return SemanticEndpointResult(
            is_complete=False,
            transcript="hello world maybe",
        )

    with tempfile.TemporaryDirectory() as temp_dir:
        segmenter = StreamSegmenter(
            output_dir=Path(temp_dir),
            segment_queue=segment_queue,
            stop_event=threading.Event(),
            hard_silence_seconds=0.2,
            semantic_endpoint_detector=detector,
        )
        segmenter.start_segment()
        segmenter.write_segment_chunks([silent_chunk()])
        attach_endpoint_queues(segmenter)
        segmenter.latest_semantic_transcript = "hello world"
        segmenter.silent_blocks = segmenter.hard_silence_blocks_needed

        segmenter.handle_hard_silence_fallback()

        assert segment_queue.empty()
        assert segmenter.recording_started

        completed_result = complete_fallback_check(segmenter, "hello world maybe")

        assert completed_result
        completed = segment_queue.get_nowait()
        assert completed.completion_reason == "0.2s silence fallback"
        assert not segmenter.recording_started


def test_fallback_accepts_repetitive_endpoint_transcript() -> None:
    segment_queue: queue.Queue[object] = queue.Queue()
    repeated_transcript = "Some new ones, " + "maybe new ones, " * 24

    def detector(_audio_path: Path) -> SemanticEndpointResult:
        return SemanticEndpointResult(
            is_complete=False,
            transcript=repeated_transcript,
        )

    with tempfile.TemporaryDirectory() as temp_dir:
        segmenter = StreamSegmenter(
            output_dir=Path(temp_dir),
            segment_queue=segment_queue,
            stop_event=threading.Event(),
            hard_silence_seconds=0.2,
            semantic_endpoint_detector=detector,
        )
        segmenter.start_segment()
        segmenter.write_segment_chunks([silent_chunk()])
        attach_endpoint_queues(segmenter)
        segmenter.latest_semantic_transcript = "hello world"
        segmenter.silent_blocks = segmenter.hard_silence_blocks_needed

        segmenter.handle_hard_silence_fallback()

        assert segment_queue.empty()
        assert segmenter.recording_started

        completed_result = complete_fallback_check(segmenter, repeated_transcript)

        assert completed_result
        completed = segment_queue.get_nowait()
        assert completed.completion_reason == "0.2s silence fallback"
        assert not segmenter.recording_started


def test_fallback_accepts_decayed_endpoint_transcript() -> None:
    segment_queue: queue.Queue[object] = queue.Queue()
    meaningful_prefix = "Given an array of integers I would use a hash map"
    decayed_transcript = meaningful_prefix + " " + "maybe new ones " * 8

    def detector(_audio_path: Path) -> SemanticEndpointResult:
        return SemanticEndpointResult(
            is_complete=False,
            transcript=decayed_transcript,
        )

    with tempfile.TemporaryDirectory() as temp_dir:
        segmenter = StreamSegmenter(
            output_dir=Path(temp_dir),
            segment_queue=segment_queue,
            stop_event=threading.Event(),
            hard_silence_seconds=0.2,
            semantic_endpoint_detector=detector,
        )
        segmenter.start_segment()
        segmenter.write_segment_chunks([silent_chunk()])
        attach_endpoint_queues(segmenter)
        segmenter.latest_semantic_transcript = meaningful_prefix
        segmenter.silent_blocks = segmenter.hard_silence_blocks_needed

        segmenter.handle_hard_silence_fallback()

        assert segment_queue.empty()
        assert segmenter.recording_started

        completed_result = complete_fallback_check(segmenter, decayed_transcript)

        assert completed_result
        completed = segment_queue.get_nowait()
        assert completed.completion_reason == "0.2s silence fallback"
        assert not segmenter.recording_started


def test_fallback_accepts_rejected_endpoint_transcript() -> None:
    segment_queue: queue.Queue[object] = queue.Queue()

    def detector(_audio_path: Path) -> SemanticEndpointResult:
        return SemanticEndpointResult(
            is_complete=False,
            transcript="announce announce announce announce announce",
            is_rejected=True,
        )

    with tempfile.TemporaryDirectory() as temp_dir:
        segmenter = StreamSegmenter(
            output_dir=Path(temp_dir),
            segment_queue=segment_queue,
            stop_event=threading.Event(),
            hard_silence_seconds=0.2,
            semantic_endpoint_detector=detector,
        )
        segmenter.start_segment()
        segmenter.write_segment_chunks([silent_chunk()])
        attach_endpoint_queues(segmenter)
        segmenter.latest_semantic_transcript = "hello world"
        segmenter.silent_blocks = segmenter.hard_silence_blocks_needed

        segmenter.handle_hard_silence_fallback()

        assert segment_queue.empty()
        assert segmenter.recording_started

        completed_result = complete_fallback_check(
            segmenter,
            "announce announce announce announce announce",
            is_rejected=True,
        )

        assert completed_result
        completed = segment_queue.get_nowait()
        assert completed.completion_reason == "0.2s silence fallback"
        assert not segmenter.recording_started


def main() -> None:
    test_stream_speech_detector_hysteresis_and_hangover()
    test_new_transcript_words()
    test_repetitive_transcript_detection()
    test_endpoint_draft_chunks_are_capped_to_recent_audio()
    test_fallback_delays_when_endpoint_transcript_gains_words()
    test_fallback_accepts_after_max_delay_count()
    test_fallback_accepts_after_max_segment_duration()
    test_fallback_accepts_when_endpoint_transcript_has_no_new_words()
    test_fallback_accepts_one_new_hallucinated_word()
    test_fallback_accepts_repetitive_endpoint_transcript()
    test_fallback_accepts_decayed_endpoint_transcript()
    test_fallback_accepts_rejected_endpoint_transcript()
    print("fallback guard tests passed")


if __name__ == "__main__":
    main()
