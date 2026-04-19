from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import queue
import threading
import wave

from loguru import logger
import sounddevice as sd

from audio.levels import (
    audio_blocksize,
    block_count_for_seconds,
    chunk_is_speech,
    create_pre_roll_buffer,
)
from audio.wav import open_wav_writer, write_wav_file
from constants import (
    AUDIO_CHANNELS,
    AUDIO_SAMPLE_RATE,
    DEFAULT_SILENCE_THRESHOLD,
    STREAM_HARD_SILENCE_SECONDS,
    STREAM_SEMANTIC_SILENCE_SECONDS,
)

SemanticEndpointDetector = Callable[[Path], bool]


@dataclass(frozen=True)
class SemanticEndpointJob:
    """Snapshot of an in-progress segment to classify off the recorder thread."""

    segment_index: int
    pause_index: int
    chunks: list[bytes]


@dataclass(frozen=True)
class SemanticEndpointResult:
    """Semantic completion result for one queued draft segment snapshot."""

    segment_index: int
    pause_index: int
    is_complete: bool


class StreamSegmenter:
    """Continuously record voice-triggered utterance segments to WAV files."""

    def __init__(
        self,
        output_dir: Path,
        segment_queue: queue.Queue[Path | Exception],
        stop_event: threading.Event,
        hard_silence_seconds: float = STREAM_HARD_SILENCE_SECONDS,
        silence_threshold: int = DEFAULT_SILENCE_THRESHOLD,
        semantic_silence_seconds: float = STREAM_SEMANTIC_SILENCE_SECONDS,
        semantic_endpoint_detector: SemanticEndpointDetector | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.segment_queue = segment_queue
        self.stop_event = stop_event
        self.hard_silence_seconds = hard_silence_seconds
        self.silence_threshold = silence_threshold
        self.semantic_silence_seconds = semantic_silence_seconds
        self.semantic_endpoint_detector = semantic_endpoint_detector

        self.blocksize = audio_blocksize()
        self.hard_silence_blocks_needed = block_count_for_seconds(hard_silence_seconds)
        self.semantic_silence_blocks_needed = block_count_for_seconds(
            semantic_silence_seconds
        )
        self.pre_roll = create_pre_roll_buffer()
        self.recorded_chunks: list[bytes] = []
        self.silent_blocks = 0
        self.semantic_check_queued_this_pause = False
        self.semantic_pause_index = 0
        self.recording_started = False
        self.segment_index = 0
        self.wav_file: wave.Wave_write | None = None
        self.segment_path: Path | None = None

        self.semantic_job_queue: queue.Queue[SemanticEndpointJob | None] | None = None
        self.semantic_result_queue: queue.Queue[SemanticEndpointResult] | None = None
        self.semantic_worker: threading.Thread | None = None

    def run(self) -> None:
        """Run the recorder loop until stopped, queueing completed segments."""
        self.start_semantic_worker()
        logger.info("Audio stream active. Waiting for speech...")
        try:
            try:
                self.run_audio_stream()
            except Exception as error:
                self.segment_queue.put(error)
        finally:
            self.finish_active_segment()
            self.stop_semantic_worker()

    def start_semantic_worker(self) -> None:
        """Start the semantic endpoint worker when a detector is configured."""
        if self.semantic_endpoint_detector is None:
            return

        self.semantic_job_queue = queue.Queue()
        self.semantic_result_queue = queue.Queue()
        self.semantic_worker = threading.Thread(
            target=run_semantic_endpoint_worker,
            args=(
                self.output_dir,
                self.semantic_job_queue,
                self.semantic_result_queue,
                self.semantic_endpoint_detector,
            ),
        )
        self.semantic_worker.start()

    def stop_semantic_worker(self) -> None:
        """Stop the semantic endpoint worker if it was started."""
        if self.semantic_job_queue is not None:
            self.semantic_job_queue.put(None)
        if self.semantic_worker is not None:
            self.semantic_worker.join()

    def run_audio_stream(self) -> None:
        """Read microphone chunks and update the segment state machine."""
        with sd.RawInputStream(
            samplerate=AUDIO_SAMPLE_RATE,
            channels=AUDIO_CHANNELS,
            dtype="int16",
            blocksize=self.blocksize,
        ) as stream:
            while not self.stop_event.is_set():
                data, overflowed = stream.read(self.blocksize)
                chunk = bytes(data)
                if overflowed:
                    logger.warning("Audio warning: input overflowed")

                self.handle_audio_chunk(chunk)

    def handle_audio_chunk(self, chunk: bytes) -> None:
        """Process one raw audio chunk."""
        is_speech = chunk_is_speech(chunk, self.silence_threshold)

        if not self.recording_started:
            self.pre_roll.append(chunk)
            if is_speech:
                self.start_segment()
            return

        self.write_segment_chunks([chunk])
        self.update_silence_state(is_speech)

        if self.handle_semantic_endpoint_results():
            return

        if self.should_queue_semantic_check():
            self.semantic_check_queued_this_pause = True
            self.queue_semantic_endpoint_check()

        if self.silent_blocks >= self.hard_silence_blocks_needed:
            logger.info(
                "Audio segment fallback trigger: {:g}s of silence.",
                self.hard_silence_seconds,
            )
            self.finish_segment_and_reset()

    def start_segment(self) -> None:
        """Start writing a new stream segment."""
        logger.debug("Utterance started. Capturing audio...")
        self.segment_index += 1
        self.segment_path = self.output_dir / f"stream-segment-{self.segment_index:04d}.wav"
        self.wav_file = open_wav_writer(self.segment_path)
        self.recording_started = True
        self.write_segment_chunks(self.pre_roll)
        self.silent_blocks = 0
        self.semantic_check_queued_this_pause = False

    def finish_active_segment(self) -> None:
        """Queue the active segment during shutdown, if any."""
        if self.recording_started:
            self.finish_segment()

    def finish_segment_and_reset(self) -> None:
        """Finish the current segment and reset recorder state."""
        self.finish_segment()
        self.recording_started = False
        self.silent_blocks = 0
        self.semantic_check_queued_this_pause = False
        self.pre_roll.clear()

    def finish_segment(self) -> None:
        """Close and queue the current segment path."""
        if self.wav_file is None or self.segment_path is None:
            return

        self.wav_file.close()
        self.segment_queue.put(self.segment_path)
        self.wav_file = None
        self.segment_path = None
        self.recorded_chunks = []

    def write_segment_chunks(self, chunks: Iterable[bytes]) -> None:
        """Write raw chunks into the current segment and snapshot buffer."""
        if self.wav_file is None:
            return

        for item in chunks:
            self.wav_file.writeframes(item)
            self.recorded_chunks.append(item)

    def update_silence_state(self, is_speech: bool) -> None:
        """Update silence counters and pause identity."""
        if is_speech:
            if self.silent_blocks > 0 or self.semantic_check_queued_this_pause:
                self.semantic_pause_index += 1
            self.silent_blocks = 0
            self.semantic_check_queued_this_pause = False
        else:
            self.silent_blocks += 1

    def should_queue_semantic_check(self) -> bool:
        """Return whether the current silence pause needs a semantic check."""
        return (
            not self.semantic_check_queued_this_pause
            and self.silent_blocks >= self.semantic_silence_blocks_needed
        )

    def queue_semantic_endpoint_check(self) -> None:
        """Queue one semantic endpoint job for the current segment snapshot."""
        if (
            self.semantic_job_queue is None
            or self.segment_path is None
            or not self.recorded_chunks
        ):
            return

        self.semantic_job_queue.put(
            SemanticEndpointJob(
                segment_index=self.segment_index,
                pause_index=self.semantic_pause_index,
                chunks=list(self.recorded_chunks),
            )
        )
        logger.debug(
            "Semantic endpoint check queued after {:g}s pause.",
            self.semantic_silence_seconds,
        )

    def handle_semantic_endpoint_results(self) -> bool:
        """Apply current semantic endpoint results without blocking capture."""
        if self.semantic_result_queue is None:
            return False

        while True:
            try:
                result = self.semantic_result_queue.get_nowait()
            except queue.Empty:
                return False

            if not self.recording_started:
                continue
            if self.semantic_result_is_stale(result):
                logger.debug(
                    "Ignoring stale semantic endpoint result for segment {} pause {}.",
                    result.segment_index,
                    result.pause_index,
                )
                continue

            if not result.is_complete:
                logger.debug("Semantic endpoint check: incomplete; waiting for more speech.")
                continue

            logger.info(
                "Audio segment trigger: semantic completion after {:g}s pause.",
                self.semantic_silence_seconds,
            )
            self.finish_segment_and_reset()
            return True

    def semantic_result_is_stale(self, result: SemanticEndpointResult) -> bool:
        """Return whether a semantic result no longer matches current state."""
        return (
            result.segment_index != self.segment_index
            or result.pause_index != self.semantic_pause_index
        )


def stream_utterance_segments(
    output_dir: Path,
    segment_queue: queue.Queue[Path | Exception],
    stop_event: threading.Event,
    hard_silence_seconds: float = STREAM_HARD_SILENCE_SECONDS,
    silence_threshold: int = DEFAULT_SILENCE_THRESHOLD,
    semantic_silence_seconds: float = STREAM_SEMANTIC_SILENCE_SECONDS,
    semantic_endpoint_detector: SemanticEndpointDetector | None = None,
) -> None:
    """Continuously record voice-triggered utterance segments to WAV files."""
    StreamSegmenter(
        output_dir=output_dir,
        segment_queue=segment_queue,
        stop_event=stop_event,
        hard_silence_seconds=hard_silence_seconds,
        silence_threshold=silence_threshold,
        semantic_silence_seconds=semantic_silence_seconds,
        semantic_endpoint_detector=semantic_endpoint_detector,
    ).run()


def run_semantic_endpoint_worker(
    output_dir: Path,
    job_queue: queue.Queue[SemanticEndpointJob | None],
    result_queue: queue.Queue[SemanticEndpointResult],
    semantic_endpoint_detector: SemanticEndpointDetector,
) -> None:
    """Run semantic endpoint checks away from the real-time recorder loop."""
    while True:
        job = job_queue.get()
        if job is None:
            return

        draft_path = output_dir / (
            f"stream-segment-{job.segment_index:04d}"
            f"-semantic-check-{job.pause_index:04d}.wav"
        )
        try:
            write_wav_file(draft_path, job.chunks)
            is_complete = semantic_endpoint_detector(draft_path)
        except Exception as error:
            logger.warning("Semantic endpoint check failed: {}", error)
            is_complete = False
        finally:
            draft_path.unlink(missing_ok=True)

        result_queue.put(
            SemanticEndpointResult(
                segment_index=job.segment_index,
                pause_index=job.pause_index,
                is_complete=is_complete,
            )
        )
