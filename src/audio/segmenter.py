from collections.abc import Iterable
from pathlib import Path
import queue
import threading
import wave

from loguru import logger
import sounddevice as sd

from audio.levels import (
    audio_blocksize,
    block_count_for_seconds,
    create_pre_roll_buffer,
    rms_level,
)
from audio.stream_types import (
    CompletedStreamSegment,
    SemanticEndpointDetector,
    SemanticEndpointJob,
    SemanticEndpointResult,
    StreamingTranscriber,
    TranscriptionJob,
    TranscriptionResult,
)
from audio.stream_workers import run_semantic_endpoint_worker, run_transcription_worker
from audio.transcript_utils import (
    TRANSCRIPT_WORD_PATTERN,
)
from audio.wav import open_wav_writer
from audio.constants import (
    AUDIO_CHANNELS,
    AUDIO_CHUNK_SECONDS,
    AUDIO_SAMPLE_RATE,
    DEFAULT_SILENCE_THRESHOLD,
    STREAM_BACKGROUND_POLL_SECONDS,
    STREAM_HARD_SILENCE_SECONDS,
    STREAM_TRANSCRIPTION_INTERVAL_SECONDS,
    STREAM_TRANSCRIPT_AGREEMENT_COUNT,
    STREAM_SPEECH_CONTINUE_THRESHOLD_RATIO,
    STREAM_SPEECH_HANGOVER_SECONDS,
)

class StreamSpeechDetector:
    """Speech activity detector with separate start/continue thresholds."""

    def __init__(
        self,
        start_threshold: int,
        continue_threshold_ratio: float = STREAM_SPEECH_CONTINUE_THRESHOLD_RATIO,
        hangover_seconds: float = STREAM_SPEECH_HANGOVER_SECONDS,
    ) -> None:
        self.start_threshold = start_threshold
        self.continue_threshold = start_threshold * continue_threshold_ratio
        self.hangover_blocks = block_count_for_seconds(hangover_seconds)
        self.hangover_blocks_left = 0
        self.speech_started = False

    def is_speech(self, chunk: bytes) -> bool:
        """Return whether the chunk should count as speech for stream segmentation."""
        level = rms_level(chunk)
        if not self.speech_started:
            if level >= self.start_threshold:
                self.mark_speech()
                return True
            return False

        if level >= self.continue_threshold:
            self.mark_speech()
            return True

        if self.hangover_blocks_left > 0:
            self.hangover_blocks_left -= 1
            return True

        return False

    def mark_speech(self) -> None:
        """Record speech activity and refresh hangover protection."""
        self.speech_started = True
        self.hangover_blocks_left = self.hangover_blocks

    def reset(self) -> None:
        """Reset detector state for a new segment."""
        self.hangover_blocks_left = 0
        self.speech_started = False


class StreamSegmenter:
    """Continuously record voice-triggered utterance segments to WAV files."""

    def __init__(
        self,
        output_dir: Path,
        segment_queue: queue.Queue[CompletedStreamSegment | Exception],
        stop_event: threading.Event,
        transcriber: StreamingTranscriber,
        hard_silence_seconds: float = STREAM_HARD_SILENCE_SECONDS,
        silence_threshold: int = DEFAULT_SILENCE_THRESHOLD,
        semantic_endpoint_detector: SemanticEndpointDetector | None = None,
        transcription_interval_seconds: float = STREAM_TRANSCRIPTION_INTERVAL_SECONDS,
        transcript_agreement_count: int = STREAM_TRANSCRIPT_AGREEMENT_COUNT,
    ) -> None:
        self.output_dir = output_dir
        self.segment_queue = segment_queue
        self.stop_event = stop_event
        self.transcriber = transcriber
        self.hard_silence_seconds = hard_silence_seconds
        self.silence_threshold = silence_threshold
        self.semantic_endpoint_detector = semantic_endpoint_detector
        self.transcription_interval_blocks = block_count_for_seconds(
            transcription_interval_seconds
        )
        self.transcript_agreement_count = max(1, transcript_agreement_count)
        self.speech_detector = StreamSpeechDetector(silence_threshold)

        self.blocksize = audio_blocksize()
        self.background_poll_blocks = block_count_for_seconds(
            STREAM_BACKGROUND_POLL_SECONDS
        )
        self.hard_silence_blocks_needed = block_count_for_seconds(hard_silence_seconds)
        self.pre_roll = create_pre_roll_buffer()
        self.recorded_chunks: list[bytes] = []
        self.segment_chunks = 0
        self.silent_blocks = 0
        self.blocks_since_background_poll = 0
        self.pending_overflow_blocks = 0
        self.semantic_check_in_flight = False
        self.semantic_pause_index = 0
        self.last_transcription_job_chunks = 0
        self.last_seen_transcript_key = ""
        self.latest_transcript = ""
        self.transcript_agreements = 0
        self.locked_transcript = ""
        self.locked_transcript_key = ""
        self.locked_chunk_index = 0
        self.last_semantic_transcript_key = ""
        self.recording_started = False
        self.segment_index = 0
        self.wav_file: wave.Wave_write | None = None
        self.segment_path: Path | None = None
        self.pending_completion_reason = "shutdown"

        self.transcription_job_queue: queue.Queue[TranscriptionJob | None] | None = None
        self.transcription_result_queue: queue.Queue[TranscriptionResult] | None = None
        self.transcription_worker: threading.Thread | None = None
        self.semantic_job_queue: queue.Queue[SemanticEndpointJob | None] | None = None
        self.semantic_result_queue: queue.Queue[SemanticEndpointResult] | None = None
        self.semantic_worker: threading.Thread | None = None

    def run(self) -> None:
        """Run the recorder loop until stopped, queueing completed segments."""
        self.start_transcription_worker()
        self.start_semantic_worker()
        logger.info("Audio stream active. Waiting for audio input...")
        try:
            try:
                self.run_audio_stream()
            except Exception as error:
                self.segment_queue.put(error)
        finally:
            self.finish_active_segment()
            self.stop_transcription_worker()
            self.stop_semantic_worker()

    def start_transcription_worker(self) -> None:
        """Start the single streaming ASR worker."""
        self.transcription_job_queue = queue.Queue()
        self.transcription_result_queue = queue.Queue()
        self.transcription_worker = threading.Thread(
            target=run_transcription_worker,
            args=(
                self.output_dir,
                self.transcription_job_queue,
                self.transcription_result_queue,
                self.transcriber,
            ),
        )
        self.transcription_worker.start()

    def stop_transcription_worker(self) -> None:
        """Stop the streaming ASR worker if it was started."""
        if self.transcription_job_queue is not None:
            self.transcription_job_queue.put(None)
        if self.transcription_worker is not None:
            self.transcription_worker.join()

    def start_semantic_worker(self) -> None:
        """Start the semantic endpoint worker when a detector is configured."""
        if self.semantic_endpoint_detector is None:
            return

        self.semantic_job_queue = queue.Queue()
        self.semantic_result_queue = queue.Queue()
        self.semantic_worker = threading.Thread(
            target=run_semantic_endpoint_worker,
            args=(
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
                    self.pending_overflow_blocks += 1
                else:
                    self.flush_overflow_warning()

                self.handle_audio_chunk(chunk)
        self.flush_overflow_warning()

    def handle_audio_chunk(self, chunk: bytes) -> None:
        """Process one raw audio chunk."""
        is_speech = self.speech_detector.is_speech(chunk)

        if not self.recording_started:
            self.pre_roll.append(chunk)
            if is_speech:
                self.start_segment()
            return

        self.write_segment_chunks([chunk])
        self.update_silence_state(is_speech)
        self.blocks_since_background_poll += 1

        if self.should_poll_background_results():
            self.blocks_since_background_poll = 0
            if self.handle_transcription_results():
                return
            if self.handle_semantic_endpoint_results():
                return

        if self.should_queue_transcription_check():
            self.queue_transcription_check()

        if self.silent_blocks >= self.hard_silence_blocks_needed:
            self.accept_hard_silence()

    def start_segment(self) -> None:
        """Start writing a new stream segment."""
        logger.info("Audio input detected. Listening...")
        self.segment_index += 1
        self.segment_path = self.output_dir / f"stream-segment-{self.segment_index:04d}.wav"
        self.wav_file = open_wav_writer(self.segment_path)
        self.recording_started = True
        self.write_segment_chunks(self.pre_roll)
        self.segment_chunks = len(self.pre_roll)
        self.silent_blocks = 0
        self.semantic_check_in_flight = False
        self.last_transcription_job_chunks = 0
        self.last_seen_transcript_key = ""
        self.latest_transcript = ""
        self.transcript_agreements = 0
        self.locked_transcript = ""
        self.locked_transcript_key = ""
        self.locked_chunk_index = 0
        self.last_semantic_transcript_key = ""
        self.speech_detector.mark_speech()
        self.blocks_since_background_poll = 0

    def finish_active_segment(self) -> None:
        """Queue the active segment during shutdown, if any."""
        if self.recording_started:
            self.finish_segment()

    def finish_segment_and_reset(self) -> None:
        """Finish the current segment and reset recorder state."""
        self.finish_segment()
        self.recording_started = False
        self.silent_blocks = 0
        self.semantic_check_in_flight = False
        self.last_transcription_job_chunks = 0
        self.last_seen_transcript_key = ""
        self.latest_transcript = ""
        self.transcript_agreements = 0
        self.locked_transcript = ""
        self.locked_transcript_key = ""
        self.locked_chunk_index = 0
        self.last_semantic_transcript_key = ""
        self.pre_roll.clear()
        self.speech_detector.reset()
        self.blocks_since_background_poll = 0
        logger.info("Waiting for audio input...")

    def finish_segment(self) -> None:
        """Close and queue the current segment path."""
        if self.wav_file is None or self.segment_path is None:
            return

        self.wav_file.close()
        self.segment_queue.put(
            CompletedStreamSegment(
                path=self.segment_path,
                completion_reason=self.pending_completion_reason,
                transcript=self.latest_transcript,
            )
        )
        self.wav_file = None
        self.segment_path = None
        self.recorded_chunks = []
        self.segment_chunks = 0
        self.pending_completion_reason = "shutdown"
        self.latest_transcript = ""
        self.locked_transcript = ""
        self.locked_transcript_key = ""
        self.locked_chunk_index = 0
        self.last_semantic_transcript_key = ""

    def write_segment_chunks(self, chunks: Iterable[bytes]) -> None:
        """Write raw chunks into the current segment and snapshot buffer."""
        if self.wav_file is None:
            return

        for item in chunks:
            self.wav_file.writeframes(item)
            self.recorded_chunks.append(item)
            self.segment_chunks += 1

    def current_silence_seconds(self) -> float:
        """Return the current consecutive silence duration."""
        return self.silent_blocks * AUDIO_CHUNK_SECONDS

    def update_silence_state(self, is_speech: bool) -> None:
        """Update silence counters and pause identity."""
        if is_speech:
            if self.silent_blocks > 0 or self.semantic_check_in_flight:
                self.semantic_pause_index += 1
            self.silent_blocks = 0
            self.semantic_check_in_flight = False
        else:
            self.silent_blocks += 1

    def should_poll_background_results(self) -> bool:
        """Poll worker result queues in small batches instead of every audio block."""
        if self.blocks_since_background_poll >= self.background_poll_blocks:
            return True
        return self.silent_blocks > 0 or self.semantic_check_in_flight

    def flush_overflow_warning(self) -> None:
        """Log one aggregated warning for one burst of input overflows."""
        if self.pending_overflow_blocks <= 0:
            return
        overflow_seconds = self.pending_overflow_blocks * AUDIO_CHUNK_SECONDS
        logger.warning(
            "Audio warning: input overflowed {} block(s) over {:.1f}s.",
            self.pending_overflow_blocks,
            overflow_seconds,
        )
        self.pending_overflow_blocks = 0

    def should_queue_transcription_check(self) -> bool:
        """Return whether enough new audio has arrived for another ASR snapshot."""
        return (
            self.transcription_job_queue is not None
            and self.segment_path is not None
            and not self.transcription_worker_has_pending_jobs()
            and self.segment_chunks - self.last_transcription_job_chunks
            >= self.transcription_interval_blocks
        )

    def queue_transcription_check(self) -> None:
        """Queue one single-path ASR job for the current segment snapshot."""
        if self.transcription_job_queue is None:
            return

        self.transcription_job_queue.put(
            TranscriptionJob(
                segment_index=self.segment_index,
                pause_index=self.semantic_pause_index,
                start_chunk_index=self.locked_chunk_index,
                end_chunk_index=self.segment_chunks,
                chunks=list(self.recorded_chunks[self.locked_chunk_index : self.segment_chunks]),
            )
        )
        self.last_transcription_job_chunks = self.segment_chunks

    def transcription_worker_has_pending_jobs(self) -> bool:
        """Return whether the ASR worker queue already has pending work."""
        return (
            self.transcription_job_queue is not None
            and not self.transcription_job_queue.empty()
        )

    def semantic_worker_has_pending_jobs(self) -> bool:
        """Return whether the semantic worker queue already has pending work."""
        return self.semantic_job_queue is not None and not self.semantic_job_queue.empty()

    def accept_hard_silence(self) -> None:
        """Force the segment to end once hard silence is reached."""
        silence_seconds = self.current_silence_seconds()
        logger.info(
            "Audio segment trigger: hard silence after {:.1f}s.",
            silence_seconds,
        )
        self.pending_completion_reason = f"{silence_seconds:.1f}s hard silence"
        self.finish_segment_and_reset()

    def handle_transcription_results(self) -> bool:
        """Apply streaming ASR results without blocking capture."""
        if self.transcription_result_queue is None:
            return False

        while True:
            try:
                result = self.transcription_result_queue.get_nowait()
            except queue.Empty:
                return False

            if not self.recording_started:
                continue
            if self.semantic_result_is_stale(result):
                continue

            transcript = self.cleaned_transcript(result)
            if transcript or self.locked_transcript:
                full_transcript = self.combined_transcript(transcript)
                self.latest_transcript = full_transcript
                self.update_transcript_agreement(full_transcript)
                self.maybe_lock_confirmed_transcript(full_transcript, result.end_chunk_index)
            else:
                self.update_transcript_agreement("")

            if self.should_queue_semantic_check():
                self.queue_semantic_endpoint_check()
        return False

    def cleaned_transcript(self, result: TranscriptionResult) -> str:
        """Return a cleaned transcript for agreement tracking and final output."""
        if result.is_rejected:
            return ""
        return result.transcript

    def update_transcript_agreement(self, transcript: str) -> None:
        """Update transcript agreement state using a normalized transcript key."""
        transcript_key = self.normalized_transcript_key(transcript)
        if transcript_key and transcript_key == self.last_seen_transcript_key:
            self.transcript_agreements += 1
        elif transcript_key:
            self.last_seen_transcript_key = transcript_key
            self.transcript_agreements = 1
        else:
            self.transcript_agreements = 0

    def combined_transcript(self, transcript: str) -> str:
        """Combine locked transcript prefix with freshly transcribed suffix."""
        if not self.locked_transcript:
            return transcript
        if not transcript:
            return self.locked_transcript
        return f"{self.locked_transcript} {transcript}"

    def maybe_lock_confirmed_transcript(
        self,
        transcript: str,
        end_chunk_index: int,
    ) -> None:
        """Lock transcript once it reaches agreement so it is never re-transcribed."""
        transcript_key = self.normalized_transcript_key(transcript)
        if not transcript_key:
            return
        if self.transcript_agreements < self.transcript_agreement_count:
            return
        if transcript_key == self.locked_transcript_key:
            return
        if end_chunk_index <= self.locked_chunk_index:
            return

        self.locked_transcript = transcript
        self.locked_transcript_key = transcript_key
        self.locked_chunk_index = end_chunk_index
        logger.debug(
            "Locked transcript through chunk {} after {} agreements.",
            self.locked_chunk_index,
            self.transcript_agreements,
        )

    def normalized_transcript_key(self, transcript: str) -> str:
        """Return a normalized transcript key for agreement comparison."""
        words = [match.group(0) for match in TRANSCRIPT_WORD_PATTERN.finditer(transcript.lower())]
        return " ".join(words)

    def should_queue_semantic_check(self) -> bool:
        """Return whether new stabilized transcript text is ready for classification."""
        return (
            bool(self.locked_transcript_key)
            and self.locked_transcript_key != self.last_semantic_transcript_key
            and not self.semantic_worker_has_pending_jobs()
            and not self.semantic_check_in_flight
        )

    def queue_semantic_endpoint_check(self) -> None:
        """Queue semantic classification for the latest stabilized transcript text."""
        if self.semantic_job_queue is None:
            return
        self.semantic_check_in_flight = True
        self.last_semantic_transcript_key = self.locked_transcript_key
        self.semantic_job_queue.put(
            SemanticEndpointJob(
                segment_index=self.segment_index,
                pause_index=self.semantic_pause_index,
                transcript=self.locked_transcript,
                transcript_key=self.locked_transcript_key,
            )
        )
        logger.info(
            "Semantic check queued for stabilized transcript with n agreement = {}.",
            self.transcript_agreements,
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
            self.semantic_check_in_flight = False

            if result.is_rejected:
                logger.debug(
                    "Semantic endpoint transcript ignored as rejected: {}",
                    result.transcript,
                )
                continue

            if not result.is_complete:
                logger.info("Semantic INCOMPLETE for current transcript buffer.")
                if result.transcript_key == self.locked_transcript_key:
                    self.last_semantic_transcript_key = ""
                continue

            if result.transcript_key and result.transcript_key != self.locked_transcript_key:
                logger.info("Semantic COMPLETE ignored because transcript buffer changed.")
                continue

            logger.info("Semantic COMPLETE for current transcript buffer.")
            self.pending_completion_reason = "semantic completion from transcript buffer"
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
    segment_queue: queue.Queue[CompletedStreamSegment | Exception],
    stop_event: threading.Event,
    transcriber: StreamingTranscriber,
    hard_silence_seconds: float = STREAM_HARD_SILENCE_SECONDS,
    silence_threshold: int = DEFAULT_SILENCE_THRESHOLD,
    semantic_endpoint_detector: SemanticEndpointDetector | None = None,
    transcription_interval_seconds: float = STREAM_TRANSCRIPTION_INTERVAL_SECONDS,
    transcript_agreement_count: int = STREAM_TRANSCRIPT_AGREEMENT_COUNT,
) -> None:
    """Continuously record voice-triggered utterance segments to WAV files."""
    StreamSegmenter(
        output_dir=output_dir,
        segment_queue=segment_queue,
        stop_event=stop_event,
        transcriber=transcriber,
        hard_silence_seconds=hard_silence_seconds,
        silence_threshold=silence_threshold,
        semantic_endpoint_detector=semantic_endpoint_detector,
        transcription_interval_seconds=transcription_interval_seconds,
        transcript_agreement_count=transcript_agreement_count,
    ).run()
