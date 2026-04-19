# SecondVoice Design Doc

SecondVoice is a local mock-interview assistant. It listens to microphone audio, splits speech into interview segments, transcribes each segment locally, optionally estimates whether the speaker sounds like the enrolled interviewee, optionally attaches interview-board photos, and submits the resulting prompt to ChatGPT for classification and feedback.

The application is intentionally small and file-oriented. `main.py` handles command-line setup, while `src/app.py` coordinates the runtime workflow across focused modules for audio capture, transcription, speaker identification, camera capture, browser automation, constants, and logging.

## Goals

- Capture mock-interview speech with minimal manual control.
- Keep transcription local through `faster-whisper`.
- Preserve interviewer context across ChatGPT submissions by sending an initial mode prompt and then incremental transcript segments.
- Help ChatGPT distinguish interviewer and interviewee speech with an optional enrolled voice hint.
- Add visual problem-board context through test or live camera photos when enabled.
- Reuse a logged-in ChatGPT browser session through Chrome DevTools Protocol.

## Non-Goals

- The app does not implement its own general-purpose LLM backend. It uses a small local Ollama model only for endpoint completion classification.
- The app does not store interview transcripts or ChatGPT responses in structured history.
- The app does not do local speaker diarization. It only provides a similarity hint against one enrolled interviewee profile.
- The app does not own the ChatGPT UI. Browser automation depends on current ChatGPT selectors and may need maintenance if the UI changes.

## Runtime Paths

### 1. Stream Mode

Command:

```sh
python main.py
```

This is the default path.

1. `main.py` configures logging, parses CLI flags, and calls `app.run()`.
2. `app.run()` enters `stream_loop()` unless enrollment mode is requested.
3. `stream_loop()` creates a temporary directory for WAV segments.
4. `start_stream_recorder()` starts a background thread running `audio.stream_utterance_segments()`.
5. If photo capture is enabled, `start_photo_timer()` starts a second background thread running `capture_photos_on_interval()`.
6. The main thread loads the configured endpoint and final transcribers, `SpeakerIdentifier`, and `PhotoUploadTracker`.
7. During recording, the audio loop queues a semantic endpoint job after a 3-second pause. A semantic worker thread draft-transcribes the snapshot with the fast endpoint Whisper model, sends that transcript to local Ollama `qwen2.5:1.5b`, and returns a result. The recorder keeps reading microphone input while that happens and cuts the segment early only when the current, non-stale result is `COMPLETE`. If the detector returns incomplete or fails, the hard 10-second silence fallback still cuts the segment.
8. For every queued WAV segment, `process_stream_segment()`:
   - Builds a speaker hint from the WAV file.
   - Transcribes the WAV file locally.
   - Deletes the temporary WAV segment.
   - Prints the transcript and speaker hint.
   - Builds a ChatGPT prompt.
   - Attaches a photo only when the selected photo exists and changed since the last successful upload.
   - Submits the prompt through Playwright browser automation.
9. `Ctrl+C` stops the recorder and photo timer threads cleanly.

The first successful ChatGPT submission includes the full `STREAM_PROMPT`, which defines SecondVoice's behavior. Later submissions include only the segment prompt so the current ChatGPT conversation can maintain context.

### 2. Transcript-Only Mode

Command:

```sh
python main.py --no-ask
```

This follows the same audio capture, transcription, and speaker-hint path as stream mode, but skips `browser.submit_to_chatgpt()`. It is useful for testing microphone capture, silence thresholds, Whisper transcription, and voice matching without touching ChatGPT.

### 3. Voice Enrollment Path

Command:

```sh
python main.py --enroll
```

This path records an interviewee voice profile.

1. `app.run()` calls `enroll_interviewee_voice()`.
2. The app prompts the user to read each sentence in `SPEAKER_ENROLLMENT_PROMPTS`.
3. `audio.capture_enrollment_utterance()` records each sentence into a temporary WAV file.
4. `SpeakerIdentifier.enroll_from_clips()` encodes each clip with SpeechBrain, averages the embeddings, normalizes the result, and persists it.
5. The profile embedding is saved to `~/.secondvoice/interviewee-voice-embedding.pt`.
6. Metadata is saved to `~/.secondvoice/interviewee-voice-profile.json`.

Streaming mode can then include a confidence value showing how similar each segment sounds to the enrolled interviewee.

### 4. Photo Context Paths

The `--photo-mode` flag controls whether the main app captures and uploads photos.

```sh
python main.py --photo-mode none
python main.py --photo-mode test
python main.py --photo-mode live
```

- `none`: disables all photo capture and upload. This is the default and is useful with `--no-ask` for transcription-only testing.
- `test`: captures `/Users/flora/interview/test.jpg` after 60 seconds and then every 60 seconds.
- `live`: captures `/Users/flora/interview/live.jpg` after 10 minutes and then every 10 minutes.

`PhotoUploadTracker` stores the last uploaded photo signature as `(mtime_ns, size)`. If the selected photo is missing, empty, or unchanged, the app submits text only.

### 5. Manual ChatGPT Submission Test

Command:

```sh
python scripts/test_chatgpt_submit.py
```

This script opens the automation Chrome profile on macOS only when the CDP endpoint is not already running, builds a stream prompt from a fixed two-sum-style transcript, and submits it to ChatGPT. It bypasses microphone recording, local transcription, and interactive transcript entry, so it is useful for testing browser automation and fixed photo upload behavior. It does not accept a photo mode; it always attaches `/Users/flora/interview/static.jpg`.

### 6. Standalone Camera Capture

Command:

```sh
python src/camera.py
```

This captures one image with `imagesnap` from the named macOS camera and writes it to the live photo path. The default camera is `"FaceTime HD Camera"`.

## Main Data Flow

```text
Microphone
  -> audio.stream_utterance_segments()
  -> temporary WAV segment
  -> app.process_stream_segment()
  -> speaker_id.SpeakerIdentifier.match()
  -> transcription.Transcriber.transcribe()
  -> app.build_stream_prompt()
  -> optional photo selected by app.next_photo_upload()
  -> browser.submit_to_chatgpt()
  -> ChatGPT conversation
```

Enrollment has a separate data flow:

```text
Microphone
  -> audio.capture_enrollment_utterance()
  -> temporary enrollment WAV clips
  -> speaker_id.SpeakerIdentifier.enroll_from_clips()
  -> ~/.secondvoice voice profile files
```

## File and Method Reference

### `main.py`

Entrypoint for the command-line app.

- `main()`: Configures Loguru logging and starts the application with parsed runtime options.
- `parse_args()`: Defines CLI flags and returns `RuntimeOptions`.

Current CLI flags:

- `--no-ask`: Turns off ChatGPT submission while keeping capture and transcription active.
- `--enroll`: Runs voice enrollment instead of stream mode.
- `--photo-mode`: Selects `none`, `test`, or `live` photo behavior.

### `src/app.py`

The orchestration layer. It owns high-level runtime paths, threading, prompt construction, photo decisions, and console output.

- `RuntimeOptions`: Immutable dataclass carrying CLI-selected behavior.
- `PhotoUploadTracker`: Mutable dataclass storing the last uploaded photo signature.
- `run(options)`: Dispatches to enrollment or stream mode.
- `enroll_interviewee_voice()`: Records prompted enrollment clips and persists a voice profile.
- `stream_loop(options)`: Runs the continuous capture/transcribe/submit loop.
- `start_stream_recorder(output_dir, segment_queue, stop_event)`: Starts audio recording in a background thread.
- `start_photo_timer(stop_event, photo_mode)`: Starts periodic photo capture in a background thread.
- `photo_capture_settings(photo_mode)`: Maps `test` and `live` modes to path, initial delay, and interval.
- `capture_photos_on_interval(stop_event, photo_path, initial_seconds, interval_seconds)`: Captures photos until stopped.
- `next_stream_segment(segment_queue)`: Polls the recorder queue and raises recorder exceptions on the main thread.
- `process_stream_segment(...)`: Handles one completed WAV segment end to end.
- `next_photo_upload(photo_mode, photo_tracker)`: Selects a photo only if it is available and changed.
- `current_photo_signature(photo_path)`: Returns `(mtime_ns, size)` for a valid photo file.
- `interview_photo_path(photo_mode)`: Maps `test` and `live` modes to their fixed photo paths; `none` returns no path.
- `build_speaker_hint(audio_path, speaker_identifier)`: Produces a `SpeakerHint`, falling back to `unknown` if voice matching fails.
- `build_stream_prompt(...)`: Builds the prompt sent to ChatGPT.
- `speaker_hint_value(speaker_hint)`: Formats confidence for the prompt.
- `speaker_hint_role(speaker_hint)`: Formats role hint for the prompt.
- `print_enrollment_prompt(index, total, prompt)`: Logs one enrollment sentence.
- `print_speaker_hint(speaker_hint)`: Logs speaker-match information.
- `print_stream_mode_banner(options)`: Logs active runtime settings.
- `print_transcript(transcript)`: Logs the transcript or a no-speech marker.

### `src/audio/`

Audio capture, segmentation, WAV writing, and amplitude helpers. `src/audio/__init__.py` re-exports the public functions used by the app, so callers can continue to import from `audio`.

#### `src/audio/enrollment.py`

- `capture_enrollment_utterance(...)`: Records one voice-enrollment sentence using speech-start and silence-stop triggers.

#### `src/audio/segmenter.py`

- `SemanticEndpointJob`: Draft chunk snapshot submitted from the recorder thread to the semantic worker.
- `SemanticEndpointResult`: Completion decision sent from the semantic worker back to the recorder thread.
- `StreamSegmenter`: Owns stream-recording mutable state, including open WAV file, segment index, pause index, silence counters, pre-roll, semantic queues, and segment finalization.
- `stream_utterance_segments(...)`: Compatibility wrapper that creates a `StreamSegmenter` and runs it.
- `run_semantic_endpoint_worker(...)`: Consumes semantic draft jobs, writes temporary draft WAVs, runs the detector, and publishes results without blocking microphone capture.

#### `src/audio/wav.py`

- `write_wav_file(output_path, chunks)`: Writes raw chunks into a complete WAV file, used for semantic endpoint draft snapshots.
- `open_wav_writer(output_path)`: Opens a mono int16 WAV writer using app audio constants.
- `write_chunks(wav_file, chunks)`: Writes raw audio chunks into a WAV file.

#### `src/audio/levels.py`

- `audio_blocksize()`: Converts chunk duration into sample count.
- `block_count_for_seconds(seconds)`: Converts seconds into audio chunk count.
- `create_pre_roll_buffer()`: Keeps a small buffer of audio before speech starts, so segment starts are not clipped.
- `chunk_is_speech(chunk, threshold)`: Uses RMS amplitude to classify a chunk as speech or silence.
- `rms_level(chunk)`: Computes RMS amplitude for int16 audio samples.

### `src/endpoint_detector.py`

Local semantic endpointing through Ollama.

- `OllamaSemanticEndpointDetector`: Holds the configured Ollama model name and the fast endpoint `Transcriber`. Until the transcriber is attached, checks conservatively return incomplete.
- `OllamaSemanticEndpointDetector.set_transcriber(transcriber)`: Attaches the loaded Whisper transcriber after app startup.
- `OllamaSemanticEndpointDetector.is_complete(audio_path)`: Transcribes a draft WAV snapshot, classifies the transcript through Ollama, logs the result, and returns `True` only for `COMPLETE`.
- `classify_endpoint_transcript(transcript, model, url, timeout_seconds)`: Sends the shared endpoint prompt to the Ollama chat API and returns the normalized label plus latency.

### `src/transcription.py`

Local transcription backend interface and implementations.

- `Transcriber`: Protocol shared by transcription backends.
- `create_transcriber(backend, model)`: Factory that returns the configured backend implementation.
- `model_path_for_run(backend, model, use_local_cache)`: Uses repo names directly for refresh/test runs, but resolves MLX Hugging Face repo names to local cached snapshot paths for normal ChatGPT runs.
- `cached_huggingface_snapshot_path(repo_id)`: Finds the newest local Hugging Face cache snapshot for an MLX model without refreshing metadata.
- `FasterWhisperTranscriber`: Loads and runs `faster-whisper`, currently used for fast draft endpoint transcription.
- `MlxWhisperTranscriber`: Runs `mlx-whisper`, currently used for final completed-segment transcription on Apple Silicon.
- `transcribe(audio_path)`: Backend implementations transcribe a WAV file with the coding-interview and system-design initial prompt where supported, then return joined plain text. Calls are serialized per transcriber instance.

### `src/speaker_id.py`

Interviewee voice enrollment and matching.

- `SpeakerHint`: Dataclass carrying `role_hint`, prompt confidence, raw similarity, and profile availability.
- `SpeakerHint.prompt_value()`: Formats confidence for ChatGPT.
- `SpeakerHint.log_value()`: Formats speaker hint details for logs.
- `SpeakerIdentifier.__init__(log_missing_profile)`: Loads an existing profile if available.
- `SpeakerIdentifier.has_profile`: Returns whether a profile is loaded.
- `SpeakerIdentifier.enroll_from_clips(audio_paths)`: Creates and saves an averaged normalized voice embedding.
- `SpeakerIdentifier.match(audio_path)`: Compares a segment embedding to the profile and returns a `SpeakerHint`.
- `SpeakerIdentifier._encode(audio_path)`: Loads audio and returns a normalized embedding.
- `SpeakerIdentifier._classifier()`: Lazily loads the SpeechBrain encoder model.
- `SpeakerIdentifier._load_profile()`: Loads the saved embedding from disk.
- `SpeakerIdentifier._save_metadata(clip_count)`: Writes voice-profile metadata JSON.
- `SpeakerIdentifier._torch()`: Imports `torch` with a user-facing error if dependencies are missing.

### `src/browser.py`

Playwright automation for submitting prompts to ChatGPT.

- `BrowserSession`: Dataclass that owns browser context resources.
- `BrowserSession.close()`: Closes resources only when this process owns them.
- `submit_to_chatgpt(prompt, photo_path, cdp_url)`: Opens ChatGPT, attaches an optional photo, fills the prompt, submits it, and brings Chrome forward on macOS.
- `connect_to_cdp_browser(playwright, cdp_url)`: Connects to an existing Chrome over CDP.
- `open_chatgpt_page(context)`: Reuses the dedicated marked SecondVoice ChatGPT tab or opens and marks a new one.
- `is_secondvoice_chatgpt_page(page)`: Checks whether a ChatGPT tab has the SecondVoice tab marker.
- `mark_secondvoice_chatgpt_page(page)`: Marks one ChatGPT tab as the dedicated SecondVoice automation tab and adds a visible title prefix plus in-page badge.
- `activate_chrome()`: Brings the CDP automation Chrome process to the foreground on macOS, with a generic Chrome activation fallback.
- `automation_chrome_pid()`: Finds the main Chrome process launched with the automation profile and CDP port.
- `activate_process(pid)`: Activates one macOS process by PID through System Events.
- `stabilize_chatgpt_theme(page)`: Pins the automation page to the dark color scheme before submission.
- `fill_prompt(prompt_box, prompt)`: Writes the prompt into the composer.
- `submit_prompt(page, prompt_box, wait_for_upload)`: Sends the prompt by button click or Enter fallback.
- `find_send_button(page)`: Tries known ChatGPT send-button selectors.
- `attach_file(page, file_path)`: Attaches a local file through file inputs or visible upload controls.
- `wait_for_attachment_upload(page, timeout)`: Waits briefly for upload completion.
- `find_prompt_box(page, timeout)`: Tries known ChatGPT composer selectors.

### `src/camera.py`

macOS still-photo capture.

- `CameraCaptureError`: Raised when photo capture fails.
- `take_photo(output_path, camera_name)`: Uses `imagesnap` to capture one photo to a target path.
- `parse_args()`: Defines standalone camera CLI flags.
- `main()`: Captures one photo and prints the saved path.

### `src/constants.py`

Central configuration values.

Important groups:

- Audio shape: sample rate, channels, sample width, chunk size, pre-roll.
- Segmentation: silence duration and RMS threshold.
- Transcription: configurable endpoint and final Whisper backends/models, plus separate coding-interview and system-design vocabulary prompts for terms like BFS, DFS, topological sort, idempotency, relational databases, consistency, queues, reliability, and common architecture phrases.
- Photo paths and capture intervals.
- ChatGPT browser URL and CDP URL.
- Speaker-profile paths, model source, threshold, and enrollment prompts.
- `STREAM_PROMPT`, the system-style instruction sent on the first ChatGPT submission.

### `src/logging_config.py`

Loguru setup.

- `configure_logging(log_path)`: Writes human-readable INFO logs to stderr and DEBUG logs to `python.log`.

### `scripts/test_chatgpt_submit.py`

Manual browser-submission harness.

- `SMOKE_TEST_TRANSCRIPT`: Fixed two-sum-style transcript used for browser smoke testing.
- `main()`: Ensures the smoke-test browser path is ready, builds a prompt from `SMOKE_TEST_TRANSCRIPT`, and submits it to ChatGPT with `/Users/flora/interview/static.jpg`.
- `open_automation_chrome()`: On macOS, opens Google Chrome with remote debugging and the `~/.secondvoice/cdp-browser-profile` profile only when CDP is not already reachable.
- `cdp_browser_is_running()`: Checks whether the Chrome CDP endpoint is already reachable before launching a browser.

### `scripts/test_endpoint_detector.py`

Ollama semantic-endpoint benchmark harness.

- `main()`: Runs the fixed endpoint-completion case set against one or more Ollama models.
- `prompt_for_models()`: Asks for model names when none are provided on the command line.
- `run_model_benchmark(model)`: Prints per-case predictions and summary latency/error counts.
- `classify(model, transcript)`: Calls the shared `endpoint_detector.classify_endpoint_transcript()` helper so benchmarks and runtime use the same prompt and Ollama request shape.

### `scripts/benchmark_transcriber.py`

Local transcription backend benchmark harness.

- `main()`: Creates one configured transcriber and runs it against one or more WAV files.
- `parse_args()`: Accepts audio paths plus optional backend/model overrides so `faster-whisper` and `mlx-whisper` models can be compared without changing the live app.

## Threading Model

Stream mode uses the main thread for transcription, voice matching, prompt construction, and browser submission. Audio capture runs in a background thread so recording can continue while segments are processed. Photo capture can run in a second background thread for `test` and `live` modes.

Communication between the recorder and main thread is via `queue.Queue[Path | Exception]`. Completed segments are queued as `Path` objects. Recorder failures are queued as exceptions and re-raised by `next_stream_segment()`.

Endpointing is hybrid. The recorder queues a semantic endpoint job after `STREAM_SEMANTIC_SILENCE_SECONDS`, currently 3 seconds, but it does not run Whisper or Ollama on the recorder thread. The job contains a copy of the current chunk snapshot plus segment and pause IDs. A semantic worker thread writes a draft WAV snapshot, transcribes that draft locally with the fast endpoint transcription backend, and asks Ollama `qwen2.5:1.5b` whether the transcript is `COMPLETE` or `INCOMPLETE` using the same prompt benchmarked by `scripts/test_endpoint_detector.py`.

The recorder polls semantic results without blocking microphone reads. It accepts only results whose segment and pause IDs still match the current in-progress segment, so stale `COMPLETE` results are ignored if the speaker resumed talking or the hard fallback already cut the segment. The detector is intentionally conservative at the integration boundary: if the transcriber is not loaded yet, Ollama is not reachable, the draft transcript is empty, or the model returns anything other than `COMPLETE`, the recorder keeps listening. If no semantic endpoint is accepted, the recorder falls back to `STREAM_HARD_SILENCE_SECONDS`, currently 10 seconds, and cuts the segment.

Shutdown is coordinated with a shared `threading.Event`. `KeyboardInterrupt` sets the event, joins background threads, and logs completion.

## Audio Queue Design

Audio recording has two separate queue-based handoffs. They both involve the recorder thread, but they solve different problems.

### Completed Segment Queue

The completed segment queue connects the recorder thread to the main app thread.

```text
Recorder thread
  captures microphone audio
  decides a segment is finished
  closes the segment WAV
  puts the WAV path into segment_queue
        |
        v
Main app thread
  reads segment_queue
  runs final transcription
  runs speaker matching
  prints or submits the segment
```

This queue carries finished work. Its items are either completed WAV paths or recorder exceptions. The recorder should not do final transcription, speaker matching, prompt construction, or browser automation because those operations are slower than real-time microphone capture.

### Semantic Endpoint Queues

Semantic endpointing uses a second queue pair inside the audio layer.

```text
Recorder thread
  keeps reading microphone audio
  after a 3-second pause, copies the current chunks
  puts SemanticEndpointJob into semantic_job_queue
        |
        v
Semantic worker thread
  writes a temporary draft WAV
  runs fast Whisper draft transcription
  asks Ollama COMPLETE vs INCOMPLETE
  puts SemanticEndpointResult into semantic_result_queue
        |
        v
Recorder thread
  polls semantic_result_queue without blocking
  cuts the segment only if the current result is COMPLETE
```

This queue pair carries draft work. It exists so Whisper and Ollama latency cannot block `sounddevice.RawInputStream.read()`. Blocking the recorder thread during endpoint detection can let the microphone input buffer overflow, which may drop audio samples.

Each semantic job/result includes a `segment_index` and `pause_index`. The recorder accepts a result only when both IDs still match the current in-progress segment. This prevents stale results from cutting audio after the speaker resumes talking, after a newer pause begins, or after the hard fallback has already ended the segment.

This means SecondVoice may sometimes avoid cutting a segment that was complete at an earlier pause. For example, if a semantic job is queued after `"I would sort the array first"` but the speaker resumes with `"then I would use two pointers"` before the worker returns, the old result is ignored even if it says `COMPLETE`. The tradeoff is intentional: a slightly later cut or larger segment is better than letting stale information split resumed speech in the middle of a continuing answer.

### Semantic Timeline Examples

If the speaker stays silent after the first semantic check, an early cut can happen before the 10-second fallback:

```text
t+0.0  silence starts
t+3.0  semantic job queued for segment 5, pause 10
t+3.5  semantic result returns COMPLETE for segment 5, pause 10
t+3.5  recorder accepts the current result and cuts the segment
```

If the speaker resumes before the semantic result returns, the old result is stale and ignored:

```text
t+0.0  silence starts
t+3.0  semantic job queued for segment 5, pause 10
t+3.1  speaker resumes; recorder increments pause_index to 11
t+4.0  old semantic result returns COMPLETE for segment 5, pause 10
t+4.0  recorder ignores the stale result
```

If the speaker pauses again after resuming, the recorder queues a new semantic check for the new pause:

```text
t+0.0  first silence starts
t+3.0  semantic job queued for segment 5, pause 10
t+3.1  speaker resumes; recorder increments pause_index to 11
t+4.0  old result for pause 10 is ignored as stale
t+5.0  second silence starts
t+8.0  semantic job queued for segment 5, pause 11
```

The 10-second hard fallback is also counted from the current continuous silence. In the last example, if no current semantic result is accepted, fallback would happen around `t+15.0`, not `t+10.0`, because speech resumed at `t+3.1`.

The semantic worker never mutates recorder-owned state. It does not close segment files, clear buffers, or decide final boundaries directly. The recorder remains the single owner of mutable recording state and the only code path that calls `finish_segment()`.

## External Dependencies

- `sounddevice`: microphone input.
- `faster-whisper`: local transcription.
- `speechbrain`, `torch`, `torchaudio`: voice embedding and matching.
- `playwright`: ChatGPT browser automation.
- `loguru`: structured console and file logging.
- `imagesnap`: external macOS command for named-camera still capture.
- Google Chrome with remote debugging enabled for the default browser submission path. The real stream path expects this CDP browser to already be running; the smoke test opens it on macOS if it is missing.

## Persistent and Temporary State

Persistent state:

- `python.log`: debug log in the current working directory.
- `~/.secondvoice/interviewee-voice-embedding.pt`: enrolled interviewee embedding.
- `~/.secondvoice/interviewee-voice-profile.json`: metadata for the enrolled voice profile.
- `~/.secondvoice/cdp-browser-profile`: recommended Chrome profile for reusable ChatGPT login.
- `/Users/flora/interview/test.jpg`, `live.jpg`: main app photo-context inputs.
- `/Users/flora/interview/static.jpg`: fixed photo fixture for `scripts/test_chatgpt_submit.py`.

Temporary state:

- Stream WAV segments are created in a `secondvoice-eval-*` temporary directory and deleted after transcription.
- Enrollment WAV clips are created in a `secondvoice-enroll-*` temporary directory and discarded after profile creation.

## Error Handling

- Missing voice profile does not stop streaming; the prompt receives an unavailable speaker hint.
- Voice-matching dependency failures are logged and converted into an unknown speaker hint during streaming.
- Recorder-thread exceptions are passed through the queue and raised on the main thread.
- Missing or unchanged photos result in text-only ChatGPT submissions.
- Camera capture failures are logged as warnings during periodic capture.
- Browser connection or prompt-box failures stop the run with actionable setup guidance.

## Maintenance Notes

- Browser selectors in `src/browser.py` are intentionally defensive, but ChatGPT UI changes can still break submission or upload.
- Photo paths are hard-coded for `/Users/flora/interview`; changing this should be centralized in `src/constants.py`.
