# SecondVoice Design Doc

SecondVoice is a local mock-interview assistant. It listens to microphone audio, splits speech into interview segments, transcribes each segment locally, optionally estimates whether the speaker sounds like the enrolled interviewee, optionally attaches interview-board photos, and submits the resulting prompt to ChatGPT for classification and feedback.

The application is intentionally small and file-oriented. `main.py` handles command-line setup, `src/preflight.py` checks local dependencies, and `src/app.py` coordinates the stream runtime across focused packages for audio capture, speech processing, prompt construction, photo handling, camera capture, browser automation, constants, and logging.

Completed stream segments now carry the latest streaming transcript plus the locked transcript prefix boundary produced by the live ASR path. The app still preserves the raw WAV for debugging, and it is designed to transcribe audio once as much as possible: after segment completion, it re-transcribes only the unresolved audio tail, allowing a small overlap before the locked boundary to improve stitching.

## Goals

- Capture mock-interview speech with minimal manual control.
- Keep transcription local through `mlx-whisper` by default.
- Transcribe audio once as much as possible, allowing only small overlap re-transcription around stabilized transcript boundaries when needed for accuracy.
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
.venv/bin/python main.py
```

This is the default path.

1. `main.py` configures logging, parses CLI flags, runs `preflight.check_runtime_dependencies()`, and calls `app.run()`.
2. `app.run()` enters `stream_loop()` unless enrollment mode is requested.
3. `stream_loop()` creates a temporary directory for WAV segments.
4. `start_stream_recorder()` starts a background thread running `audio.stream_utterance_segments()`.
5. If photo capture is enabled, `photo.start_photo_timer()` starts a second background thread running `capture_photos_on_interval()`.
6. The main thread loads the configured endpoint and final transcribers, `SpeakerIdentifier`, and `PhotoUploadTracker`.
7. During recording, `StreamSegmenter` owns segment boundaries. It uses RMS speech activity, a single streaming transcription worker, transcript-agreement stabilization, and semantic endpoint checks on the stabilized transcript buffer as it evolves.
8. For every queued completed segment, `process_stream_segment()`:
   - Reads the latest transcript plus locked transcript prefix metadata already attached to the completed segment.
   - Runs one final transcription pass over only the unresolved tail audio, with a small overlap before the locked boundary when available.
   - Deletes the temporary raw WAV segment.
   - Prints the finalized transcript.
   - Logs whether the segment ended by semantic completion, hard silence, or shutdown.
   - Builds a ChatGPT prompt.
   - Attaches a photo only when the selected photo exists and changed since the last successful upload.
   - Submits the prompt through Playwright browser automation.
9. `Ctrl+C` stops the recorder and photo timer threads cleanly.

The first successful ChatGPT submission includes the full `gpt.constants.STREAM_PROMPT`, which defines SecondVoice's behavior. Later submissions include only the segment prompt so the current ChatGPT conversation can maintain context.

### 2. Transcript-Only Mode

Command:

```sh
.venv/bin/python main.py --no-ask
```

This follows the same audio capture, transcription, and speaker-hint path as stream mode, but skips `gpt.submit_to_chatgpt()`. It is useful for testing microphone capture, silence thresholds, Whisper transcription, and voice matching without touching ChatGPT.

### 3. Voice Enrollment Path

Command:

```sh
.venv/bin/python main.py --enroll
```

This path records an interviewee voice profile.

1. `app.run()` calls `speech.enroll_interviewee_voice()`.
2. The app prompts the user to read each sentence in `SPEAKER_ENROLLMENT_PROMPTS`.
3. `audio.capture_enrollment_utterance()` records each sentence into a temporary WAV file.
4. `SpeakerIdentifier.enroll_from_clips()` encodes each clip with SpeechBrain, averages the embeddings, normalizes the result, and persists it.
5. The profile embedding is saved to `~/.secondvoice/interviewee-voice-embedding.pt`.
6. Metadata is saved to `~/.secondvoice/interviewee-voice-profile.json`.

Streaming mode can then include a confidence value showing how similar each segment sounds to the enrolled interviewee.

### 4. Photo Context Paths

The `--photo-mode` flag controls whether the main app captures and uploads photos.

```sh
.venv/bin/python main.py --photo-mode none
.venv/bin/python main.py --photo-mode test
.venv/bin/python main.py --photo-mode live
```

- `none`: disables all photo capture and upload. This is the default and is useful with `--no-ask` for transcription-only testing.
- `test`: captures `/Users/flora/interview/test.jpg` after 60 seconds and then every 60 seconds.
- `live`: captures `/Users/flora/interview/live.jpg` after 10 minutes and then every 10 minutes.

`photo.PhotoUploadTracker` stores the last uploaded photo signature as `(mtime_ns, size)`. If the selected photo is missing, empty, or unchanged, the app submits text only.

### 5. Manual ChatGPT Submission Test

Command:

```sh
.venv/bin/python scripts/test_chatgpt_submit.py
```

This script opens the automation Chrome profile on macOS only when the CDP endpoint is not already running, builds a stream prompt from a fixed two-sum-style transcript, and submits it to ChatGPT. It bypasses microphone recording, local transcription, and interactive transcript entry, so it is useful for manual browser automation, fixed photo upload, and scroll-behavior checks. It does not assert scroll position automatically, and it does not accept a photo mode; it always attaches `/Users/flora/interview/static.jpg`.

## Main Data Flow

```text
Microphone
  -> audio.stream_utterance_segments()
  -> temporary WAV segment
  -> streaming transcript snapshots
  -> semantic completion classification on stabilized transcript
  -> tail-only final transcription with small overlap before locked boundary
  -> app.process_stream_segment()
  -> gpt.build_stream_prompt()
  -> optional photo selected by vision.next_photo_upload()
  -> gpt.submit_to_chatgpt()
  -> ChatGPT conversation
```

Enrollment has a separate data flow:

```text
Microphone
  -> audio.capture_enrollment_utterance()
  -> temporary enrollment WAV clips
  -> speech.SpeakerIdentifier.enroll_from_clips()
  -> ~/.secondvoice voice profile files
```

## Stream Boundary Pipeline

`StreamSegmenter` is the only owner of active recording state. It decides when a segment starts, when a segment ends, and what completion reason is attached to the queued WAV.

### 1. Speech Activity

Recording starts when RMS audio crosses `DEFAULT_SILENCE_THRESHOLD`. Once a segment is active, `StreamSpeechDetector` uses a lower continuation threshold plus a short hangover so quiet syllables and tiny gaps do not immediately count as silence.

### 2. Streaming Transcription

While a segment is active, the recorder periodically snapshots the current audio into a transcription worker. The worker runs a single ASR path and returns cleaned draft transcripts. `StreamSegmenter` keeps the latest cleaned transcript plus an agreement counter. When the same normalized draft arrives `STREAM_TRANSCRIPT_AGREEMENT_COUNT`, currently 2, times in a row, the transcript is treated as stabilized.

### 3. Semantic Endpoint Check

When the stabilized transcript buffer changes (after `n`-agreement), the recorder queues a semantic endpoint check using the locked transcript text itself. The semantic worker asks Ollama `qwen2.5:1.5b` whether that stabilized transcript is `COMPLETE` or `INCOMPLETE`.

The recorder accepts only current results. Each job/result carries `segment_index` and `pause_index`; if the segment already ended or a newer pause began, the old result is stale and ignored. A `COMPLETE` result is accepted only when it still matches the current locked transcript key; if newer stabilized text arrived, that old `COMPLETE` result is ignored.

### 4. Final Tail Transcription

When a segment ends, the main thread does not re-transcribe the whole segment by default. Instead, the completed-segment payload carries the locked transcript prefix plus the chunk index where that prefix ended. The finalization step re-transcribes only the remaining tail audio, with a small overlap before the locked boundary, then stitches the tail transcript back onto the locked prefix. This keeps total transcription work close to one pass over the audio while still allowing a little redundant context at the seam.

### 5. Hard Silence

If no semantic endpoint is accepted first, the recorder reaches `STREAM_HARD_SILENCE_SECONDS`, currently 2 seconds, and cuts the segment immediately using the latest cleaned transcript it has. The main thread then finalizes only the unresolved tail audio rather than paying for a second full-pass transcription over the whole segment.

## Threading Model

Stream mode uses the main thread for voice matching, prompt construction, and browser submission. Audio capture runs in a background thread so recording can continue while segments are processed. The streaming transcription worker and semantic endpoint worker run separately so ASR and Ollama latency cannot block microphone reads. Photo capture can run in a second background thread for `test` and `live` modes.

Communication between the recorder and main thread is via `queue.Queue[CompletedStreamSegment | Exception]`. Completed segments carry the WAV path and completion reason. Recorder failures are queued as exceptions and re-raised by `next_stream_segment()`.

Shutdown is coordinated with a shared `threading.Event`. `KeyboardInterrupt` sets the event, joins background threads, and logs completion.

## Audio Queue Design

Audio recording has three queue-based handoffs with clear ownership boundaries.

### 1) Completed Segment Queue

```text
Recorder thread
  captures microphone audio
  finalizes segment boundary
  puts CompletedStreamSegment into segment_queue
        |
        v
Main app thread
  reads segment_queue
  runs speaker matching on raw WAV
  prints/submits transcript
```

This queue carries finished segments (or recorder exceptions). A completed segment includes the raw WAV path, latest transcript, locked transcript prefix metadata, and completion trigger.

### 2) Streaming Transcription Queues

```text
Recorder thread
  snapshots in-progress audio periodically
  puts TranscriptionJob into transcription_job_queue
        |
        v
Transcription worker thread
  writes temp WAV snapshot
  runs single streaming ASR pass
  puts TranscriptionResult into transcription_result_queue
        |
        v
Recorder thread
  updates latest transcript
  tracks agreement count
```

This replaces the older dual-pass model. Segment-finalization may do one extra transcription pass, but only for the unresolved tail audio and a small overlap window, not for the full segment.

### 3) Semantic Classification Queues

```text
Recorder thread
  when stabilized transcript buffer changes
  puts SemanticEndpointJob(transcript text, transcript_key) into semantic_job_queue
        |
        v
Semantic worker thread
  classifies transcript COMPLETE vs INCOMPLETE via Ollama
  puts SemanticEndpointResult (including transcript_key) into semantic_result_queue
        |
        v
Recorder thread
  accepts only current (non-stale) results
  cuts segment only on current COMPLETE for current locked transcript key
```

Each semantic job/result includes `segment_index` and `pause_index`. Results are ignored when stale (speaker resumed, a newer pause started, or the segment already ended).

## Timeline Examples

These examples use `segment_index=5` and start with `pause_index=10`. Times are illustrative.

### Semantic Completion

```text
t+0.0  stabilized transcript key changes
t+0.0  semantic job queued for segment 5, pause 10, key K1
t+0.6  semantic result returns COMPLETE for segment 5, pause 10, key K1
t+0.6  recorder accepts result and cuts segment
```

### Stale Semantic Result

```text
t+0.0  stabilized transcript key changes
t+0.0  semantic job queued for segment 5, pause 10, key K1
t+0.3  speaker resumes; recorder increments pause_index to 11
t+0.6  old semantic result returns COMPLETE for segment 5, pause 10, key K1
t+0.6  recorder ignores stale result
```

### Key-Mismatch Semantic Result

```text
t+0.0  semantic job queued with key K1
t+0.2  newer stabilized text arrives; key becomes K2
t+0.6  semantic result returns COMPLETE for key K1
t+0.6  recorder ignores result because current key is K2
```

### Hard Silence Cut

```text
t+0.0  segment remains active with no accepted COMPLETE result
t+3.0  hard silence reached; recorder cuts segment
```

The recorder remains the single owner of mutable recording state and the only code path that finalizes segments.

## External Dependencies

- `sounddevice`: microphone input.
- `mlx-whisper`: default local transcription backend.
- `faster-whisper`: alternate local transcription backend for benchmark comparisons and experiments.
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

- ChatGPT selectors in `src/gpt/actions.py` are intentionally defensive, but ChatGPT UI changes can still break submission or upload.
- Photo paths are hard-coded for `/Users/flora/interview`; changing this should be centralized in `src/vision/constants.py`.

## File and Method Reference

### `main.py`

Entrypoint for the command-line app.

- `main()`: Configures Loguru logging and starts the application with parsed runtime options.
- `parse_args()`: Defines CLI flags and returns `RuntimeOptions`.

Current CLI flags:

- `--no-ask`: Turns off ChatGPT submission while keeping capture and transcription active.
- `--enroll`: Runs voice enrollment instead of stream mode.
- `--photo-mode`: Selects `none`, `test`, or `live` photo behavior.

### `src/preflight.py`

Startup dependency checks.

- `check_runtime_dependencies(options)`: Runs all required startup checks and aborts before microphone capture when a dependency is missing.
- `ollama_model_is_ready()`: Checks that Ollama is reachable and `qwen2.5:1.5b` is installed.
- `cdp_browser_is_ready()`: Checks that Chrome CDP is reachable for ChatGPT submission runs.

### `src/app.py`

The stream-runtime orchestration layer. It owns high-level runtime paths, recorder threading, segment processing, speaker hint fallback, and console output.

- `RuntimeOptions`: Immutable dataclass carrying CLI-selected behavior.
- `run(options)`: Dispatches to enrollment or stream mode.
- `stream_loop(options)`: Runs the continuous capture/transcribe/submit loop.
- `start_stream_recorder(output_dir, segment_queue, stop_event)`: Starts audio recording in a background thread.
- `next_stream_segment(segment_queue)`: Polls the recorder queue and raises recorder exceptions on the main thread.
- `process_stream_segment(...)`: Handles one completed WAV segment end to end, including tail-only final transcription before GPT submission.
- `build_speaker_hint(audio_path, speaker_identifier)`: Produces a `SpeakerHint`, falling back to `unknown` if voice matching fails.
- `print_speaker_hint(speaker_hint)`: Logs speaker-match information.
- `print_stream_mode_banner(options)`: Logs active runtime settings.
- `print_transcript(transcript)`: Logs the transcript or a no-speech marker.

### `src/audio/`

Audio capture, segmentation, WAV writing, and amplitude helpers. `src/audio/__init__.py` re-exports the public functions used by the app, so callers can continue to import from `audio`.

#### `src/audio/enrollment.py`

- `capture_enrollment_utterance(...)`: Records one voice-enrollment sentence using speech-start and silence-stop triggers.

#### `src/audio/segmenter.py`

- `StreamSpeechDetector`: Stream-only RMS detector with hysteresis: the normal threshold starts recording, a lower continuation threshold keeps active speech alive, and a short hangover protects word/syllable gaps.
- `StreamSegmenter`: Owns stream-recording mutable state, including open WAV file, segment index, pause index, silence counters, pre-roll, queue handoff, transcript agreement state, and segment finalization.
- Stream logs announce idle/listening transitions, streaming transcript stabilization, semantic endpoint checks, and hard-silence cuts.
- `stream_utterance_segments(...)`: Compatibility wrapper that creates a `StreamSegmenter` and runs it.

#### `src/audio/stream_types.py`

- `TranscriptionJob`: In-progress segment snapshot submitted from recorder to streaming ASR worker.
- `TranscriptionResult`: Cleaned transcript result returned from the streaming ASR worker.
- `SemanticEndpointJob`: Stabilized transcript text submitted from recorder to semantic worker.
- `SemanticEndpointResult`: Completion decision and transcript returned from semantic worker to recorder.
- `CompletedStreamSegment`: Finalized segment payload consumed by the main app thread.

#### `src/audio/stream_workers.py`

- `run_transcription_worker(...)`: Consumes transcription jobs, writes temporary WAV snapshots, runs the single streaming ASR path, and publishes draft transcripts without blocking microphone capture.
- `run_semantic_endpoint_worker(...)`: Consumes stabilized transcript jobs and classifies completeness without re-transcribing audio.

#### `src/audio/transcript_utils.py`

- `TRANSCRIPT_WORD_PATTERN`: Shared transcript-token regex used for transcript-key normalization.

#### `src/audio/wav.py`

- `write_wav_file(output_path, chunks)`: Writes raw chunks into a complete WAV file, used for streaming transcription snapshots.
- `write_wav_slice(output_path, source_path, start_frame)`: Writes one tail WAV file from a source WAV starting at a chosen frame offset.
- `open_wav_writer(output_path)`: Opens a mono int16 WAV writer using app audio constants.
- `write_chunks(wav_file, chunks)`: Writes raw audio chunks into a WAV file.

#### `src/audio/levels.py`

- `audio_blocksize()`: Converts chunk duration into sample count.
- `block_count_for_seconds(seconds)`: Converts seconds into audio chunk count.
- `create_pre_roll_buffer()`: Keeps a small buffer of audio before speech starts, so segment starts are not clipped.
- `chunk_is_speech(chunk, threshold)`: Uses RMS amplitude to classify a chunk as speech or silence.
- `rms_level(chunk)`: Computes RMS amplitude for int16 audio samples.

### `src/speech/`

Speech understanding, transcription, endpointing, and speaker identification. `src/speech/__init__.py` re-exports the public types and functions used by the app.

#### `src/speech/enrollment.py`

Interviewee voice enrollment flow.

- `enroll_interviewee_voice()`: Records prompted enrollment clips and persists a voice profile.
- `print_enrollment_prompt(index, total, prompt)`: Logs one enrollment sentence.

#### `src/speech/endpoint_detector.py`

Local semantic endpointing through Ollama.

- `OllamaSemanticEndpointDetector`: Holds the configured Ollama model name and classifies transcript completeness through Ollama.
- `OllamaSemanticEndpointDetector.classify_transcript(transcript)`: Trims deterministic repeated ASR tails, classifies the cleaned transcript through Ollama, logs the result, and returns a `SemanticEndpointResult`.
- `classify_endpoint_transcript(transcript, model, url, timeout_seconds)`: Sends the shared endpoint prompt to the Ollama chat API and returns the normalized label plus latency.

#### `src/speech/transcription.py`

Local transcription backend interface and implementations.

- `Transcriber`: Protocol shared by transcription backends.
- `create_transcriber(backend, model)`: Factory that returns the configured backend implementation.
- `model_path_for_run(backend, model, use_local_cache)`: Uses repo names directly for refresh/test runs, but resolves MLX Hugging Face repo names to local cached snapshot paths for normal ChatGPT runs.
- `cached_huggingface_snapshot_path(repo_id)`: Finds the newest local Hugging Face cache snapshot for an MLX model without refreshing metadata.
- `FasterWhisperTranscriber`: Loads and runs `faster-whisper`, available for benchmark comparisons and alternate endpoint experiments.
- `MlxWhisperTranscriber`: Runs `mlx-whisper`, currently using `tiny.en` as the default live streaming ASR model on Apple Silicon.
- `transcribe(audio_path)`: Backend implementations transcribe a WAV file with the coding-interview and system-design initial prompt where supported, then return joined plain text. Calls are serialized per transcriber instance.

#### `src/speech/speaker_id.py`

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
- `SpeakerIdentifier._classifier_source()`: Resolves the SpeechBrain model from the local Hugging Face cache when available.
- `SpeakerIdentifier._load_profile()`: Loads the saved embedding from disk.
- `SpeakerIdentifier._save_metadata(clip_count)`: Writes voice-profile metadata JSON.
- `SpeakerIdentifier._torch()`: Imports `torch` with a user-facing error if dependencies are missing.

### `src/automation/`

Generic browser/CDP helpers used by browser automation. `src/automation/__init__.py` re-exports the public Chrome automation helpers.

#### `src/automation/chrome.py`

- `BrowserSession`: Dataclass that owns browser context resources.
- `BrowserSession.close()`: Closes resources only when this process owns them.
- `connect_to_cdp_browser(playwright, cdp_url)`: Connects to an existing Chrome over CDP.
- `activate_chrome()`: Brings the CDP automation Chrome process to the foreground on macOS, with a generic Chrome activation fallback.
- `automation_chrome_pid()`: Finds the main Chrome process launched with the automation profile and CDP port.
- `activate_process(pid)`: Activates one macOS process by PID through System Events.

#### `src/automation/constants.py`

- `DEFAULT_CDP_URL`: Chrome DevTools Protocol endpoint used by Playwright.
- `CDP_BROWSER_PROFILE_MARKER`: Command-line marker used to find the dedicated automation Chrome process.

### `src/gpt/`

ChatGPT-specific browser automation and prompt construction. `src/gpt/__init__.py` re-exports `submit_to_chatgpt()` and `build_stream_prompt()`.

#### `src/gpt/actions.py`

- `submit_to_chatgpt(prompt, photo_path, cdp_url)`: Opens ChatGPT, attaches an optional photo, fills the prompt, submits it, scrolls the conversation into view, waits briefly for the response to finish, and brings Chrome forward on macOS.
- `open_chatgpt_page(context)`: Reuses the dedicated marked SecondVoice ChatGPT tab or opens and marks a new one.
- `is_secondvoice_chatgpt_page(page)`: Checks whether a ChatGPT tab has the SecondVoice tab marker.
- `mark_secondvoice_chatgpt_page(page)`: Marks one ChatGPT tab as the dedicated SecondVoice automation tab and adds a visible title prefix plus in-page badge.
- `stabilize_chatgpt_theme(page)`: Pins the automation page to the dark color scheme before submission.
- `stop_auto_scroll_to_bottom(page)`: Clears any old SecondVoice auto-scroll timer or observer left in the ChatGPT tab.
- `force_scroll_to_bottom(page)`: Immediately forces likely ChatGPT scroll containers to the bottom after submitting a prompt.
- `wait_for_chatgpt_response(page)`: Waits until ChatGPT appears done responding, using the stop button when visible and the enabled send button as a fallback signal.
- `find_stop_button(page)`: Tries known ChatGPT stop-generating button selectors.
- `scroll_down_short_times(page)`: Optionally performs a small number of post-response scroll nudges. The count is controlled by `CHATGPT_SHORT_SCROLL_COUNT`; `0` disables these nudges.
- `fill_prompt(prompt_box, prompt)`: Writes the prompt into the composer.
- `submit_prompt(page, prompt_box, wait_for_upload)`: Sends the prompt by button click or Enter fallback.
- `find_send_button(page)`: Tries known ChatGPT send-button selectors.
- `attach_file(page, file_path)`: Attaches a local file through file inputs or visible upload controls.
- `wait_for_attachment_upload(page, timeout)`: Waits briefly for upload completion.
- `find_prompt_box(page, timeout)`: Tries known ChatGPT composer selectors.

#### `src/gpt/prompts.py`

- `build_stream_prompt(...)`: Builds the prompt sent to ChatGPT.
- `speaker_hint_value(speaker_hint)`: Formats normalized voice confidence for the prompt.
- `speaker_hint_similarity(speaker_hint)`: Formats raw cosine similarity for the prompt.
- `speaker_hint_role(speaker_hint)`: Formats role hint for the prompt.

#### `src/gpt/constants.py`

- `STREAM_PROMPT`: System-style instruction sent on the first ChatGPT submission.
- `PHOTO_CONTEXT_PROMPT`: Instruction inserted when a photo is attached.
- ChatGPT URL, dark theme setting, response wait timeout, scroll tuning, and dedicated SecondVoice tab marker constants.

### `src/vision/`

Photo capture and upload-selection runtime. `src/vision/__init__.py` re-exports the public camera and photo helpers used by the app.

#### `src/vision/photo.py`

- `PhotoUploadTracker`: Mutable dataclass storing the last uploaded photo signature.
- `start_photo_timer(stop_event, photo_mode)`: Starts periodic photo capture in a background thread.
- `photo_capture_settings(photo_mode)`: Maps `test` and `live` modes to path, initial delay, and interval.
- `capture_photos_on_interval(stop_event, photo_path, initial_seconds, interval_seconds)`: Captures photos until stopped.
- `next_photo_upload(photo_mode, photo_tracker)`: Selects a photo only if it is available and changed.
- `current_photo_signature(photo_path)`: Returns `(mtime_ns, size)` for a valid photo file.
- `interview_photo_path(photo_mode)`: Maps `test` and `live` modes to their fixed photo paths; `none` returns no path.

#### `src/vision/camera.py`

macOS still-photo capture.

- `CameraCaptureError`: Raised when photo capture fails.
- `take_photo(output_path, camera_name)`: Uses `imagesnap` to capture one photo to a target path.

#### `src/vision/constants.py`

- Photo paths and capture intervals.
- Default camera name.

### `src/audio/constants.py`

Audio capture and segmentation configuration.

- Audio shape: sample rate, channels, sample width, chunk size, and pre-roll.
- Segmentation: hard silence duration, RMS start threshold, active-speech continuation ratio, hangover duration, streaming transcription interval, and transcript agreement count.

### `src/speech/constants.py`

Speech processing configuration.

- Transcription: configurable endpoint and final Whisper backends/models.
- ASR prompt terms: separate coding-interview and system-design vocabulary for terms like BFS, DFS, topological sort, idempotency, relational databases, consistency, queues, reliability, and common architecture phrases.
- Endpointing: Ollama URL, model name, keep-alive, request timeout, labels, and semantic endpoint prompt.
- Speaker ID: profile paths, model source, match threshold, and enrollment prompts.

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
- `classify(model, transcript)`: Calls the shared `speech.classify_endpoint_transcript()` helper so benchmarks and runtime use the same prompt and Ollama request shape.

### `scripts/benchmark_transcriber.py`

Local transcription backend benchmark harness.

- `main()`: Creates one configured transcriber and runs it against one or more WAV files.
- `parse_args()`: Accepts audio paths plus optional backend/model overrides so `faster-whisper` and `mlx-whisper` models can be compared without changing the live app.
