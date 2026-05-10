# SecondVoice Design Doc

SecondVoice is a local mock-interview assistant. It listens to microphone audio, splits speech into interview-sized segments, transcribes each segment locally, optionally attaches a current interview-board photo, and submits a round-specific prompt to ChatGPT through an existing Chrome DevTools Protocol session.

The application is intentionally small and file-oriented. `main.py` handles command-line setup and startup checks, `src/app.py` coordinates the stream runtime, and focused packages handle audio capture, transcription, semantic endpointing, prompt construction, photo capture, browser automation, constants, and logging.

Completed stream segments carry the latest streaming transcript plus a stabilized transcript prefix boundary. After a segment ends, the app re-transcribes only the unresolved audio tail, with a small overlap before the locked boundary, and stitches that final tail back onto the locked prefix.

## Goals

- Capture mock-interview speech with minimal manual control.
- Keep transcription local through `mlx-whisper` by default.
- Avoid full duplicate transcription passes; finalize only unresolved audio tails when possible.
- Use local Ollama endpoint classification to cut segments when a stabilized transcript sounds complete.
- Preserve interview context in a single ChatGPT conversation by sending the full round prompt on the first successful submission and shorter segment prompts after that.
- Support coding, system design, behavioral, and offer-negotiation round modes with editable context files under `context/`.
- Add visual problem-board context through test or live camera photos when enabled.
- Reuse a logged-in ChatGPT browser session through Chrome DevTools Protocol.

## Non-Goals

- The app does not implement its own general-purpose LLM backend. It uses a small local Ollama model only for endpoint completion classification.
- The app does not store transcript history or ChatGPT response history in a structured database.
- The app does not perform speaker diarization, voice enrollment, or voice matching.
- The app does not own the ChatGPT UI. Browser automation depends on current ChatGPT selectors and may need maintenance if the UI changes.

## Runtime Paths

### 1. Stream Mode

Command:

```sh
.venv/bin/python main.py
```

This is the default path.

1. `main.py` configures logging, parses CLI flags, runs `preflight.check_runtime_dependencies()`, and calls `app.run()`.
2. `app.run()` enters `stream_loop()`.
3. `stream_loop()` ensures round context files exist, loads the selected round context, creates a temporary directory for WAV segments, and initializes `BehaviorState` and `DesignState`.
4. The main thread creates one endpoint transcriber, one final transcriber, an `OllamaSemanticEndpointDetector`, and a `PhotoUploadTracker`.
5. `start_stream_recorder()` starts a background thread running `audio.stream_utterance_segments()`.
6. If photo capture is enabled, `vision.start_photo_timer()` starts a second background thread that periodically writes the configured photo path.
7. During recording, `StreamSegmenter` owns segment boundaries. It uses RMS speech activity, pre-roll, one streaming transcription worker, transcript-agreement stabilization, and semantic endpoint checks on locked transcript text.
8. For every queued completed segment, `process_stream_segment()`:
   - Reads the latest transcript and locked-prefix metadata attached to the completed segment.
   - Runs a final transcription pass over only the unresolved tail audio when a locked boundary exists.
   - Deletes the temporary raw WAV segment.
   - Prints the finalized transcript.
   - Skips ChatGPT when the transcript is empty or `--no-ask` is set.
   - Builds a round-specific ChatGPT prompt.
   - Attaches a photo only when the selected photo exists and changed since the last successful upload.
   - Submits the prompt through Playwright browser automation.
   - In behavioral mode, parses `actual_story_id=...` from the latest ChatGPT response and excludes that story from later suggestions in the same run.
9. `Ctrl+C` sets the shared stop event and joins recorder/photo threads cleanly.

The first successful ChatGPT submission includes the selected round prompt and `Round Context:` block. Later submissions include only the segment prompt, plus the behavioral used-story block when applicable, so the active ChatGPT conversation can maintain context.

### 2. Transcript-Only Mode

Command:

```sh
.venv/bin/python main.py --no-ask
```

This follows the same microphone capture, streaming transcription, semantic endpointing, and tail-finalization path as stream mode, but skips `gpt.submit_to_chatgpt()`. It is useful for testing microphone capture, silence thresholds, Whisper transcription, and endpoint cuts without touching ChatGPT.

### 3. Round Modes and Context

Command:

```sh
.venv/bin/python main.py --round coding
.venv/bin/python main.py --round design
.venv/bin/python main.py --round behavior
.venv/bin/python main.py --round offer
```

Round prompts live in `src/gpt/constants.py`. Editable context files live in `context/`:

- `coding.md`
- `design.md`
- `behavior.md`
- `offer.md`

`ensure_context_templates()` creates any missing context file with a placeholder. If a matching legacy file exists under `~/.secondvoice/context/`, it is copied into the repo context directory.

Behavioral mode keeps an in-memory set of story IDs used during the current run. The prompt includes `Used Story IDs:` on every behavioral segment, and `parse_actual_story_id()` reads the machine-parsable metadata bullet returned by ChatGPT.

Design mode keeps an in-memory set of proposed deep-dive topic IDs during the current run. The prompt includes `Used Design Deep Dive Topic IDs:` on every design segment, and `parse_design_deep_dive_topic_ids()` reads the machine-parsable metadata bullet returned by ChatGPT.

Design mode also asks ChatGPT to silently infer the current system-design phase: question phase, clarify functional requirement, clarify nonfunctional requirement, data model, API design, architecture, or deep dive. Suggested deep-dive topics should be specific to that inferred phase and should not jump ahead unless the transcript has already moved there. The repo-local `context/design.md` is organized under the same phase headings.

### 4. Photo Context Paths

The `--photo-mode` flag controls whether the main app captures and uploads photos.

```sh
.venv/bin/python main.py --photo-mode none
.venv/bin/python main.py --photo-mode test
.venv/bin/python main.py --photo-mode live
```

- `none`: disables photo capture and upload. This is the default.
- `test`: captures `/Users/flora/interview/test.jpg` after 60 seconds and then every 60 seconds.
- `live`: captures `/Users/flora/interview/live.jpg` after 10 minutes and then every 10 minutes.

`PhotoUploadTracker` stores the last successfully uploaded photo signature as `(mtime_ns, size)`. If the selected photo is missing, empty, or unchanged, the app submits text only.

### 5. Manual ChatGPT Submission Test

Command:

```sh
.venv/bin/python scripts/test_chatgpt_submit.py
```

This script opens the automation Chrome profile on macOS only when the CDP endpoint is not already running, builds a coding stream prompt from a fixed two-sum-style transcript, and submits it to ChatGPT with `/Users/flora/interview/static.jpg`. It bypasses microphone recording and local transcription, so it is useful for manual browser automation, fixed photo upload, and scroll-behavior checks.

## Main Data Flow

```text
Microphone
  -> audio.stream_utterance_segments()
  -> temporary WAV segment
  -> streaming transcript snapshots
  -> transcript agreement lock
  -> semantic completion classification on locked transcript
  -> tail-only final transcription with small overlap before locked boundary
  -> app.process_stream_segment()
  -> gpt.build_stream_prompt()
  -> optional photo selected by vision.next_photo_upload()
  -> gpt.submit_to_chatgpt()
  -> ChatGPT conversation
```

## Stream Boundary Pipeline

`StreamSegmenter` is the only owner of active recording state. It decides when a segment starts, when a segment ends, and what completion reason is attached to the queued WAV.

### 1. Speech Activity

Recording starts when RMS audio crosses `DEFAULT_SILENCE_THRESHOLD`, currently `500`. Once a segment is active, `StreamSpeechDetector` uses a lower continuation threshold plus a short hangover so quiet syllables and tiny gaps do not immediately count as silence. A short pre-roll is included so speech starts are less likely to be clipped.

### 2. Streaming Transcription

While a segment is active, the recorder periodically snapshots audio into a transcription worker. Jobs contain only audio from `locked_chunk_index` through the current segment end, so confirmed prefixes are not repeatedly transcribed. The worker writes a temporary WAV snapshot, runs the configured local ASR backend, deletes the snapshot, and returns plain text.

`StreamSegmenter` combines each fresh suffix with the locked prefix, normalizes the words, and tracks repeated agreement. When the same normalized transcript arrives `STREAM_TRANSCRIPT_AGREEMENT_COUNT` times in a row, currently `2`, that transcript is locked through the snapshot end chunk.

### 3. Semantic Endpoint Check

When locked transcript text changes, the recorder queues one semantic endpoint job. The semantic worker asks local Ollama `qwen2.5:1.5b` whether the locked transcript is `COMPLETE` or `INCOMPLETE`.

The recorder accepts only current results. Each transcription and semantic result carries `segment_index` and `pause_index`; if the segment already ended or the speaker resumed after that pause, the old result is stale and ignored. A `COMPLETE` result is also ignored when its `transcript_key` no longer matches the current locked transcript key.

### 4. Final Tail Transcription

When a segment ends, the main thread finalizes the transcript. If there is no locked prefix, it transcribes the whole saved segment. If there is a locked prefix, it creates a tail WAV starting `FINAL_TRANSCRIPTION_OVERLAP_SECONDS`, currently `0.4`, before the locked boundary. The final tail transcript is stitched onto the locked prefix with up to 12 words of case-insensitive overlap removal.

If final transcription fails or returns empty text, the app logs a warning and falls back to the live streaming transcript.

### 5. Hard Silence

If no semantic endpoint is accepted first, the recorder reaches `STREAM_HARD_SILENCE_SECONDS`, currently `2`, and cuts the segment with completion reason `2.0s hard silence`.

## Threading Model

Stream mode uses the main thread for prompt construction, optional photo selection, ChatGPT submission, and final tail transcription. Audio capture runs in a background thread so recording can continue while segments are processed. The streaming transcription worker and semantic endpoint worker run separately so ASR and Ollama latency do not block microphone reads. Photo capture can run in another background thread for `test` and `live` modes.

Communication between the recorder and main thread is via `queue.Queue[CompletedStreamSegment | Exception]`. Recorder failures are queued as exceptions and re-raised by `next_stream_segment()`.

Shutdown is coordinated with a shared `threading.Event`. `KeyboardInterrupt` sets the event, joins background threads, and logs completion.

## Queue Handoffs

### Completed Segment Queue

```text
Recorder thread
  captures microphone audio
  finalizes segment boundary
  puts CompletedStreamSegment into segment_queue
        |
        v
Main app thread
  reads segment_queue
  finalizes transcript tail
  prints/submits transcript
```

This queue carries finished segments or recorder exceptions. A completed segment includes the raw WAV path, latest transcript, locked transcript prefix metadata, and completion trigger.

### Streaming Transcription Queues

```text
Recorder thread
  snapshots unlocked in-progress audio periodically
  puts TranscriptionJob into transcription_job_queue
        |
        v
Transcription worker thread
  writes temp WAV snapshot
  runs local ASR
  puts TranscriptionResult into transcription_result_queue
        |
        v
Recorder thread
  combines with locked prefix
  tracks agreement count
  locks confirmed transcript prefixes
```

### Semantic Classification Queues

```text
Recorder thread
  when locked transcript text changes
  puts SemanticEndpointJob(transcript text, transcript_key) into semantic_job_queue
        |
        v
Semantic worker thread
  classifies transcript COMPLETE vs INCOMPLETE via Ollama
  puts SemanticEndpointResult into semantic_result_queue
        |
        v
Recorder thread
  accepts only current non-stale COMPLETE results
```

## External Dependencies

- `sounddevice`: microphone input.
- `mlx-whisper`: default local transcription backend.
- `faster-whisper`: alternate transcription backend for benchmarks and experiments.
- `ollama`: local semantic endpoint classification with `qwen2.5:1.5b`.
- `playwright`: ChatGPT browser automation.
- `loguru`: console and file logging.
- `imagesnap`: external macOS command for named-camera still capture.
- Google Chrome with remote debugging enabled for the default browser submission path. The live stream path expects this CDP browser to already be running; the smoke test opens it on macOS if it is missing.

## Persistent and Temporary State

Persistent state:

- `python.log`: debug log in the current working directory.
- `~/.secondvoice/cdp-browser-profile`: recommended Chrome profile for reusable ChatGPT login.
- `/Users/flora/interview/test.jpg`, `live.jpg`: main app photo-context inputs.
- `/Users/flora/interview/static.jpg`: fixed photo fixture for `scripts/test_chatgpt_submit.py`.
- `context/*.md`: editable round context loaded into the first prompt.

Temporary state:

- Stream WAV segments are created in a `secondvoice-eval-*` temporary directory and deleted after segment processing.
- Streaming transcription snapshot WAVs and final-tail WAVs are temporary and deleted after use.

## Error Handling

- Startup preflight aborts before microphone capture when Ollama is unavailable, the endpoint model is missing, or Chrome CDP is unavailable for ChatGPT submission runs.
- Recorder-thread exceptions are passed through the completed segment queue and raised on the main thread.
- Failed or empty final tail transcription falls back to the latest streaming transcript.
- Empty transcripts are printed and not submitted.
- Missing, empty, unchanged, or failed-upload photos result in text-only or skipped ChatGPT submissions as logged.
- Camera capture failures are logged as warnings during periodic capture.
- Browser connection or prompt-box failures stop the run with setup guidance.

## Maintenance Notes

- ChatGPT selectors in `src/gpt/actions.py` are intentionally defensive, but ChatGPT UI changes can still break submission or upload.
- Photo paths are hard-coded for `/Users/flora/interview`; changing this should be centralized in `src/vision/constants.py`.
- The live app currently has no CLI flag for changing transcription backend/model, silence threshold, endpoint model, or CDP URL; those are constants in `src/audio/constants.py`, `src/speech/constants.py`, and `src/automation/constants.py`.

## File and Method Reference

### `main.py`

- `main()`: Configures logging, parses options, runs preflight checks, and starts the app.
- `parse_args()`: Defines `--round`, `--no-ask`, and `--photo-mode`.

### `src/preflight.py`

- `check_runtime_dependencies(options)`: Runs startup checks and aborts before microphone capture when a dependency is missing.
- `ollama_model_is_ready()`: Checks that Ollama is reachable and `qwen2.5:1.5b` is installed.
- `cdp_browser_is_ready()`: Checks that Chrome CDP is reachable for ChatGPT submission runs.

### `src/app.py`

- `RuntimeOptions`: Immutable dataclass carrying CLI-selected behavior.
- `run(options)`: Starts stream mode.
- `stream_loop(options)`: Runs the continuous capture/transcribe/submit loop.
- `start_stream_recorder(...)`: Starts audio recording in a background thread.
- `next_stream_segment(segment_queue)`: Polls the recorder queue and raises recorder exceptions on the main thread.
- `process_stream_segment(...)`: Handles one completed segment end to end.
- `finalize_segment_transcript(...)`: Runs final tail transcription and merges it with the locked prefix.
- `build_final_transcription_audio(...)`: Chooses whole-segment or tail-only audio for final transcription.
- `combine_locked_and_tail_transcript(...)`: Removes overlap and joins locked prefix plus final tail.
- `print_stream_mode_banner(options)`: Logs active runtime settings.
- `print_transcript(transcript)`: Logs the transcript or a no-speech marker.

### `src/audio/`

- `segmenter.py`: Owns stream recording, speech activity detection, transcript agreement, semantic endpoint acceptance, and segment finalization.
- `stream_types.py`: Defines queue payload dataclasses.
- `stream_workers.py`: Runs transcription and semantic endpoint workers.
- `levels.py`: Converts chunk durations, keeps pre-roll, and computes RMS levels.
- `wav.py`: Writes complete WAV files and tail slices.
- `transcript_utils.py`: Provides transcript word normalization regex.
- `constants.py`: Defines audio shape, thresholds, timing, and transcript agreement settings.

### `src/speech/`

- `transcription.py`: Defines the transcriber protocol, MLX Whisper and faster-whisper backends, local cache resolution, and WAV loading.
- `endpoint_detector.py`: Calls the Ollama chat API and normalizes endpoint labels.
- `constants.py`: Defines transcription backend/model defaults, ASR vocabulary prompt terms, Ollama URL/model/timeout, endpoint labels, and endpoint prompt.

### `src/gpt/`

- `context.py`: Creates and loads round context files.
- `prompts.py`: Builds round-specific prompts, serializes behavioral used-story IDs and design deep-dive topic IDs, and parses returned metadata.
- `constants.py`: Stores round prompt contracts, photo context prompt, ChatGPT URL/theme/timeout settings, and dedicated tab markers.
- `actions.py`: Connects to Chrome CDP, reuses or opens the marked ChatGPT tab, attaches optional photos, submits prompts, waits for responses, scrolls, and returns the latest assistant message text.

### `src/vision/`

- `photo.py`: Handles photo timer settings, periodic capture loop, upload change detection, and fixed photo paths per mode.
- `camera.py`: Uses `imagesnap` to capture one macOS still photo.
- `constants.py`: Stores photo paths, capture intervals, and default camera name.

### `src/automation/`

- `chrome.py`: Connects to Chrome over CDP and activates the automation browser on macOS.
- `constants.py`: Stores the default CDP URL and automation profile marker.

### `src/logging_config.py`

- `configure_logging(log_path)`: Writes INFO logs to stderr and DEBUG logs to `python.log`.

### `scripts/`

- `test_chatgpt_submit.py`: Manual ChatGPT browser submission harness with static photo fixture.
- `test_endpoint_detector.py`: Ollama semantic-endpoint benchmark harness.
- `test_streaming_pipeline.py`: Local streaming pipeline test harness.
- `test_round_prompts.py`: Prompt construction checks for round modes.
- `benchmark_transcriber.py`: Local transcription backend benchmark harness.
