## SecondVoice

SecondVoice is a voice-triggered mock interview assistant.

It captures microphone audio, transcribes it locally with `mlx-whisper`, cuts segments using local semantic endpoint detection, and can submit each finalized transcript segment to ChatGPT for live interview coaching.

The live pipeline uses one streaming ASR path while recording. It snapshots in-progress audio, waits for repeated transcript agreement, locks stable transcript prefixes, asks local Ollama whether the locked transcript sounds complete, and after each segment ends re-transcribes only the unresolved audio tail before submitting.

### Install

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m playwright install chromium
```

Pull the local endpoint detector model:

```sh
ollama pull qwen2.5:1.5b
```

### Setup

Before each run, keep Ollama running in a terminal:

```sh
ollama serve
```

Start Chrome in CDP mode:

```sh
open -na 'Google Chrome' --args \
  --remote-debugging-port=9222 \
  --user-data-dir=$HOME/.secondvoice/cdp-browser-profile
```

In that Chrome window, open ChatGPT and log in:

```text
https://chatgpt.com/
```

On startup, `.venv/bin/python main.py` checks that Ollama is reachable and has `qwen2.5:1.5b` installed. When ChatGPT submission is enabled, it also checks that Chrome CDP is reachable at `http://127.0.0.1:9222`. If a required service is missing, SecondVoice logs the setup command and exits before microphone capture starts.

### Usage

Stream and submit to ChatGPT:

```sh
.venv/bin/python main.py
.venv/bin/python main.py --round coding
.venv/bin/python main.py --round design
.venv/bin/python main.py --round behavior
.venv/bin/python main.py --round offer
```

Print finalized transcripts without submitting to ChatGPT:

```sh
.venv/bin/python main.py --no-ask
```

Round context files are loaded from `context/` in this repository:

- `coding.md`
- `design.md`
- `behavior.md`
- `offer.md`

Missing files are auto-created with editable placeholders on startup. Legacy files under `~/.secondvoice/context/` are copied into `context/` when a repo-local file is missing.

Behavioral mode tracks story IDs used during the current run. ChatGPT is asked to include one metadata bullet with `actual_story_id=...`; SecondVoice parses that value and excludes that story from later behavioral suggestions in the same process.

Design mode tracks deep-dive topic IDs proposed during the current run. ChatGPT is asked to include `design_deep_dive_topic_ids=...`; SecondVoice parses those IDs and excludes those deep-dive topics from later design suggestions.

Run with photo mode:

```sh
.venv/bin/python main.py --photo-mode none
.venv/bin/python main.py --photo-mode test
.venv/bin/python main.py --photo-mode live
```

Photo modes:

- `none`: disables photo capture and upload. This is the default.
- `test`: captures `/Users/flora/interview/test.jpg` after 60 seconds and then every 60 seconds.
- `live`: captures `/Users/flora/interview/live.jpg` after 10 minutes and then every 10 minutes.

Photos are uploaded only when the selected file exists, is non-empty, and has changed since the last successful upload.

Benchmark transcription on a saved WAV file:

```sh
.venv/bin/python scripts/benchmark_transcriber.py /path/to/sample.wav
.venv/bin/python scripts/benchmark_transcriber.py --backend faster-whisper --model small.en /path/to/sample.wav
```

Run the browser smoke test:

```sh
.venv/bin/python scripts/test_chatgpt_submit.py
```

Run prompt tests:

```sh
.venv/bin/python scripts/test_round_prompts.py
```

### Notes

- Logs are written to `python.log` in the current working directory.
- The browser profile is stored at `~/.secondvoice/cdp-browser-profile`.
- Temporary WAV files are created under a `secondvoice-eval-*` temp directory and deleted after processing.
- On macOS, your terminal may ask for microphone permission on the first run.
- On macOS, Voice Isolation can sometimes be enabled from the menu bar Mic Mode control while an app is using the microphone. Treat it as an optional extra: Apple makes Mic Modes app-dependent, so availability can vary by app.
