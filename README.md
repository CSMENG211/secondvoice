## SecondVoice

SecondVoice is a voice-triggered mock interview evaluator.

It captures audio from your microphone, transcribes it locally with
mlx-whisper, and sends each transcript segment to ChatGPT for context
tracking and answer feedback.

SecondVoice now uses a single streaming transcription path during live capture.
Instead of one fast draft transcription plus a second slower final
transcription, it continuously builds transcript drafts with the live ASR model,
waits for repeated draft agreement, and then runs semantic completion on the
stabilized transcript text.

The raw WAV is still preserved long enough for cleanup.

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

On startup, `.venv/bin/python main.py` checks that Ollama is reachable and has
`qwen2.5:1.5b` installed. When ChatGPT submission is enabled, it also checks
that Chrome CDP is reachable at `http://127.0.0.1:9222`. If a required service
is missing, SecondVoice logs the setup command and exits before microphone
capture starts.

### Usage

Stream and submit to ChatGPT:

```sh
.venv/bin/python main.py
.venv/bin/python main.py --round coding
.venv/bin/python main.py --round system-design
.venv/bin/python main.py --round behavior
.venv/bin/python main.py --round offer-negotiation
```

Print transcripts without submitting to ChatGPT:

```sh
.venv/bin/python main.py --no-ask
```

Run without photo capture or upload:

```sh
.venv/bin/python main.py --no-ask --photo-mode none
```

Round context files are loaded from `context/` in this repository:

- `coding.md`
- `system_design.md`
- `behavior.md`
- `offer_negotiation.md`

Missing files are auto-created with editable placeholders on startup.

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

Benchmark transcription on a saved WAV file:

```sh
.venv/bin/python scripts/benchmark_transcriber.py /path/to/sample.wav
.venv/bin/python scripts/benchmark_transcriber.py --backend faster-whisper --model small.en /path/to/sample.wav
```

Run the browser smoke test:

```sh
.venv/bin/python scripts/test_chatgpt_submit.py
```

### Notes

- Logs are written to `python.log` in the current working directory.
- The browser profile is stored at `~/.secondvoice/cdp-browser-profile`.
- On macOS, your terminal may ask for microphone permission on the first run.
- On macOS, Voice Isolation can sometimes be enabled from the menu bar Mic Mode
  control while an app is using the microphone. Treat it as an optional extra:
  Apple makes Mic Modes app-dependent, so availability can vary by app.
