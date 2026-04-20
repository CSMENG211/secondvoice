## SecondVoice

SecondVoice is a voice-triggered mock interview evaluator.

It captures audio from your microphone, transcribes it locally with
mlx-whisper, and sends each transcript segment to ChatGPT for role
classification, context tracking, and answer feedback.

By default, SecondVoice preprocesses each captured WAV with an in-app
spectral-gate noise reducer before transcription. This helps reduce background
sounds such as keyboard taps without changing the raw audio used for voice
matching.

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
```

Print transcripts and interviewee voice confidence without submitting to
ChatGPT:

```sh
.venv/bin/python main.py --no-ask
```

Run without photo capture or upload:

```sh
.venv/bin/python main.py --no-ask --photo-mode none
```

Compare transcription with raw, unenhanced audio:

```sh
.venv/bin/python main.py --no-ask --audio-enhancement off
```

Record an interviewee voice profile:

```sh
.venv/bin/python main.py --enroll
```

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
  Apple makes Mic Modes app-dependent, so SecondVoice does its own in-app audio
  enhancement instead of relying on a system mode being available to the CLI.
