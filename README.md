## offerGPT

A small Python prototype for turning microphone input into text locally, then
using that text as the prompt for a later browser automation step.

This first step captures audio from your microphone and transcribes it locally
with faster-whisper.

### Setup

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run

```sh
python main.py
```

Press ENTER to start recording, speak, then press ENTER again to stop and
transcribe.

Choose a local model size:

```sh
python main.py --model tiny
python main.py --model small
python main.py --model medium
```

### Ask ChatGPT

Install Playwright's browser once:

```sh
python -m playwright install chromium
```

Then record, transcribe, open ChatGPT, paste the transcript, and submit it:

```sh
python main.py --ask-chatgpt
```

The first time, ChatGPT may ask you to log in. The browser profile is stored at
`~/.offergpt/browser-profile`, so future runs can reuse that login.

### Pick a Microphone

```sh
python main.py --list-mics
python main.py --device 2
```

On macOS, your terminal may ask for microphone permission the first time this
runs.
