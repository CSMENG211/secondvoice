## SecondVoice

SecondVoice is a voice-triggered mock interview evaluator.

It captures audio from your microphone, transcribes it locally with
faster-whisper, and sends each transcript segment to ChatGPT for role
classification, context tracking, and answer feedback.

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

By default, SecondVoice streams continuously and submits transcript segments to
ChatGPT.

Logs are written to `python.log` in the current working directory.

Photo capture and upload are disabled by default. Use `--photo-mode test` or
`--photo-mode live` when you want SecondVoice to capture and attach interview
photos.

### Stream Mode

Continuously record mock-interview segments, transcribe each segment, and submit
it to ChatGPT:

```sh
python main.py
```

Recording starts automatically when speech begins. After 10 seconds of silence,
SecondVoice transcribes that segment, sends it to ChatGPT, and continues
listening. ChatGPT classifies the segment as interviewer or interviewee.
Interviewer segments are added to the problem context; interviewee segments get
an ideal response and evaluation against that ideal.

To only print transcripts and interviewee voice confidence without sending them
to ChatGPT:

```sh
python main.py --no-ask
```

For transcription-only isolation, keep the default photo mode or pass it
explicitly:

```sh
python main.py --no-ask --photo-mode none
```

In `none` mode, SecondVoice does not capture photos and does not upload photos.

### Photo Modes

```sh
python main.py --photo-mode none
python main.py --photo-mode test
python main.py --photo-mode live
```

- `none`: disables all photo capture and upload. This is the default.
- `test`: captures `/Users/flora/interview/test.jpg` after 60 seconds and then every 60 seconds.
- `live`: captures `/Users/flora/interview/live.jpg` after 10 minutes and then every 10 minutes.

The browser smoke test always submits a fixed two-sum-style prompt and attaches
`/Users/flora/interview/static.jpg`. On macOS it opens the automation Chrome
profile when the CDP browser is not already running, then runs the ChatGPT
browser automation:

```sh
python scripts/test_chatgpt_submit.py
```

### Voice Enrollment

Record your interviewee voice profile:

```sh
python main.py --enroll
```

SecondVoice will show 10 interview-style sentences. Read each sentence out
loud using the same microphone and room you plan to use for mock interviews. The
enrollment is saved locally in `~/.secondvoice` and reused until the next time
you run `--enroll`, which replaces the previous profile.

During streaming, each segment includes an interviewee voice match confidence
from `0.0` to `1.0` in the ChatGPT prompt. Higher means the audio sounds more
like the enrolled interviewee; lower means it is more likely the interviewer or
unknown. ChatGPT treats this as a hint alongside the transcript text.

### Ask ChatGPT

Install Playwright's browser once:

```sh
python -m playwright install chromium
```

Then stream, transcribe segments, open ChatGPT, paste each prompt, and submit it:

```sh
python main.py
```

The first time, ChatGPT may ask you to log in. The browser profile is stored at
`~/.secondvoice/cdp-browser-profile`, so future runs can reuse that login.

The real stream path connects to an already-running Chrome instance that you
start with remote debugging:

```sh
open -na 'Google Chrome' --args \
  --remote-debugging-port=9222 \
  --user-data-dir=$HOME/.secondvoice/cdp-browser-profile
```

Then run:

```sh
python main.py
```

SecondVoice uses one dedicated ChatGPT tab in that profile. The tab is marked
internally, pinned to dark mode, prefixed with `SecondVoice -` in the tab title,
and labeled with a small `SecondVoice` badge in the page.

On macOS, your terminal may ask for microphone permission the first time this
runs.
