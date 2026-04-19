import argparse
import json
import time
from dataclasses import dataclass
from urllib.error import URLError
from urllib.request import Request, urlopen


OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen2.5:1.5b"

SYSTEM_PROMPT = (
    "You are an endpoint detector for live coding interview speech. "
    "Return exactly one word: COMPLETE or INCOMPLETE. "
    "COMPLETE means the transcript is usable as a standalone segment. "
    "It may be short. A stated value, action, condition, complexity, "
    "edge case, or plan step can be COMPLETE. "
    "INCOMPLETE means the transcript ends mid-phrase, after a connector, "
    "or clearly expects more words immediately. "
    "Examples: Transcript: so it is -> INCOMPLETE. "
    "Transcript: so it is true -> COMPLETE. "
    "Transcript: return -> INCOMPLETE. "
    "Transcript: return false -> COMPLETE. "
    "Transcript: the time complexity is -> INCOMPLETE. "
    "Transcript: the time complexity is O of n -> COMPLETE. "
    "Transcript: I would use a hash map because -> INCOMPLETE. "
    "Transcript: I would use a hash map because lookup is constant time -> COMPLETE. "
    "Transcript: and then -> INCOMPLETE. "
    "Transcript: and then I move the left pointer -> COMPLETE."
)


@dataclass(frozen=True)
class EndpointCase:
    transcript: str
    expected: str
    note: str


CASES = [
    EndpointCase("so it is", "INCOMPLETE", "trailing setup"),
    EndpointCase("so it is true", "COMPLETE", "final boolean answer"),
    EndpointCase("the time complexity is", "INCOMPLETE", "missing complexity"),
    EndpointCase("the time complexity is O of n", "COMPLETE", "complexity stated"),
    EndpointCase("I would use a hash map because", "INCOMPLETE", "dangling because"),
    EndpointCase(
        "I would use a hash map because lookup is constant time",
        "COMPLETE",
        "reason completed",
    ),
    EndpointCase("return", "INCOMPLETE", "bare action"),
    EndpointCase("return false", "COMPLETE", "return value stated"),
    EndpointCase("yes", "COMPLETE", "short final answer"),
    EndpointCase("no", "COMPLETE", "short final answer"),
    EndpointCase("and then", "INCOMPLETE", "connector"),
    EndpointCase("and then I move the left pointer", "COMPLETE", "step completed"),
    EndpointCase("if the number is", "INCOMPLETE", "unfinished condition"),
    EndpointCase("if the number is negative, I skip it", "COMPLETE", "condition resolved"),
    EndpointCase("we can optimize this by", "INCOMPLETE", "unfinished method"),
    EndpointCase("we can optimize this by sorting the array first", "COMPLETE", "method stated"),
    EndpointCase("the edge case is when", "INCOMPLETE", "unfinished edge case"),
    EndpointCase(
        "the edge case is when the input array is empty",
        "COMPLETE",
        "edge case stated",
    ),
    EndpointCase("space complexity would be", "INCOMPLETE", "missing complexity"),
    EndpointCase("space complexity would be constant", "COMPLETE", "complexity stated"),
    EndpointCase("I think the answer is", "INCOMPLETE", "missing answer"),
    EndpointCase("I think the answer is two", "COMPLETE", "answer supplied"),
    EndpointCase("so the invariant is", "INCOMPLETE", "missing invariant"),
    EndpointCase(
        "so the invariant is that everything to the left is sorted",
        "COMPLETE",
        "invariant supplied",
    ),
    EndpointCase("first I would", "INCOMPLETE", "unfinished plan"),
    EndpointCase("first I would clarify the input constraints", "COMPLETE", "plan step stated"),
    EndpointCase("the base case is", "INCOMPLETE", "missing base case"),
    EndpointCase("the base case is when the node is null", "COMPLETE", "base case stated"),
    EndpointCase("that means", "INCOMPLETE", "unfinished implication"),
    EndpointCase("that means we found a valid pair", "COMPLETE", "implication completed"),
]


def main() -> None:
    args = parse_args()
    models = args.models or prompt_for_models()
    for model in models:
        run_model_benchmark(model)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Ollama endpoint-completion classification models."
    )
    parser.add_argument(
        "models",
        nargs="*",
        help=(
            "Ollama model names to test. If omitted, prompt interactively."
        ),
    )
    return parser.parse_args()


def prompt_for_models() -> list[str]:
    """Ask which Ollama model or models to benchmark."""
    try:
        raw_models = input(
            "Ollama model(s), comma or space separated "
            f"[{DEFAULT_MODEL}]: "
        ).strip()
    except EOFError:
        raw_models = ""
    if not raw_models:
        return [DEFAULT_MODEL]

    return [
        model
        for model in raw_models.replace(",", " ").split()
        if model
    ]


def run_model_benchmark(model: str) -> None:
    print(f"\n== {model} ==")
    failures = []
    false_complete = 0
    false_incomplete = 0
    latencies = []

    for index, case in enumerate(CASES, start=1):
        try:
            prediction, duration_ms = classify(model, case.transcript)
        except URLError as error:
            print(f"Could not reach Ollama at {OLLAMA_CHAT_URL}: {error}")
            return

        latencies.append(duration_ms)
        passed = prediction == case.expected
        if not passed:
            failures.append((case, prediction))
            if prediction == "COMPLETE":
                false_complete += 1
            elif prediction == "INCOMPLETE":
                false_incomplete += 1

        status = "PASS" if passed else "FAIL"
        print(
            f"{index:02d}. {status} {duration_ms:7.1f} ms "
            f"expected={case.expected:<10} got={prediction:<10} "
            f"{case.note}: {case.transcript!r}"
        )

    print_summary(
        total=len(CASES),
        failures=failures,
        false_complete=false_complete,
        false_incomplete=false_incomplete,
        latencies=latencies,
    )


def classify(model: str, transcript: str) -> tuple[str, float]:
    payload = {
        "model": model,
        "stream": False,
        "keep_alive": "30m",
        "options": {
            "temperature": 0,
            "num_predict": 3,
        },
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Transcript: {transcript}"},
        ],
    }
    request = Request(
        OLLAMA_CHAT_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    started = time.perf_counter()
    with urlopen(request, timeout=30) as response:
        body = json.loads(response.read().decode("utf-8"))
    duration_ms = (time.perf_counter() - started) * 1000

    content = body["message"]["content"].strip().upper()
    if "INCOMPLETE" in content:
        return "INCOMPLETE", duration_ms
    if "COMPLETE" in content:
        return "COMPLETE", duration_ms
    return f"OTHER:{content}", duration_ms


def print_summary(
    total: int,
    failures: list[tuple[EndpointCase, str]],
    false_complete: int,
    false_incomplete: int,
    latencies: list[float],
) -> None:
    passed = total - len(failures)
    sorted_latencies = sorted(latencies)
    p50 = percentile(sorted_latencies, 0.50)
    p95 = percentile(sorted_latencies, 0.95)
    average = sum(latencies) / len(latencies)
    print(
        f"Summary: {passed}/{total} pass; "
        f"false COMPLETE={false_complete}; "
        f"false INCOMPLETE={false_incomplete}; "
        f"avg={average:.1f} ms; p50={p50:.1f} ms; p95={p95:.1f} ms"
    )
    if failures:
        print("Failures:")
        for case, prediction in failures:
            print(
                f"  expected={case.expected:<10} got={prediction:<10} "
                f"{case.note}: {case.transcript!r}"
            )


def percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1, round((len(sorted_values) - 1) * fraction))
    return sorted_values[index]


if __name__ == "__main__":
    main()
