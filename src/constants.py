from pathlib import Path


AUDIO_SAMPLE_RATE = 16_000
AUDIO_CHANNELS = 1
AUDIO_SAMPLE_WIDTH_BYTES = 2
AUDIO_CHUNK_SECONDS = 0.1
AUDIO_PRE_ROLL_SECONDS = 0.3

STREAM_SEMANTIC_SILENCE_SECONDS = 3.0
STREAM_HARD_SILENCE_SECONDS = 10.0
DEFAULT_SILENCE_THRESHOLD = 500

TRANSCRIPTION_BACKEND_FASTER_WHISPER = "faster-whisper"
TRANSCRIPTION_BACKEND_MLX_WHISPER = "mlx-whisper"
DEFAULT_ENDPOINT_TRANSCRIPTION_BACKEND = TRANSCRIPTION_BACKEND_FASTER_WHISPER
DEFAULT_ENDPOINT_TRANSCRIPTION_MODEL = "small"
DEFAULT_FINAL_TRANSCRIPTION_BACKEND = TRANSCRIPTION_BACKEND_MLX_WHISPER
DEFAULT_FINAL_TRANSCRIPTION_MODEL = "mlx-community/whisper-medium.en-mlx"
CODING_TRANSCRIPTION_TERMS = (
    "two sum, three sum, sliding window, two pointers, left pointer, right pointer, "
    "hash map, hash set, stack, queue, heap, priority queue, deque, trie, graph, "
    "tree, binary tree, binary search tree, linked list, node, edge, vertex, "
    "BFS, DFS, breadth-first search, depth-first search, topological sort, "
    "Dijkstra, Bellman-Ford, Floyd-Warshall, union find, disjoint set union, "
    "dynamic programming, DP, recursion, backtracking, memoization, tabulation, "
    "binary search, merge sort, quick sort, prefix sum, suffix sum, bitmask, "
    "bit manipulation, monotonic stack, greedy, invariant, constraints, "
    "time complexity, space complexity, Big O, O of n, O(n), O(log n), O(n log n), "
    "O(n squared), constant time, linear time, return true, return false, null, "
    "None, base case, edge case."
)
SYSTEM_DESIGN_TRANSCRIPTION_TERMS = (
    "system design, API, REST, gRPC, HTTP, WebSocket, request, response, endpoint, "
    "idempotent, idempotency, retry, timeout, backoff, rate limiter, throttling, "
    "load balancer, reverse proxy, gateway, service discovery, microservice, "
    "monolith, stateless, stateful, horizontal scaling, vertical scaling, "
    "availability, reliability, durability, consistency, eventual consistency, "
    "strong consistency, CAP theorem, quorum, leader, follower, primary, replica, "
    "replication, failover, partition, shard, sharding, consistent hashing, "
    "relational database, SQL, NoSQL, key-value store, document store, graph database, "
    "schema, index, secondary index, transaction, ACID, isolation level, lock, "
    "cache, Redis, Memcached, CDN, cache invalidation, write-through, write-back, "
    "message queue, Kafka, RabbitMQ, pub sub, event bus, stream processing, "
    "batch processing, async worker, cron job, object storage, blob storage, S3, "
    "data lake, warehouse, OLTP, OLAP, latency, throughput, p99, tail latency, "
    "SLA, SLO, observability, metrics, logs, tracing, alerting, circuit breaker, "
    "backpressure, saga, two-phase commit, exactly once, at least once, "
    "deduplication, hot key, fan out, fan in, pagination, cursor, authentication, "
    "authorization, OAuth, JWT, TLS, encryption, multi-tenant."
)
TRANSCRIPTION_INITIAL_PROMPT = (
    "This is a coding interview or system design interview transcript. "
    f"Coding interview terms may include: {CODING_TRANSCRIPTION_TERMS} "
    f"System design terms may include: {SYSTEM_DESIGN_TRANSCRIPTION_TERMS}"
)

TEST_PHOTO_CAPTURE_INITIAL_SECONDS = 60
TEST_PHOTO_CAPTURE_INTERVAL_SECONDS = 60
LIVE_PHOTO_CAPTURE_INITIAL_SECONDS = 10 * 60
LIVE_PHOTO_CAPTURE_INTERVAL_SECONDS = 10 * 60
INTERVIEW_PHOTO_DIR = Path("/Users/flora/interview")
STATIC_INTERVIEW_PHOTO_PATH = INTERVIEW_PHOTO_DIR / "static.jpg"
TEST_INTERVIEW_PHOTO_PATH = INTERVIEW_PHOTO_DIR / "test.jpg"
LIVE_INTERVIEW_PHOTO_PATH = INTERVIEW_PHOTO_DIR / "live.jpg"

CHATGPT_URL = "https://chatgpt.com/"
DEFAULT_CDP_URL = "http://127.0.0.1:9222"

SPEAKER_PROFILE_DIR = Path.home() / ".secondvoice"
SPEAKER_PROFILE_METADATA_PATH = SPEAKER_PROFILE_DIR / "interviewee-voice-profile.json"
SPEAKER_PROFILE_EMBEDDING_PATH = SPEAKER_PROFILE_DIR / "interviewee-voice-embedding.pt"
SPEAKER_MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
SPEAKER_MODEL_DIR = SPEAKER_PROFILE_DIR / "speaker-model"
SPEAKER_MATCH_THRESHOLD = 0.65
SPEAKER_ENROLLMENT_SILENCE_SECONDS = 1.5
SPEAKER_ENROLLMENT_MAX_SECONDS = 10.0
SPEAKER_ENROLLMENT_PROMPTS = [
    "I would start by clarifying the input constraints and expected output.",
    "My first approach is brute force, then I would optimize using a hash map.",
    "The time complexity is O of n, and the space complexity is O of n.",
    "Could I confirm whether the array contains negative numbers or duplicates?",
    "Let me walk through a small example to verify the logic.",
    "For the system design, I would first clarify the scale, users, and latency requirements.",
    "The main components are an API gateway, application servers, a database, and a cache.",
    "For high read traffic, I would add caching with Redis and define cache invalidation.",
    "The database choice depends on query patterns, consistency needs, and write volume.",
    "Let me summarize the tradeoffs before choosing the final design.",
]

STREAM_PROMPT = """\
You are SecondVoice, a mock interview evaluator.

For each transcript segment, first classify whether the text is from the interviewer or the interviewee.

You may receive an enrolled interviewee voice match confidence from 0.0 to 1.0. Use it only as a hint: higher values suggest the segment sounds like the enrolled interviewee, while lower values suggest the segment may be the interviewer or unknown. Prefer the transcript and accumulated context when they strongly disagree with the audio hint.

If it is from the interviewer, treat it as problem context for the rest of the mock interview. Respond with only two short lines: Classification: Interviewer and Context updated: a one-sentence summary of the new context.

If it is from the interviewee, use the accumulated interviewer context to write the ideal concise answer, then evaluate the interviewee's response against that ideal by calling out gaps only. For interviewee segments, respond with only three sections: Classification, Ideal concise answer, and Evaluation.

Keep the evaluation to one short paragraph focused on missing clarity, missing structure, technical gaps, and missed signals. If the speaker role is ambiguous, say so briefly inside the Classification line, make the best reasonable classification from the text, and continue. Do not invent problem facts beyond the accumulated transcript context.
"""
