from pathlib import Path


TRANSCRIPTION_BACKEND_FASTER_WHISPER = "faster-whisper"
TRANSCRIPTION_BACKEND_MLX_WHISPER = "mlx-whisper"
DEFAULT_ENDPOINT_TRANSCRIPTION_BACKEND = TRANSCRIPTION_BACKEND_MLX_WHISPER
DEFAULT_ENDPOINT_TRANSCRIPTION_MODEL = "mlx-community/whisper-tiny.en-mlx"
DEFAULT_FINAL_TRANSCRIPTION_BACKEND = TRANSCRIPTION_BACKEND_MLX_WHISPER
DEFAULT_FINAL_TRANSCRIPTION_MODEL = "mlx-community/whisper-small.en-mlx"

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

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
DEFAULT_ENDPOINT_MODEL = "qwen2.5:1.5b"
OLLAMA_KEEP_ALIVE = "60m"
OLLAMA_REQUEST_TIMEOUT_SECONDS = 10
ENDPOINT_LABEL_COMPLETE = "COMPLETE"
ENDPOINT_LABEL_INCOMPLETE = "INCOMPLETE"

ENDPOINT_SYSTEM_PROMPT = (
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
