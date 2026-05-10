# System Design Deep Dive Context

## Common Deep Dives

### Read Heavy vs Write Heavy
- Read heavy:
  - Replicate data for read scale.
  - Use cache aggressively.
- Write heavy:
  - Batch writes.
  - Prefer NoSQL when high throughput and horizontal scale are required.

### Cache Strategy
- Use cache for hot read paths and latency reduction.
- Keep invalidation strategy explicit.
- Prefer cache + replication for read-heavy services.

### Microservice Split Heuristics
- Split by load characteristics.
- Split by workflow type:
  - IO-bound paths
  - CPU-bound paths

### SQL vs NoSQL vs In-Memory KV
- SQL:
  - More expensive.
  - ACID transactions.
  - Suitable for many read-heavy transactional workflows.
- NoSQL:
  - Better sharding + replication support.
  - Suitable for write-heavy/high-throughput workloads.
- In-memory KV:
  - Very fast.
  - Suitable for relatively small working sets.

### Pull vs Push
- Pull:
  - Simple.
  - Wasteful when updates are infrequent.
- Push:
  - Faster content delivery.
  - More complex (server must maintain user -> machine:port/session mapping).

### Prevent Cascading Failures
- Regional failover.
- Exponential backoff + jitter to avoid retry storms.
- Load shedding for low-priority requests.
- RPC timeouts.
- Circuit breaker to stop hammering failing dependencies.

### Queue vs RPC
- Queue:
  - Async, decoupled, supports batching.
  - Easier independent evolution and integration.
  - Load smoothing protects downstream.
  - Durable retries + DLQ.
  - Helps isolate upstream from cascading failures.
  - Tradeoff: operational overhead.
- RPC:
  - Realtime and simpler request tracing/debugging.
  - Tighter coupling and greater dependency-failure blast radius.

### Dual Write Risks
- No single source of truth.
- One side can fail and corrupt consistency.
- Use retries + idempotency.

### Sharding: Item vs User
- Shard by item:
  - Even item distribution.
  - User list queries require scatter-gather.
- Shard by user:
  - Single-shard user queries.
  - Hotspot risk for VIP/high-traffic users.
- Hotspot mitigation:
  - Add salt to key (if hotkeys are predictable).
  - Move hotspots to idle servers.
  - Rate limit.
  - Queue + batch operations.

### Fault Tolerance for Stateful Components
- Snapshot in-memory state with watermark.
- Recover from latest snapshot, then replay after watermark.
- If history is short, full replay is acceptable.

### Exactly-Once / Concurrency
- Optimistic concurrency (version/CAS).
- Pessimistic concurrency (locking).
- Lease-based ownership.
- Fencing tokens:
  - Monotonic token per lock acquisition.
  - Advance accepted epoch only when business state/outbox commit succeeds.
  - Practical ownership = whoever successfully advances state.
  - Useful when keeping lease churn off hot rows.

### Transactions
- Shard-local transactions.
- Cross-shard transactions.
- 2PC with transaction coordinator when required.

### Idempotency
- Client-generated UUID across retries (or server-issued).
- Scope dedup key under entity (e.g., `user_id + idempotency_key`).
- Validate same key => same request intent.
- Expire keys (e.g., 24h or 7d).
- Canonical intent keys examples:
  - `user_uuid + trip_uuid`
  - hash of canonical request payload
- For webhooks/queues/CDC, use upstream event IDs.

### Event Timestamp vs Processing Timestamp
- Event timestamp:
  - Reflects real-world event time.
  - More complex (stragglers, watermarks).
  - Better retry/backfill consistency.
  - Higher latency/memory when waiting for late data.
- Processing timestamp:
  - Simpler and lower latency.
  - Trusted server time.
  - Less accurate for real-world ordering.
  - Not replay-reproducible unless externalized.

### Read Your Own Writes
- Use strong-read semantics from leader when needed.

### gRPC vs HTTP
- gRPC:
  - Lower overhead/faster.
  - Strong schema and codegen tooling.
- HTTP:
  - More overhead.
  - Better browser/native ecosystem support.

### TCP vs UDP
- TCP:
  - Reliable, ordered delivery, integrity checks.
  - Higher latency/overhead.
- UDP:
  - Lower latency.
  - Useful for streaming/DNS-like patterns.
  - No delivery/order guarantees.

### State-Driven vs Event-Driven
- State-driven pitfalls:
  - State explosion with retries/callbacks/timeouts.
  - Control-state leakage into business schema.
  - Poor explainability of transitions.
- Event-driven tradeoff:
  - Replay/materialization complexity can be overkill for simple flows.

## Workflow Engine Script (Payments Example)

Let me start from the simplest design.
The transaction API hands the request to a worker, and the worker talks to the payment gateway. That is easy to build, but it has a big problem: if the worker crashes in the middle, we may not know exactly what already happened. For example, charge may already have succeeded, but if we lose that result, we might restart from the wrong place and do duplicate work.

So the first thing I add is a durable event history store.
It records every important step that has happened so far: payment requested, authorization requested, authorization succeeded, charge requested, charge succeeded, and so on. This becomes the source of truth. If an event is durably committed, we treat it as having happened. If not, we assume it did not happen yet.

Once I have that, I want to separate what happened from what still needs to be done.
So I will introduce a task queue. The event history tells me the current progress of the payment, and from the current state plus a new event, the system can decide the next task to enqueue. For example, if authorization succeeds, we enqueue a charge task. Workers pull tasks from the queue and execute the external work.

Next, I need to handle the fact that payment gateways are usually asynchronous.
I do not want workers sitting there polling and waiting. So instead, after a worker sends a request to the gateway, the gateway can call us back through a webhook. The callback handler appends a new event into the history store, like authorization succeeded or charge succeeded, and that drives the next task.

I also need to handle missing callbacks, so I add a durable timer.
When I send a request to the gateway, I also schedule a timer. If the callback arrives in time, great. If not, the timer fires and appends a timeout event, and then the system can decide whether to retry, reconcile, or fail safely.

The tricky part is the charge step, because retries can create duplicate charges.
So if the payment gateway supports idempotency, I generate an idempotency key and reuse it on retries. If it does not, then on ambiguous failures I move the payment into a pending-verification state and trigger reconciliation, either automatically by querying the gateway for status or manually by creating a ticket for an agent.

Finally, if a worker dies while holding a task, I detect that with heartbeat plus lease timeout.
Once the lease expires, another healthy worker can pick up the task and continue, using the event history to resume from the last confirmed step.

So the overall idea is:
the event history store records what has happened, the task queue holds what to do next, callbacks and timers append new facts, and workers can always resume safely from durable history.

## System Design Questions Breakdown

### Interview Framing and L6 Signals
- Start with assumptions and explicitly invite correction.
- Summarize I/O expectations early.
- Analyze tradeoffs before introducing new components.
- Focus on heart-of-problem topics first; de-prioritize low-signal details unless asked.
- Periodically ask interviewer which area to deep dive.
- L5 vs L6:
  - L5: defend one solution.
  - L6: propose multiple options and analyze tradeoffs/hybrids.

### Proximity Service
- Example assumptions:
  - users: 1B
  - latency: 1s
  - freshness: 1h
- Core: information retrieval + geo indexing.
- Input/output expectation:
  - Given `<lat, lng>`, return ~1k nearby POI IDs.
- Indexing:
  - Quadtree, geohash.
  - In-memory index for fast lookup.
  - On-disk copy for reconstruction after failure.
- Multiple criteria search:
  - Search engine/query optimizer path.
  - App-level reduction of search space as early as possible.

### Online Judge / Job Scheduler
- Asynchronous jobs, low QPS.
- Core entities:
  - JobIntent
  - NextRun
- Non-functional:
  - high availability
  - real-time + near-real-time execution
  - at-least-once execution
- Data/indexing:
  - precompute `next_run_at`
  - index `next_run_at` to avoid full scans
- Fast vs regular path:
  - high-urgency queue vs regular queue
  - possible synchronous fast path
- Design topics:
  - state transitions
  - internal vs external status
  - client retry vs server retry
  - idempotency (client and side-effect layer)
  - stale worker detection
- Security:
  - sandbox workers (Lambda/function style isolation)
  - restricted worker access
- Queue vs DB:
  - DB scans can be slow
  - queue adds dependency and ops/debug overhead
- Sharding/index:
  - shard key tradeoffs: `next_run_at` vs `run_id` vs `user_id`
  - global secondary index vs local secondary index
- Storage:
  - code storage: blob store vs DB
  - cap code size

### Chat Service
- Core model:
  - user, chat, message, client (multi-device)
- Non-functional:
  - prioritize availability
  - best-effort consistency
  - effectively-once delivery
  - ordering guarantees
  - real-time delivery
- Delivery architecture:
  - centralized realtime server for peer communication
  - maintain `user_id -> client_connections`
  - consistent hashing for connection-owner machine
  - pub/sub fanout to push nodes
- Reconnect/replay:
  - inbox table for undelivered messages
  - last-received watermark per client
  - client sends last event id on reconnect
- Push vs pull:
  - push for active users
  - polling/fanout-on-read for inactive users
- Ordering and IDs:
  - monotonic message IDs
  - NTP time sync
  - ID generator tradeoff: dedicated service vs UUID
- DB modeling:
  - SQL vs NoSQL for message DB
  - message sharding key
  - batch writes
  - `UserChat` read receipts, `DeviceChat` delivery receipts
  - immutable `MessageEvent` + mutable `Message` state table
- Follow-ups:
  - group chat, rich media, edit/delete, online indicator, receipts, video chat

### Instagram/Twitter-style Feed
- Read-heavy with massive fanout.
- Hot vs cold storage.
- Photo binary in blob store (separate from metadata table).
- Photo chunking and upload pipeline.
- SQL vs NoSQL: either can work; complexity often in sharding/replication/cache.
- Sharding key should be data-driven from workload analysis.
- Workload isolation:
  - high-bandwidth workloads on specialized machines.
- Fanout on read vs write tradeoff.
- Latency:
  - cache/CDN for photos
  - potentially optimize CDN usage for popular/paid tiers
- Consistency:
  - dual-write between blob + metadata DB handling
- Upload path:
  - presigned URL direct-to-blob
- Pagination:
  - cache recent data for first-page latency

### Dropbox-style File Storage
- Goal: persist/retrieve massive files.
- Non-functional:
  - availability, low latency, security, durability
- Upload path:
  - chunking + chunk size policy
  - file/chunk hashing
  - parallel upload
  - retry failed chunks only
  - presigned URL uploads
- Storage tiering:
  - hot/cold (e.g., S3 vs Glacier conceptually)
  - CDN for reads
- Security:
  - encryption + ACL
- Data modeling:
  - metadata sharding key selection
  - versioning (file/chunk)
  - write conflict handling
- Reliability:
  - checksum validation
  - failure recovery
  - dual-write ordering between DB and blob store
- Principles:
  - avoid over-indexing on specific vendor tools/libraries

### Auction System
- Requirement split:
  - view path: high availability
  - bidding path: strong consistency + realtime highest bid
- Write correctness:
  - OCC/locking for highest-bid updates
  - tie-breaker (first valid bid wins)
- Read path:
  - push vs pull updates
  - cache maintenance
  - cache sharding by item vs user
- Write path:
  - SQL vs NoSQL
  - in-memory KV for `item -> highest_bid` + WAL for recovery
  - fail-fast optimization (many bids fail)
- Scale/fairness:
  - request dedupe
  - surge buffering via queue
  - delayed tasks for auction end/extension
  - next-highest bidder fallback if winner fails payment

### Gaming Leaderboard
- Efficient retrieval via in-memory index/sorted structure.
- Throughput estimation first.
- Top-10 optimization via replica/cache.
- Hotspot vs scatter-gather tradeoff.
- In-memory sorted list vs DB secondary index on score.
- Optional write queue + write-through cache.
- Sharding + replication for recovery.

### Payment System / Wallet
- Prioritize correctness: exactly/effectively-once semantics.
- Non-functional:
  - security, auditability, durability, order guarantees, realtime status
- Core entities:
  - payor, payee, payment intent, transaction
- Exactly-once pattern:
  - at-least-once + idempotency dedupe
- Unknown/timeout failures:
  - do not blindly retry
  - reconcile side effects (API query/batch files/manual)
- State management:
  - finite state machine (`not_started/pending/succeed/failed`)
  - retry metadata (`retry_times`, `last_retry_at`, `created_at`)
- Event log vs mutable state table tradeoff.
- Idempotency vs distributed transactions:
  - idempotency localizes correctness per component
  - distributed transactions add coordinator/coupling/throughput cost
- Performance/ops:
  - async processing + callbacks
  - queue for load smoothing and batch cost control
  - connection reuse, timeout policy
- Additional concerns:
  - settlement files
  - fraud detection
  - duplicate-payment anomaly handling

### Metric/Monitoring System
- High availability first.
- Meta-monitoring (monitor the monitor).
- Multi-region/provider resilience.
- Tiered explanations and concise summaries.
- Push vs pull telemetry tradeoff.
- Replication + alerting reliability.
- Time-series storage.
- At-least-once alert delivery, retry until ACK, multi-channel.
- Raw vs aggregate retention based on customer need.

### Advertisement Event Aggregation
- Focus: counting + realtime query + strong tracking consistency.
- Global vs local state strategy.
- Cache global states when recalculation is expensive.
- Stragglers: freshness vs accuracy tradeoff.
- Queue sharding:
  - `message_id` gives even distribution but out-of-order per ad
  - `ads_id` preserves per-ad order but hot-spot risk
- Stream processor maps to output schema and business logic.
- In-memory KV vs DB under memory constraints.
- Cache as optimization vs hard dependency.
- Lambda style (batch + streaming) option.
- Idempotent click/impression counting via unique impression IDs.
- Audit trail for click/impression events.
- Hot shard mitigation:
  - split hot key into subkeys
  - batch increments
  - roll-up tables for larger windows
- Primary vs secondary requirement tradeoff framing.

### Ticketmaster-style Booking
- Search/view: prioritize availability.
- Booking: prioritize consistency.
- Hotspot handling for popular events.
- Lease table separate from hot seat rows.
- Keep lease churn off critical-path seat table.
- Delayed queue for lease expiration.
- Transaction/CAS for seat status race prevention.
- Virtual waiting queue.

### FB Live Comments
- Viewing availability + realtime comments + pagination.
- Push vs poll:
  - poll for mega streams to avoid fanout-on-write explosion.
- UX smoothing:
  - comment sampling
  - avoid frequent mode flapping
- Handle SSE/websocket disconnect scenarios.

### KV Store Design Topics
- TTL deletion:
  - eager deletion via delayed queue
  - lazy deletion on read + background compaction
- Compaction/GC (amortized costs).
- Replication modes:
  - sync/async
  - leader-follower / P2P
- Sharding and hotspots:
  - hot read: cache/read replicas
  - hot write: batching/subkeys/dynamic split/hotspot move
- Connection pool/cache strategies.

### General Response Framework for New Questions
- Circle back on unclear details.
- Identify primary requirement vs secondary requirement early.
- Divide into read path and write path.
- Prefer first-principles indexing/throughput estimation.
- Evaluate tradeoffs before naming components.
- Avoid over-specific technology/framework anchoring unless interviewer asks.
- Proactively propose moving to another deep-dive topic when coverage is sufficient.
