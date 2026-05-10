# System Design Context

Use these notes by interview phase. During a live round, first identify which phase the conversation is in, then pull only the most relevant prompts and tradeoffs from that section.

## 1. Question Phase

### Restate the Problem
- Confirm the product/user action being designed.
- State the primary entities and the core user journey in one sentence.
- Ask whether the interviewer wants a broad end-to-end design or a focused subsystem.

### Scope Control
- Clarify whether this is real-time, near-real-time, or high-latency/batch.
- Clarify whether the product is read-heavy, write-heavy, or mixed.
- Clarify expected scale early: active users, QPS, data size, fanout size, geographic scope.
- Identify whether the system is user-facing, internal, or infrastructure.

### Early Tradeoff Framing
- Mention that different endpoints can prioritize different properties, such as availability vs consistency.
- Mention "transaction" for dual-write-sensitive workflows, then save details for deep dive.
- If the problem involves files, feeds, payments, notifications, or location, call out the likely hard part early.

## 2. Clarify Functional Requirements

### User Actions
- Create/update/delete core objects.
- Read/list/search core objects.
- Follow/subscribe/join/leave when there is a relationship graph.
- Upload/download when files or media are involved.
- Send/receive/acknowledge when messaging or notifications are involved.

### Realtime vs Pull
- Pull:
  - Simple.
  - Wasteful when updates are infrequent.
- Push:
  - Faster content delivery.
  - More complex because the server must maintain user -> machine:port/session mapping.

### API Pagination
- Use cursor pagination for large or changing result sets.
- Prefer stable sort keys and tie breakers.
- Define page size and next cursor behavior.

### File Upload and Download
- Blob storage for large files.
- Chunking upload/download.
- File versioning.
- Compression before uploading.
- Hot vs cold storage.
- CDN for download-heavy paths.

## 3. Clarify Nonfunctional Requirements

### Availability vs Consistency
- Different endpoints can prioritize different properties.
- Strong consistency for money, ownership, permissions, and read-your-own-write-sensitive workflows.
- Eventual consistency for feeds, counters, analytics, and recommendations when acceptable.

### Latency and Throughput
- Define p50/p95/p99 expectations.
- Separate write latency from read latency.
- Call out whether batching is acceptable.

### Reliability
- Regional failover.
- Exponential backoff + jitter to avoid retry storms.
- Load shedding for low-priority requests.
- RPC timeouts.
- Circuit breaker to stop hammering failing dependencies.

### Read Heavy vs Write Heavy
- Read heavy:
  - Replicate data for read scale.
  - Use cache aggressively.
- Write heavy:
  - Batch writes.
  - Prefer NoSQL when high throughput and horizontal scale are required.

### Read Your Own Writes
- Use strong-read semantics from leader when needed.
- Route the user's immediate follow-up read to the write leader or use session/version tokens.

## 4. Data Model

### Entity and Relationship Modeling
- Name the core tables/collections before optimizing.
- Include primary keys, foreign keys/entity references, timestamps, and status fields.
- Decide whether relationship queries are user-centric or item-centric.

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

### Sharding: Item vs User
- Shard by item:
  - Even item distribution.
  - User list queries require scatter-gather.
- Shard by user:
  - Single-shard user queries.
  - Hotspot risk for VIP/high-traffic users.
- Hotspot mitigation:
  - Add salt to key if hot keys are predictable.
  - Move hotspots to idle servers.
  - Rate limit.
  - Queue + batch operations.

### Transactions
- Shard-local transactions.
- Cross-shard transactions.
- 2PC with transaction coordinator when required.
- Optimistic Concurrency Control (versioning/CAS).

### Event Timestamp vs Processing Timestamp
- Event timestamp:
  - Reflects real-world event time.
  - More complex because of stragglers and watermarks.
  - Better retry/backfill consistency.
  - Higher latency/memory when waiting for late data.
- Processing timestamp:
  - Simpler and lower latency.
  - Trusted server time.
  - Less accurate for real-world ordering.
  - Not replay-reproducible unless externalized.

## 5. API Design

### API Shape
- Start with the key read and write endpoints.
- Include idempotency keys for retried writes.
- Include pagination params for list endpoints.
- Include auth context and ownership checks.
- Define request/response status fields for async workflows.

### Idempotency
- Client-generated UUID across retries or server-issued key.
- Scope dedup key under entity, for example `user_id + idempotency_key`.
- Validate same key => same request intent.
- Expire keys, for example 24h or 7d.
- Canonical intent key examples:
  - `user_uuid + trip_uuid`
  - Hash of canonical request payload.
- For webhooks/queues/CDC, use upstream event IDs.

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
- Prefer transactional outbox or event history when workflow correctness matters.

## 6. Architecture

### Basic Architecture Flow
- Client/API gateway.
- Service layer split by workflow and load.
- Primary datastore.
- Cache for hot reads.
- Queue/event bus for async work.
- Worker fleet for expensive or retryable tasks.
- Object storage/CDN for large blobs.
- Observability for metrics/logs/tracing.

### Microservice Split Heuristics
- Split by load characteristics.
- Split by workflow type:
  - IO-bound paths.
  - CPU-bound paths.
- Keep the first pass simple; split only where scaling, ownership, or failure isolation demands it.

### Cache Strategy
- Use cache for hot read paths and latency reduction.
- Keep invalidation strategy explicit.
- Prefer cache + replication for read-heavy services.
- CDN for static or media-heavy reads.

### Fanout and Read vs Write
- Fanout-on-write:
  - Fast reads.
  - Expensive writes.
  - Good when reads dominate and fanout is bounded.
- Fanout-on-read:
  - Simple writes.
  - Expensive reads.
  - Good when writes dominate or fanout is huge.
- Hybrid:
  - Precompute for normal users.
  - Pull or rank dynamically for celebrities/hot keys.

### Message Ordering
- Per-entity message ordering.
- Bounded buffering.
- Reconciliation for late or out-of-order events.

## 7. Deep Dive

### Lease
- Monotonic fencing token per lock acquisition.
- Advance accepted epoch only when business state/outbox commit succeeds.
- Practical ownership = whoever successfully advances state.
- Keep lease churn away from hot resource rows.
- Delayed queue for expiring leases.

### Fault Tolerance for In-Memory State
- Snapshot in-memory state with watermark.
- Recover from latest snapshot, then replay after watermark.
- If history is short, full replay is acceptable.

### State-Driven vs Event-Driven
- State-driven pitfalls:
  - State explosion with retries/callbacks/timeouts.
  - Control-state leakage into business schema.
  - Poor explainability of transitions.
- Event-driven tradeoff:
  - Replay/materialization complexity can be overkill for simple flows.

### Eager vs Lazy Detection of Expiry
- Eager:
  - Better user-visible freshness.
  - More background work and timer management.
- Lazy:
  - Simpler and cheaper.
  - Expired state may remain until touched.

### Lambda Architecture - Fast vs Slow
- Fast path:
  - Low latency and approximate/incremental.
  - More operational complexity.
- Slow path:
  - Accurate, replayable, and good for correction/backfill.
  - Higher latency.

### Workflow Engine Script (Payments Example)

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
