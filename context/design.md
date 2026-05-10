# System Design Context

Use these notes by interview phase. During a live round, first privately infer and identify which phase the conversation is in, then pull only the most relevant prompts and tradeoffs from that section.

## 1. Question Phase

## 2. Clarify Functional Requirements
- Summarize the key requirements in one sentence
- Summarize the system under design in one sentence

## 3. Clarify Nonfunctional Requirements
- Clarify number of active users
- Real time vs. High Latency 
- Availablity vs Consistency
  -  Different endpoints can prioritize different property
- Is Auditability, Durability or Security relevant?

## 4. Data Model
- Core entities only
- Mention derived entities, relationship entites as needed
  - Derived entity: twitter timeline. inbox in chat applications
  - relationship entity: Follow relations. UserChat or DeviceChat

## 5. API Design
- Use REST-style endpoints with standard HTTP methods:
  - `GET` for reads
  - `POST` for creates/actions
  - `PUT/PATCH` for updates
  - `DELETE` for deletes
- For each endpoint, state:
  - path
  - request body or query params
  - response shape
  - key error cases
- Pagination if needed

## 6. Architecture

## 7. Deep Dive

### Pull vs Push
- Pull:
  - Simple.
  - Wasteful when updates are infrequent.
- Push:
  - Faster content delivery.
  - More complex (server must maintain user -> machine:port/session mapping).

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

### Fanout and read vs write

### Idempotency
- Client-generated UUID across retries (or server-issued).
- Scope dedup key under entity (e.g., `user_id + idempotency_key`).
- Validate same key => same request intent.
- Expire keys (e.g., 24h or 7d).
- Canonical intent keys examples:
  - `user_uuid + trip_uuid`
  - hash of canonical request payload
- For webhooks/queues/CDC, use upstream event IDs.

### Transactions
- Shard-local transactions.
- Cross-shard transactions.
- 2PC with transaction coordinator when required.
- Optimistic Concurrency Control (versioning/CAS)

### Lease
- Monotonic fencing token per lock acquisition.
- Advance accepted epoch only when business state/outbox commit succeeds.
- Practical ownership = whoever successfully advances state.
- Keep lease churn away from hot resource rows.
- Delayed queue for expiring leases

### Message Ordering
  - Per-entity message ordering
  - bounded buffering 
  - reconciliation 

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
- CDN

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
- Just mention "transaction" in the initial design. Deep dive later.
- No single source of truth.
- One side can fail and corrupt consistency.
- Use retries + idempotency.

### Read Your Own Writes
- Use strong-read semantics from leader when needed.

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

### Fault Tolerance for In-memory State
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

### Eager vs Lazy detection of expiry

### Lambda architecture - fast vs. slow

### File upload and download
- Blob storage for large files
- Chucking upload/download
- File versioning
- Compression before uploading
- Hot vs cold storage
- CDN
