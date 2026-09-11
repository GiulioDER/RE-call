# RE-call Hosted 1.0 vendor integration

## Product identity

`RE-call Hosted 1.0` is the fixed hosted distribution of the open source RE-call project. The
hosted product uses the public RE-call storage and retrieval layers plus the `recall_aml` API,
compiler, query planning, and evidence packing package in the same repository. The evaluated
deployment reports its immutable Git commit and prompt digest through `/version`.

The service is intended to remain reachable and configuration frozen for at least 30 days after
an AML submission. A release candidate must not be submitted until the contract, isolation,
concurrency, latency, persistence, and recovery gates in the committed preregistration pass.

## API

The base URL is supplied privately to AML. All request and response bodies are JSON. Add, Search,
and Delete accept either `Authorization: Bearer <key>` or `X-Api-Key: <key>`. The evaluation uses
a dedicated credential.

### `POST /v1/add`

```json
{
  "request_id": "stable-retry-key",
  "messages": [
    {
      "role": "user",
      "content": "The exact ordered message text",
      "timestamp": "2026-09-12T10:00:00Z"
    }
  ],
  "user_id": "exact AML user id",
  "session_id": "exact AML session id"
}
```

The service returns HTTP 200 only after raw evidence and one or more compiled or deterministic
fallback records have been embedded, atomically persisted, and made searchable. An identical
`request_id` replay returns the stored original response. Reusing the key with a different body
returns HTTP 409. The maximum request body is 2,000,000 bytes and one call accepts 1 through 256
messages.

### `POST /v1/search`

```json
{
  "query": "original query",
  "user_id": "exact AML user id",
  "top_k": 100,
  "options": {
    "choices": ["optional choice"],
    "context_chars": 7000,
    "historical": false,
    "include_raw": true
  }
}
```

The response is `{ "data": [...] }` in product rank order. Each item is stored memory evidence,
not a generated answer. Search returns no more than the requested `top_k`, 12 items, or the active
character budget, whichever limit is reached first.

### `GET /health`

Returns HTTP 200 only after the configured model clients were constructed and the serving
database schema and active generation passed readiness. Dependency failures return HTTP 503.

### `GET /version`

Returns product version, Git commit, schema version, embedding profile, retrieval profile,
generation model, reranker, and compiler prompt digest. It contains no credential, database URL,
host inventory, or other infrastructure identifier.

### `POST /v1/delete`

Accepts `{ "user_id": "exact AML user id" }`. It atomically deletes that user's hosted chunks,
learned sparse sidecars if present, and Add replay receipts. Other user tenants are not touched.

## Model and retrieval configuration

Every generative Add and Search operation uses exactly `gpt-4o-mini` with temperature zero and a
structured JSON response. Add stores ordered raw messages and up to eight `CodingMemoryRecord`
objects per chunk. Failed compilation degrades to a deterministic technical extract, never to an
unsuccessful Add.

The `hosted-quality` profile uses `voyage-4` passage and query embeddings, PostgreSQL lexical
retrieval, 100 candidates per retrieval leg, reciprocal rank fusion, and `voyage:rerank-2.5` over
the complete fused pool. A reranker failure serves the deterministic fused order. Query planning
may add at most four evidence-seeking facets and cannot return an answer. Packing removes near
duplicates and superseded evidence, preserves session diversity, rescues raw evidence, and caps
the returned pack at 12 items and 7,000 characters by default.

## Capacity, timeouts, and rate limits

The declared service capacity is 16 concurrent Add requests and 16 concurrent Search requests in
one process. The database pool is sized above the combined running capacity. Add provider and
database work must complete before the caller's 45 second integration timeout. Search callers use
a 10 second transport timeout, while the release gate requires a measured Search p95 below 5
seconds and Add p95 below 30 seconds. Oversized or invalid requests return HTTP 422. Authentication
failures return HTTP 401.

## Data handling

Exact message content is sent only to the configured OpenAI compiler and Voyage embedding service,
then stored in the dedicated PostgreSQL evaluation database. Search candidates are sent to the
configured Voyage reranker. Application logs contain only truncated user, request, and query
digests, counts, latency, fallback flags, and error classes. They never contain messages, queries,
model output, API keys, or database URLs.

The internal tenant is `aml_` plus the SHA256 digest of the exact `user_id`. PostgreSQL row level
security applies the tenant inside each pooled transaction. The product stores exact `session_id`
only as evidence metadata so a result can identify its source session. The Delete endpoint provides
evaluation data erasure without affecting another tenant.

## Availability and change control

The evaluated endpoint, image, commit, environment digest, and retrieval configuration are frozen
before submission. The service is monitored through `/health`; it restarts on process failure and
uses a Cloudflare Tunnel for public HTTPS. No feature, model, prompt, candidate width, context
budget, or index generation changes during the 30 day availability commitment. Emergency security
or availability repairs require a new product version and are disclosed to AML.

