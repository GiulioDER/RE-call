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
      "timestamp": 1726133200000
    }
  ],
  "user_id": "exact AML user id",
  "session_id": "exact AML session id"
}
```

The service returns HTTP 200 only after the active variant's evidence and retrieval sidecars have
been persisted and made searchable. Coding compiler variants also persist compiled or deterministic
fallback records. An identical
`request_id` replay returns the stored original response. Reusing the key with a different body
returns HTTP 409. The maximum request body is 44 MiB so that 30 MiB of decoded inline images can
survive Base64 expansion plus JSON framing. One call accepts 1 through 256 messages.

```json
{
  "success": true,
  "request_id": "stable-retry-key",
  "user_id": "exact AML user id",
  "session_id": "exact AML session id"
}
```

### `POST /v1/search`

```json
{
  "query": "original query",
  "user_id": "exact AML user id",
  "top_k": 100,
  "options": ["A. First choice", "B. Second choice"]
}
```

`options` is an optional array of answer choice strings and is omitted for open questions. The
response is `{ "data": [...] }` in product rank order. Every item contains at least `id` and
`content`; optional ranking and provenance fields may also be present. Each item is stored memory
evidence, not a generated answer. Text packing variants return no more than the requested `top_k`,
12 items, or the active character budget, whichever limit is reached first. Multimodal preservation
variants use the requested `top_k` plus the decoded response-media budget.

The response headers `X-Recall-Facet-Fallback` and `X-Recall-Reranker-Fallback` are each `0` or `1`
and report whether that request used the corresponding deterministic fallback. They are operational
metadata and do not change the AML JSON body.

### Multimodal content

The experimental `MM0_caption`, `MM1_preserve`, and `MM2_dual` variants accept the AML ordered
content shape in both Add messages and the Search query:

```json
[
  {"type": "text", "text": "The deployment panel shows"},
  {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
  {"type": "text", "text": "after the repair"}
]
```

Only inline Base64 JPEG, PNG, and WebP images are accepted. Each decoded image is limited to 10
MiB. One Add request and one Search response are each limited to 30 MiB of decoded images.
`MM0_caption` uses supplied text and deliberately drops images. `MM1_preserve` stores each original
image once per tenant under its SHA256 and reconstructs the exact ordered parts. `MM2_dual` adds a
separate `voyage-multimodal-3.5` vector tenant and fuses its candidate order with text retrieval.
The main text chunks contain only source text and opaque media references, never Base64.

Voyage multimodal embeddings are part of the Industry configuration. The current AML Academic and
Open Source form says any model used during Add or Search must be `gpt-4o-mini`; this repository
does not claim that Voyage is eligible for those tracks without written organizer confirmation.

### `GET /health`

Returns HTTP 200 only after the serving database schema and active generation passed readiness and
startup completed one bounded live request against every model provider stage used by the active
variant. Dependency failures prevent startup or return HTTP 503.

### `GET /version`

Returns product version, Git commit, schema version, embedding profile, retrieval profile,
generation model, reranker, compiler prompt digest, and facet planner prompt digest. It contains no
credential, database URL, host inventory, or other infrastructure identifier.

### `POST /v1/delete`

Accepts `{ "user_id": "exact AML user id" }`. It atomically deletes that user's hosted chunks,
learned sparse sidecars if present, and Add replay receipts. Other user tenants are not touched.

## Model and retrieval configuration

Every generative Add and Search operation uses OpenAI `gpt-4o-mini` through OpenRouter, with the
fixed OpenRouter model identifier `openai/gpt-4o-mini`, temperature zero, and a structured JSON
response. In the submitted product variant, Add stores ordered raw messages and up to eight
`CodingMemoryRecord` objects per session. Each generated record must cite at least one exact source
span by message ordinal and character offsets. A mismatched ordinal, boundary, or quote rejects the
record. Unsupported entities are removed, while outcome and validation fields survive only when
their exact text occurs in one source message. Failed compilation degrades to a deterministic
technical extract, never to an unsuccessful Add.

Raw message content is segmented at 4,500 characters. Including the maximum role and timestamp
prefix, every segment remains returnable by the smallest registered 5,000 character evidence arm.

The `hosted-quality` profile uses `voyage-4` passage and query embeddings, PostgreSQL lexical
retrieval, 100 candidates per retrieval leg, reciprocal rank fusion, and `voyage:rerank-2.5` over
the complete fused pool. A reranker failure serves the deterministic fused order. Query planning
may add at most four evidence-seeking facets and cannot return an answer. Packing removes near
duplicates and superseded evidence, preserves session diversity, rescues raw evidence, and caps
the returned pack at 12 items and 7,000 characters by default.

Packing preserves the strongest two thirds of the ranked item budget as a relevance core. It then
prefers previously unseen sessions before returning to repeated sessions. This prevents diversity
from displacing multi-record evidence needed to resolve one task while still broadening the tail.

## Engineering experience representation experiment

The internal `E0_raw`, `E1_compiled`, and `E2_compiled_raw` variants isolate the representation
decision described in the 2026-09-17 preregistration. They keep query facets, reranking, and compact
packing disabled, so their only intended difference is whether Add persists raw message segments,
compiled records, or both. `E1_compiled` permits an Add response with `raw_count` equal to zero and
remains searchable through generated or deterministic fallback records.

These variants are experimental configurations, not three distinct AML submissions. The selected
representation must pass retrieval replay and executable task gates before it is promoted into the
single hosted candidate used for Smoke and Full evaluation.

## Capacity, timeouts, and rate limits

The declared service capacity is 16 concurrent Add requests and 16 concurrent Search requests in
one process. The database pool is sized above the combined running capacity. Add provider and
database work must complete before the caller's 45 second integration timeout. Search callers use
a 10 second transport timeout, while the release gate requires a measured Search p95 below 5
seconds and Add p95 below 30 seconds. Oversized or invalid requests return HTTP 422. Authentication
failures return HTTP 401.

Add compilation makes at most three provider attempts with an eight second timeout per attempt.
Search facet planning makes one attempt with a two second timeout, then immediately uses the
original query and reports the fallback. The shorter Search policy prevents an optional planner
outage from consuming the five second product latency gate by itself.

## Data handling

Exact text message content is sent through OpenRouter to the configured OpenAI `gpt-4o-mini`
compiler when the active coding variant enables it, and to the Voyage embedding service. In
`MM2_dual`, ordered text and image Data URIs are sent to Voyage's multimodal embedding endpoint.
Images above Voyage's 16 million pixel input limit are proportionally resized only for that
transient embedding request. RE-call stores and returns the exact original image bytes. Evidence
is stored in the dedicated PostgreSQL evaluation database. Search candidates
are sent to the configured Voyage reranker when the active variant enables it. Application logs contain only
truncated user, request, and query
digests, counts, latency, fallback flags, and error classes. They never contain messages, queries,
model output, API keys, or database URLs.

The internal tenant is `aml_` plus the SHA256 digest of the exact `user_id`. Multimodal media and
vectors use deterministic `_media` and `_mm` tenant suffixes under the same user boundary.
PostgreSQL row level security applies the tenant inside each pooled transaction. The product stores
exact `session_id` only as evidence metadata so a result can identify its source session. The Delete
endpoint erases the primary tenant and both multimodal sidecars without affecting another user.

## Measured multimodal admission result

Measured on 2026-09-20 at exact RE-call commit
`61dc7b8d0cbb8714904a1e07e40819c76a1c1971`, the frozen public MemEye Brand pilot completed all
three arms and passed its independent audit. The mechanical verdict was `NO_GAIN`.

`MM0_caption`, `MM1_preserve`, and `MM2_dual` achieved mean debiased exact match of `0.4655`,
`0.4310`, and `0.4741`, respectively. All three achieved any-clue Recall at 10 of `0.9655` and
Recall at 100 of `1.0`. Dual retrieval improved answer exact match over preservation by `0.0431`,
but it produced no Recall at 10 gain. Preservation regressed answer exact match by `0.0345`
relative to captions. Every contract, isolation, cleanup, identity, and response-budget check
passed.

This is directional evidence from one public MemEye scenario, not an AML hosted result or
leaderboard score. The frozen gate did not authorize scientific promotion or the full
eight-scenario follow-up. The user nevertheless retained `MM2_dual` as the preferred operational
multimodal candidate because it had the strongest measured answer accuracy without reducing
Recall at 10 or Recall at 100. This explicit operational selection preserves the `NO_GAIN`
verdict and does not claim that the registered retrieval gate passed.

Select the retained multimodal configuration only for a multimodal deployment:

```bash
RECALL_AML_VARIANT=MM2_dual
```

The global hosted default remains unchanged, so this track-specific choice cannot affect textual
or coding deployments. Recompute the selector and audit with the commands in
[`2026-09-20-aml-multimodal-memeye-brand-v2.md`](preregistrations/2026-09-20-aml-multimodal-memeye-brand-v2.md).

### Routed specialist use and embedding reuse

`C7_routed_specialists` incorporates the retained MM2 behavior as its visual route while keeping
Code4 as the conservative default and Context4 for explicit conversational memory. MM2 does not
enter a coding query's candidate pool. Each route performs rank fusion only inside the selected
specialist path, so heterogeneous cosine scores are never compared or averaged.

Hosted deployments may opt into the shared content addressed cache with:

```bash
RECALL_AML_EMBED_CACHE_PATH=/absolute/path/to/aml-hosted-embeddings.sqlite
```

The key binds the complete embedding profile, vector dimension, encoder purpose, and exact input.
Context4 additionally binds the complete ordered document group and chunk ordinal. Multimodal keys
bind canonical structured input and separate document from query vectors. The cache stores only
derived vectors and cannot make one tenant's rows searchable from another tenant. On VPS2 the cache
is placed inside the existing cross process embedding lock so a second hosted process rechecks the
cache after the first process fills a miss.

## Availability and change control

The evaluated endpoint, image, commit, environment digest, and retrieval configuration are frozen
before submission. The service is monitored through `/health`; it restarts on process failure and
uses a Cloudflare Tunnel for public HTTPS. No feature, model, prompt, candidate width, context
budget, or index generation changes during the 30 day availability commitment. Emergency security
or availability repairs require a new product version and are disclosed to AML.
