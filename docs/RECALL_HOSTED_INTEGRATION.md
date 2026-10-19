# RE-call Hosted 1.0 vendor integration

## Product identity

`RE-call Hosted 1.0` is the fixed hosted distribution of the open source RE-call project. The
hosted product uses the public RE-call storage and retrieval layers plus the `recall_aml` API,
compiler, query planning, and evidence packing package in the same repository. The evaluated
deployment reports its immutable Git commit and prompt digests through `/version`.

The build identity of a deployment is the Git commit that `/version` reports as `git_commit`. The
`version` field is the product version, `1.0.0`, and it is the same on every build, so it does not
identify one. Each AML run is identified by its fixed commit, listed under
[Runs of record](#runs-of-record).

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
been persisted and made searchable. Variants with the compiler on also persist the records the
compile produced; whether a failed compile stores deterministic fallback records in their place
depends on the variant (C9 stores none). An identical
`request_id` replay returns the stored original response. Reusing the key with a different body
returns HTTP 409. The maximum request body is 44 MiB so that 30 MiB of decoded inline images can
survive Base64 expansion plus JSON framing. That body limit is the only bound on an Add's size:
there is no per-message length or message-count cap, and an empty or all-blank message list is
stored as a durable, empty Add.

With `RECALL_AML_ADD_RESPONSE_BUDGET_SECONDS` set (off unless set; 0, or 10 to 95 seconds), an Add
still running when the budget runs out is answered HTTP 503 `{"error": "add_in_progress"}` with
`Retry-After: 1` and `X-Recall-Add-Pending: 1`, and the work carries on. A resend with the same
`request_id` and body joins it, or, once it has finished, is answered from its stored response.
HTTP 200 still means the Add is durable, so this 503 is a normal answer to a slow Add and not a
failure. When the service already holds its maximum number of Adds in flight, any further Add
(including a resend) waits with its body unread, and is answered the same way if its budget runs
out first.

With `RECALL_AML_TENANT_LOCK_WAIT_SECONDS` also set (off unless set; 0, or 25 to 1800 seconds, and
only together with the response budget and at least two Add slots), an Add of a user whose earlier Add is still running
queues behind it, first in, first out, and starts the moment it finishes. A delete or a sparse
backfill of that user joins the same queue and waits at most the response budget. An Add that
cannot start within that many seconds, or that finds its user's queue already full, and a delete
or backfill that cannot start within the response budget, change nothing and are answered HTTP
503 `{"error": "tenant_busy"}` with `Retry-After: 60`.

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
evidence, not a generated answer. The C9 variant returns at most the requested `top_k` items and
applies no character budget. The packing variants (`A4_pack_5000`, `A4_pack_7000`, `A4_pack_9000`)
return no more than the requested `top_k`, 12 items, or the active character budget, whichever
limit is reached first. A response that carries images is also bounded by the decoded response
media budget.

The response headers `X-Recall-Facet-Fallback` and `X-Recall-Reranker-Fallback` are each `0` or `1`
and report whether that request used the corresponding deterministic fallback. They are operational
metadata and do not change the AML JSON body. A variant with no query facets and no reranker, such
as C9, never uses either fallback.

### Multimodal content

The routed specialist variants (`C7`, `C8`, and `C9`) and the experimental `MM0_caption`,
`MM1_preserve`, and `MM2_dual` variants accept the AML ordered content shape in both Add messages
and the Search query:

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
The main text chunks contain only source text and opaque media references, never Base64. C9
preserves images as `MM1_preserve` does and embeds them with `voyage-multimodal-3.5` as `MM2_dual`
does; when its image leg runs is described under
[Model and retrieval configuration](#model-and-retrieval-configuration).

AML's rule for the Open-source Methods division is that every generative model used during Add or
Search is `gpt-4o-mini`. The Voyage models RE-call uses are embedding models, and each one is named
in the method disclosure of the run that used it.

### `GET /health`

Returns HTTP 200 only after the serving database schema and active generation passed readiness and
startup completed one bounded live request against every model provider stage used by the active
variant. Dependency failures prevent startup or return HTTP 503.

### `GET /version`

Returns the product name and version, the build identity `git_commit`, the schema version, the
`variant`, its embedding, lexical, window, routing, and rendering profiles, `active_components`,
the generation model, the configured reranker, the compiler and facet planner prompt digests, the
Add response budget in seconds (`add_response_budget_seconds`, 0 when off), and the tenant queue
budget in seconds (`tenant_lock_wait_seconds`, 0 when off). It contains no credential, database
URL, host inventory, or other infrastructure identifier.

`git_commit` is the build identity: it names the exact commit the process runs, and together with
`variant` and the profiles beside it, the configuration that commit was started with. `version` is
`1.0.0` on every build. `active_components` says which stages run (among them `compiler`,
`facets`, `reranker`, `graph_sidecar`, `multimodal_native`, `canonical_bm25`, `exact_dense`, and
`atomic_rescue`). The `reranker` field names the reranker model the service is built with and
`retrieval_profile` is the fixed label `hosted-quality`, whatever the variant, so neither says
whether a reranker runs: `active_components` does. Under C9 it reports `reranker` and `facets` as
`false`, and `compiler` as `false` when `RECALL_AML_COMPILER=0` is set.

### `POST /v1/delete`

Accepts `{ "user_id": "exact AML user id" }`. It atomically deletes that user's hosted chunks,
learned sparse sidecars if present, and Add replay receipts. Other user tenants are not touched.

## Model and retrieval configuration

### Runs of record

Every AML Cycle 2 Full that RE-call ran, except the first Multimodal Full, was served by the
variant `C9_routed_specialists_grounded_graph_atomic`, selected with
`RECALL_AML_VARIANT=C9_routed_specialists_grounded_graph_atomic`. Each fixed commit is a signed tag
in this repository.

| Track | Run | Fixed commit | Tag | Compiler |
|---|---|---|---|---|
| Textual | Full 1 | `3eb447c4ea1f04f41f941ad8f089c9a02106e763` | `aml-c2-full1-textual` | on |
| Coding | Full 1 | `1f8666df2b04c34efd33153f0751c624fb17b8f6` | `aml-c2-full1-coding` | on |
| Multimodal | Full 2 | `225e7509eedf130314e1eab2b06a412d14dc1aae` | `aml-c2-full2` | off |

Multimodal Full 2 sets four variables beyond the variant: `RECALL_AML_ROUTE_GATES=data`,
`RECALL_AML_T1_GATE=content`, `RECALL_AML_T1_RENDER=v2`, and `RECALL_AML_COMPILER=0`. Its
`/version` therefore reports `route_gates` as `data`, `multimodal_scope` as `dual`, and
`active_components.compiler` as `false`.

Two builds are outside this table. The first Adds of Textual Full 1 were served by the earlier build
`efb79146` (tag `aml-c2-full1-textual-early`) before `3eb447c4` was deployed. Multimodal Full 1 ran
the earlier variant `C8_routed_specialists_grounded_graph` at `5d82b516` (tag
`aml-c2-full1-multimodal`), with the compiler on.

The variant's fields are `HostedVariant` in `recall_aml/variants.py` at each commit. The three
commits define C9 the same way except where the sections below say otherwise.

### Models

The only generative call is the Add compiler: OpenAI `gpt-4o-mini` through OpenRouter, with the
fixed model identifier `openai/gpt-4o-mini`, temperature zero, and a JSON object response. It was
on for Textual Full 1 and Coding Full 1. It was off for Multimodal Full 2, so that run calls no
generative model during Add or Search. Search makes no generative call in any run: C9 has no query
facet planner and no reranker.

The embedding models are Voyage AI models, one per route:

| Route | Model | Profile | What it searches |
|---|---|---|---|
| code (the default) | `voyage-code-4` | `voyage-code-4-v1` | the tenant's raw windows |
| context | `voyage-context-4` | `voyage-context-4-v1` | a separate tenant holding the same windows and the compiled records |
| image leg | `voyage-multimodal-3.5` | `voyage-multimodal-3.5-v2` | vectors of the image-bearing messages |

The graph sidecar described below embeds its compiled records with `voyage-code-4`.

### Add

The messages of an Add are joined in order and stored as overlapping windows of 160 words with a
stride of 120, in a store isolated per `user_id`. Window text holds message content only, with no
role or timestamp. An image is stored once per tenant as its original bytes, and the text carries
an opaque reference to it.

From the same request, and with no model, the Add also builds atomic views: short spans cut from
its own raw windows (profile `aml-add-micro-v1`), each linked to its parent window and stored in an
isolated tenant. They depend on that request alone, so a Search between two Adds sees every view of
every Add that has returned.

When the compiler is on, the server cuts the Add's messages into evidence anchors with identifiers
it creates. The model returns at most eight records, each citing one to eight of those identifiers,
and the server builds each record's evidence spans from the cited anchors, so every span is an
exact slice of a source message. A record that cites no supplied anchor is rejected. At `3eb447c4`
the model also writes each record's fields; from `1f8666df` (`anchor_compile_output="select"`) it
writes only each record's kind and its anchor identifiers. A failed compile stores no fallback
records (`drop_compiler_fallback`); the Add still stores its raw windows and atomic views. From
`1f8666df`:

- the Add's own anchors are bounded to 150,000 encoded characters, and a longer Add is compiled
  from its first and last anchors fitted to that bound;
- the earlier compiled records sent with the compile are bounded to 145,000 characters;
- a compile that the provider refuses for exhausted credit (HTTP 402) stops the Add with a
  retryable 503, storing nothing.

Compiled records go to a graph sidecar tenant. Each carries `references` relations to the raw
windows its evidence spans fall in, and the server resolves both ends of every relation from its
own chunk identifiers; the model never supplies one. With the compiler off the sidecar stays empty.

### Search

A deterministic router (`route_query` in `recall_aml/specialists.py`) picks one route per query.
A query that carries an image takes the multimodal route. Text with a visual word (image, photo,
screenshot, chart and similar) also takes the multimodal route under the default route gates, and
the code route under `data`. Otherwise text with a code signal takes the code route, text with a
conversational memory signal and no code signal takes the context route, and anything else takes
the code route. The multimodal route searches the code route's store and adds the image leg.

On the chosen route, exact dense retrieval and canonical BM25 (`k1` 1.5, `b` 0.75) each return 100
candidates, and reciprocal rank fusion with constant 60 merges them. Each route fuses only its own
rankings, so scores from different embedding models are never compared. Ties on the code route
follow a fixed window order.

Three stages then adjust that ranking:

- **Atomic rescue**, active and placed after fusion: an atomic view that matches the query can
  lift its parent window, and the first five fused positions never change.
- **Graph promotion**: the compiled records nearest the query can move at most two raw windows
  that they reference and that are already in the ranking, never into the first eight positions.
  A failure serves the raw ranking unchanged.
- **Image leg**: `voyage-multimodal-3.5` hits are fused into the ranking. Under the default route
  gates (Textual and Coding Full 1) it runs on the multimodal route only. Under `data`
  (Multimodal Full 2) it runs on every query that carries an image and on every Search of a
  tenant that holds image vectors, and a tenant without images gets the text ranking unchanged.

The response holds at most `top_k` items in rank order. Compiled records that a later record
supersedes are left out unless the query asks about the past. An image-bearing message is returned
as its ordered parts with its original images, within the response media budget. Every item carries
the time of its source messages in `created_at`, and each text item's content begins with that
date.

From `1f8666df`, a relative time phrase in a returned text item keeps its wording, and the date it
refers to, computed from the item's own date, is inserted after it in brackets. Under the default
gate this happens on every route except the code route. Multimodal Full 2 uses the `content` gate,
which applies it on every route but leaves code-like items and phrases alone, and the `v2`
rendering, which keeps week expressions relative to an anchor date. It is a rendering step only:
nothing stored, embedded, or ranked changes.

### Earlier profiles (history)

The variants in this subsection served no AML Cycle 2 Full. They stay documented because the code
still defines them and because `A4_pack_7000` remains the default when `RECALL_AML_VARIANT` is
unset.

`A4_pack_7000`, the profile this section used to describe, compiles every Add with `gpt-4o-mini`.
Add stores ordered raw messages and up to eight `CodingMemoryRecord` objects per session. Each
generated record must cite at least one exact source span by message ordinal and character
offsets. A mismatched ordinal, boundary, or quote rejects the record. Unsupported entities are
removed, while outcome and validation fields survive only when their exact text occurs in one
source message. Failed compilation degrades to a deterministic technical extract, never to an
unsuccessful Add.

Raw message content is segmented at 4,500 characters. Including the maximum role and timestamp
prefix, every segment remains returnable by the smallest registered 5,000 character evidence arm.

Retrieval uses PostgreSQL lexical retrieval, 100 candidates per retrieval leg, reciprocal rank
fusion, and `voyage:rerank-2.5` over the complete fused pool. A reranker failure serves the
deterministic fused order. Query planning may add at most four evidence seeking facets and cannot
return an answer. Packing removes near duplicates and superseded evidence, preserves session
diversity, rescues raw evidence, and caps the returned pack at 12 items and 7,000 characters by
default. This document previously named `voyage-4` as this profile's embedding model; at
`225e7509` its embedding profile is `voyage-context-4-v1`.

Packing preserves the strongest two thirds of the ranked item budget as a relevance core. It then
prefers previously unseen sessions before returning to repeated sessions. This prevents diversity
from displacing multi-record evidence needed to resolve one task while still broadening the tail.

## Engineering experience representation experiment

These variants are history in the sense of the subsection above: none served an AML Full.

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
seconds and Add p95 below 30 seconds. HTTP 422 is permanent to the caller, so it is reserved for input
that can never be accepted: invalid JSON, a body over 44 MiB, a missing or blank `request_id`,
`user_id` or `session_id`, and a malformed or oversized image. Everything else the service can store
or answer is accepted: unknown fields are ignored, `content: null` and blank messages are dropped,
an unreadable `timestamp` is dropped, `top_k` is clamped to 0 through 100 (`null` means 100), a
query over 20,000 characters keeps its first 10,000 and last 9,999, and a blank query or
`top_k: 0` returns an empty `data` list. A failure inside the service returns a retryable 503; so
does an Add answered pending (`X-Recall-Add-Pending: 1`, see `POST /v1/add`), which is not a failure,
and an Add answered `tenant_busy` (`Retry-After: 60`, see `POST /v1/add`), which stored nothing.
Authentication failures return HTTP 401.

Add compilation makes at most three provider attempts with an eight second timeout per attempt.
The anchored compiler used by C8 and C9 also makes at most three, twelve seconds each, and sends
every evidence anchor of the Add on the first. If that attempt fails and the encoded anchors exceed
300,000 characters, the remaining attempts send the session's first and last anchors that fit the
budget instead, because an over-long prompt is refused by the model's context window every time.
From `1f8666df`, C9 also bounds the Add's own anchors to 150,000 characters before the first
attempt (see [Add](#add)).
Within one Add, independent Voyage embedding requests are sent four at a time
(`RECALL_VOYAGE_PARALLEL_REQUESTS`, 1 to 16; 1 restores the sequential behaviour). The vectors are
the ones a sequential run returns; only the wall time of a large Add changes.
In the variants with query facets (C9 has none), Search facet planning makes one attempt with a
two second timeout, then immediately uses the original query and reports the fallback. The shorter
Search policy prevents an optional planner outage from consuming the five second product latency
gate by itself.

## Data handling

Exact text message content is sent through OpenRouter to the configured OpenAI `gpt-4o-mini`
compiler when the active variant has the compiler on, and to the Voyage embedding service. In C9
and `MM2_dual`, ordered text and image Data URIs are sent to Voyage's multimodal embedding endpoint.
In C9, Search queries are sent to the Voyage embedding service only.
Images above Voyage's 16 million pixel input limit are proportionally resized only for that
transient embedding request. RE-call stores and returns the exact original image bytes. Evidence
is stored in the dedicated PostgreSQL evaluation database. Search candidates are sent to the
configured Voyage reranker only when the active variant enables it, which C9 does not. Application
logs contain only truncated user, request, and query
digests, counts, latency, fallback flags, and error classes. They never contain messages, queries,
model output, API keys, or database URLs.

The internal tenant is `aml_` plus the SHA256 digest of the exact `user_id`. Multimodal media and
vectors use deterministic `_media` and `_mm` tenant suffixes under the same user boundary.
PostgreSQL row level security applies the tenant inside each pooled transaction. The product stores
exact `session_id` only as evidence metadata so a result can identify its source session. The Delete
endpoint erases the primary tenant and both multimodal sidecars without affecting another user.

## Measured multimodal admission result

This section is history: it records how the image leg was chosen. No AML Full ran `MM2_dual`; C9
carries its `voyage-multimodal-3.5` embedding as the image leg.

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

At the time, the retained configuration was selected only for a multimodal deployment:

```bash
RECALL_AML_VARIANT=MM2_dual
```

The global hosted default remains unchanged, so this track-specific choice cannot affect textual
or coding deployments. Recompute the selector and audit with the commands in
[`2026-09-20-aml-multimodal-memeye-brand-v2.md`](preregistrations/2026-09-20-aml-multimodal-memeye-brand-v2.md).

### Routed specialist use and embedding reuse

`C7_routed_specialists` incorporates the retained MM2 behavior as its visual route while keeping
Code4 as the conservative default and Context4 for explicit conversational memory. C8 and C9 keep
this routing, except that under the `data` route gates C9's image leg also runs off the multimodal
route for a tenant that holds images (see [Search](#search)). MM2 does not
enter a coding query's candidate pool. Each route performs rank fusion only inside the selected
specialist path, so heterogeneous cosine scores are never compared or averaged.

Hosted deployments may opt into the shared content addressed cache with:

```bash
RECALL_AML_EMBED_CACHE_PATH=/absolute/path/to/aml-hosted-embeddings.sqlite
```

The key binds the complete embedding profile, vector dimension, encoder purpose, and exact input.
Context4 additionally binds the complete ordered document group and chunk ordinal. Multimodal keys
bind canonical structured input and separate document from query vectors. The cache stores only
derived vectors and cannot make one tenant's rows searchable from another tenant. The cache sits
outside the cross process embedding lock: a hit takes no lock, a passage or document miss takes it
for the provider call, and a query miss does not take it.

## Availability and change control

The evaluated endpoint, image, commit, environment digest, and retrieval configuration are frozen
before submission. The service is monitored through `/health`; it restarts on process failure and
uses a Cloudflare Tunnel for public HTTPS. No feature, model, prompt, candidate width, context
budget, or index generation changes during the 30 day availability commitment. Emergency security
or availability repairs require a new fixed commit, which `/version` reports as `git_commit` (the
product version stays `1.0.0`), and are disclosed to AML.
