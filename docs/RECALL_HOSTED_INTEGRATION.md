# RE-call Hosted 1.0: method and integration

## Product identity

`RE-call Hosted 1.0` is the fixed hosted distribution of the open source RE-call project: the
public RE-call storage and retrieval layers plus the `recall_aml` package in the same repository.
A deployment reports its build as the Git commit in `/version`'s `git_commit`, which the operator
declares through `RECALL_AML_GIT_COMMIT` and the service does not check against its code. The
`version` field stays `1.0.0` on every build, so it does not identify one. The fixed commit of each
AML run is the signed tag listed under [Runs of record](#runs-of-record). To cite RE-call, use
[CITATION.cff](../CITATION.cff).

The service is intended to remain reachable and configuration frozen for at least 30 days after
an AML submission. A release candidate must not be submitted until the contract, isolation,
concurrency, latency, persistence, and recovery gates in the committed preregistration pass.

## Method

RE-call Hosted stores what a user said and returns it as dated evidence. It never writes an
answer. Six design choices define the method.

```mermaid
flowchart LR
  subgraph add["Add"]
    msgs["Messages"] -- "no image" --> win["Word windows<br/>160 words, stride 120"]
    win --> views["Atomic views<br/>no model"]
    msgs -. "compiler on, no image" .-> rec["Records citing<br/>server anchors"]
    rec --> grf["Graph sidecar"]
    msgs --> img["Original images<br/>voyage-multimodal-3.5"]
  end
  subgraph search["Search"]
    q["Query"] --> route{"Router"}
    route --> code["Code route<br/>voyage-code-4 + BM25"]
    route --> ctx["Context route<br/>voyage-context-4 + BM25"]
    code --> fuse["Rank fusion<br/>within the route"]
    ctx --> fuse
    fuse --> lift["Atomic rescue<br/>top five fixed"]
    lift -. "Full 2" .-> rr["Voyage rerank-3"]
    lift --> promote["Graph promotion<br/>at most two"]
    rr --> promote
    promote --> leg["Image leg"]
    leg --> out["Dated evidence<br/>at most top_k"]
  end
  win -.-> code
  views -.-> lift
  grf -.-> promote
  img -.-> leg
```

1. Compiled records hold only source text the server verified: the model cites anchor identifiers
   the server created, and the server builds every quote from them (see [Add](#add)).
2. A deterministic router ranks each query in one embedding space, so the ranking never mixes
   scores from different models, and the routing itself uses no model.
3. Every item carries the date a reader needs, including the date a relative phrase refers to.
4. Each Add without an image builds its own atomic views, so a Search between two Adds already
   uses them, and the rescue cannot change the fused top five. In the Full 2 runs the reranker that follows may.
5. The model cannot draw a graph edge: edges come only from verified evidence spans, and
   promotion can only reorder windows that retrieval already found. Each of these rules, and the
   raw ranking served on failure, has a mutation test that fails when the rule is broken
   ([receipts](results/aml-graph-v1/RED_PROOF_RECEIPTS.md)).
6. The image leg searches wherever the user's memory holds images, not only when a question names
   a visual word, and leaves the ranking of a user without images unchanged.

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

The service returns HTTP 200 only after the Add's evidence, and its compiled records when the
compiler is on, have been persisted and made searchable. An identical
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
evidence, not a generated answer. The response holds at most `top_k` items, and one that carries
images is also bounded by the decoded response media budget.

### Multimodal content

C9 accepts the AML ordered content shape in both Add messages and the Search query:

```json
[
  {"type": "text", "text": "The deployment panel shows"},
  {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
  {"type": "text", "text": "after the repair"}
]
```

Only inline Base64 JPEG, PNG, and WebP images are accepted. Each decoded image is limited to 10
MiB. One Add request and one Search response are each limited to 30 MiB of decoded images. C9
stores each original image once per tenant under its SHA256, reconstructs the exact ordered parts
at Search, and embeds every message of an Add that carries an image with `voyage-multimodal-3.5` in
a separate vector tenant. The main text chunks contain only source text and opaque media references, never Base64.

AML's rule for the Open-source Methods division is that every generative model used during Add or
Search is `gpt-4o-mini`. The Voyage models RE-call uses are embedding models, and each one is named
in the method disclosure of the run that used it.

### `GET /health`

Returns HTTP 200 only after the serving database schema and active generation passed readiness and
startup completed one bounded live request against every model provider stage used by the active
variant. Dependency failures prevent startup or return HTTP 503.

### `GET /version`

Returns the product name and version, `git_commit`, the schema version, the `variant` and its
profiles, `active_components`, the generation model, the prompt digests, the Add response budget
in seconds (`add_response_budget_seconds`, 0 when off), and the tenant queue budget in seconds
(`tenant_lock_wait_seconds`, 0 when off). It contains no credential, database URL, host inventory,
or other infrastructure identifier.

`active_components` is the field that shows which stages run. `retrieval_profile` is a fixed label
present in every build. `reranker` names the reranker model the service loaded (`voyage:rerank-3` at
`625b7138`), and only `active_components.reranker` says whether it runs.

### `POST /v1/delete`

Accepts `{ "user_id": "exact AML user id" }`. It atomically deletes that user's hosted chunks,
learned sparse sidecars if present, and Add replay receipts. Other user tenants are not touched.

## Model and retrieval configuration

### Forget suppression controls

`RECALL_AML_FORGET_GATE` overrides the variant's gate for forget suppression. Its accepted values
are `route` and `ledger`; all current variants default to `route`. An unknown value stops startup.
With `route`, the code route is exempt from suppression. With `ledger`, suppression can act on
every route, using only that user's forget ledger; an empty ledger changes no results. On the
code route, `drop` and `stub` preserve code-shaped items whole and annotate them instead of
dropping or cutting them. Other items still follow the selected suppression mode.

The gate has no effect while `RECALL_AML_FORGET=off`, the current default. To populate the ledger,
select `drop`, `stub`, or `annotate` before the first Add; Adds made while forget is off do not
record forget requests. Combining active forget suppression with `RECALL_AML_ROUTE_GATES=data`
requires `RECALL_AML_FORGET_GATE=ledger`, otherwise startup refuses the configuration.

`GET /version` reports `forget_suppression.mode` and `forget_suppression.code_route`: the latter is
`exempt` with the route gate, `ledger-gated` with an active ledger gate, and `off` with a ledger
gate while forget is off.

### Runs of record

Every AML Cycle 2 Full that RE-call ran, except the first Multimodal Full, was served by the
variant `C9_routed_specialists_grounded_graph_atomic`, and so are both second Fulls, selected with
`RECALL_AML_VARIANT=C9_routed_specialists_grounded_graph_atomic`. Each fixed commit is a signed tag
in this repository.

| Track | Run | Fixed commit | Tag | Compiler |
|---|---|---|---|---|
| Textual | Full 1 | `3eb447c4ea1f04f41f941ad8f089c9a02106e763` | `aml-c2-full1-textual` | on |
| Coding | Full 1 | `1f8666df2b04c34efd33153f0751c624fb17b8f6` | `aml-c2-full1-coding` | on |
| Multimodal | Full 2 | `625b71381c2f5c7de6d34c8dc5b9f45e08370b58` | `aml-c2-full2-625b7138` | off |
| Textual | Full 2 | `625b71381c2f5c7de6d34c8dc5b9f45e08370b58` | `aml-c2-full2-625b7138` | off |

Both Full 2 runs also set `RECALL_AML_ROUTE_GATES=data`, `RECALL_AML_T1_GATE=content`,
`RECALL_AML_T1_RENDER=v2`, `RECALL_AML_COMPILER=0`, `RECALL_AML_RERANKER=1`,
`RECALL_AML_RERANK_MODEL=rerank-3` and `RECALL_AML_FEEDBACK_GUARD=1`, so `/version` reports
`route_gates` `data`, `multimodal_scope` `dual`, `reranker` `voyage:rerank-3`, and
`active_components.compiler` `false` and `active_components.reranker` `true`. The tag
`aml-c2-full2` on `225e7509` marks an earlier plan for Multimodal Full 2; no run uses it. The tag
`aml-c2-full2-b712ec33` marks the build served from 2026-10-20 until `625b7138` replaced it on
2026-10-21; both Full 2 runs use `625b7138`.

Two builds are outside this table. The first Adds of Textual Full 1 were served by the earlier
build `efb79146` (tag `aml-c2-full1-textual-early`) before `3eb447c4` was deployed. Multimodal
Full 1 ran the earlier variant `C8_routed_specialists_grounded_graph` at `5d82b516` (tag
`aml-c2-full1-multimodal`), with the compiler on.

C9's fields are `HostedVariant` in `recall_aml/variants.py`. The three commits define it the same
way except where the sections below name a commit. `625b7138` is `225e7509` plus two switches, both
off unless set, the reranker switch (`recall_aml/rerank_switch.py`) and the feedback rule
(`recall_aml/feedback_guard.py`), and the fixes of a code audit: a compiler switched off is neither
required nor probed at start, an image-only query is not reranked, the reranker has a 15 second
timeout, every Search has one time budget (`recall_aml/search_deadline.py`), and an unreadable image
is dropped from an Add rather than failing it.

### License

Every commit named above carries the PolyForm Noncommercial License 1.0.0 in its tree: each is on,
or branched from, the history before the project moved to AGPL-3.0-only.
[`LICENSE_GRANT_AGPL.md`](../LICENSE_GRANT_AGPL.md),
dated 5 October 2026, additionally licenses each of them under AGPL-3.0-only and names each one by
its full commit hash.

### Models

The only generative model is OpenAI `gpt-4o-mini` through OpenRouter, model identifier
`openai/gpt-4o-mini` (from `1f8666df` the default of `RECALL_AML_GENERATION_MODEL`, which no run of
record sets), temperature zero, and a JSON object response. It is called by the
Add compiler and, when the compiler is on, by one readiness check at service start that compiles a
fixed test sentence (`recall_aml/readiness.py`) and never sees evaluation data. In both Full 2 runs
the compiler is off, so no generative model runs during Add, Search or service start. Search never
calls one. Both Full 2 runs also reorder candidates with the Voyage reranker `rerank-3`.

The embedding models are Voyage AI models, one per route:

| Route | Model | Profile | What it searches |
|---|---|---|---|
| code (the default) | `voyage-code-4` | `voyage-code-4-v1` | the tenant's raw windows |
| context | `voyage-context-4` | `voyage-context-4-v1` | a separate tenant holding the same windows and the compiled records |
| image leg | `voyage-multimodal-3.5` | `voyage-multimodal-3.5-v2` | vectors of every message of an Add that carries an image |

The graph sidecar described below embeds its compiled records with `voyage-code-4`.

### Add

The messages of an Add are joined in order and stored as overlapping windows of 160 words with a
stride of 120, in a store isolated per `user_id`. Window text holds message content only, with no
role or timestamp.

An Add that carries an image in any of its messages is stored differently, the whole Add at once:
each message as segments of up to 4,500 characters, each prefixed with the message's role and,
when it has one, its timestamp, beside the media and the multimodal vectors. Such an Add builds no word windows, no atomic
views, no compiled records and no graph records, whether the compiler is on or not.

The Add also builds atomic views from its own raw windows, with no model: short spans (profile
`aml-add-micro-v1`), each linked to its parent window and stored in an isolated tenant.

In the Full 2 runs, a record of an earlier answer that reports a failure, about a question this user
was already asked, is not stored. Two record shapes are recognised, the shapes the public
AgentMemoryBench protocol (solomoon313/AgentMemoryBench, MIT) writes. The first is one message whose
text is a JSON object with a question, the model's answer and its outcome; that message alone is not
stored. The second is a judge unit: the user turn that ends with `Question: ...` and `Answer:`
(memories may come before them), the model's answer, and a judge message whose verdict is WRONG.
Then the whole unit is not stored, from the nearest question turn before the judge message to the
judge message, together with a system message directly before that turn; the rest of the Add is
stored. When the judge messages of one Add disagree, none of them acts. A failed record that
also carries the reference answer is kept, and so is a judge unit that holds one. Every message
outside such a record or unit is stored unchanged. Which questions this user was asked is kept in the
service process only: a restart or a Delete empties it, after which such records are stored.

When the compiler is on, the server cuts the Add's messages into evidence anchors with identifiers
it creates. The model returns at most eight records, each citing one to eight of those identifiers,
and the server builds each record's evidence spans from the cited anchors, so every span is an
exact slice of a source message. A failed compile stores no records. At `3eb447c4` the model also
writes each record's fields, and a record whose cited identifiers are all unknown is kept when its
field values occur in anchors' text, every such anchor (up to eight) becoming its evidence; from `1f8666df`
(`anchor_compile_output="select"`) the model writes only the kind and the anchor identifiers, and a
record that cites no supplied anchor is rejected.
Also from `1f8666df`, the Add's own anchors are bounded to 150,000 encoded characters (a longer Add
is compiled from its first and last anchors), the earlier records sent with it to 145,000, and a
compile refused for exhausted provider credit (HTTP 402) stops the Add with a retryable 503,
storing nothing.

Compiled records go to a graph sidecar tenant. Each carries `references` relations to the raw
windows its evidence spans fall in, and the server resolves both ends of every relation from its
own chunk identifiers. With the compiler off the sidecar stays empty.

### Search

A deterministic router (`route_query` in `recall_aml/specialists.py`) picks one route per query.
A query that carries an image takes the multimodal route, which searches the code route's store
and adds the image leg. Text with a visual word (image, photo, screenshot, chart and similar) also
takes the multimodal route; from `225e7509` it takes the code route instead when the route gates are
`data`.
Otherwise text with a code signal takes the code route, text with a conversational memory signal
and no code signal takes the context route, and anything else takes the code route.

On the chosen route, exact dense retrieval and canonical BM25 (`k1` 1.5, `b` 0.75) each return 100
candidates, and reciprocal rank fusion with constant 60 merges them. Ties on the code route follow
a fixed window order.

After fusion, atomic rescue can lift the parent window of an atomic view that matches the query,
never into the first five positions. In the Full 2 runs, Voyage `rerank-3` then reorders the whole
candidate pool, except for a query that is only an image, which keeps its fused order; a reranker
failure or timeout serves the fused order. Graph promotion then lets the compiled records nearest the
query move at most two raw windows they reference, which must already be in the ranking, never
into the first eight positions; if it fails, the raw ranking is served. The image leg fuses
`voyage-multimodal-3.5` hits into the ranking: under the `route` gates, which are the only behaviour
at `3eb447c4` and `1f8666df` because those commits predate the setting (Textual and Coding Full 1),
on the multimodal route only, and under `data` (both Full 2 runs) on every Search of a
tenant that holds image vectors. A tenant without images gets the text ranking unchanged.

Compiled records that a later record supersedes are left out unless the query asks about the
past. An image-bearing message is returned as its ordered parts with its original images; from
`625b7138`, an image that only the image leg found is returned as its image, not as placeholder
text. Every
item whose source messages carry a readable timestamp carries that time in `created_at`, and each
such text item's content begins with that date; an item from untimed messages has `created_at` null
and no date prefix.

From `1f8666df`, a relative time phrase in a returned text item keeps its wording, and the date it
refers to, computed from the item's own date, follows it in brackets. Under the default gate this
skips the code route. Both Full 2 runs use the `content` gate, which covers every route but leaves
code-like items and phrases alone, and the `v2` rendering, which keeps week expressions relative to
an anchor date. Nothing stored, embedded, or ranked changes.

### Variants that served no AML run

With `RECALL_AML_VARIANT` unset the service runs `A4_pack_7000` (`DEFAULT_VARIANT` in
`recall_aml/variants.py`): `voyage-context-4` embeddings, query facets, `voyage:rerank-2.5` over the fused pool, a
pack capped at 12 items and 7,000 characters, and deterministic fallback records when a compile
fails. No AML Cycle 2 run used it. It and the other experimental variants are defined in
`recall_aml/variants.py`.

## Capacity, timeouts, and rate limits

The declared service capacity is 16 concurrent Add requests and 16 concurrent Search requests in
one process. The database pool is sized above the combined running capacity. Add provider and
database work must complete before the caller's 45 second integration timeout. A Search request is
cut after about 100 seconds in front of the service, while the release gate requires a measured
Search p95 below 5 seconds and Add p95 below 30 seconds. HTTP 422 is permanent to the caller, so it is reserved for input
that can never be accepted: invalid JSON, a body over 44 MiB, a missing or blank `request_id`,
`user_id` or `session_id`, a malformed or oversized image, and, from `625b7138` (both Full 2 runs),
an Add in which no part can be read and a Search query image that cannot be decoded. From the same
build, an image that passes admission but cannot be decoded is dropped from an Add that holds
anything else readable; the rest of the Add is stored. Everything else
the service can store or answer is accepted: unknown fields are ignored, `content: null` and blank
messages are dropped,
an unreadable `timestamp` is dropped, `top_k` is clamped to 0 through 100 (`null` means 100), a
query over 20,000 characters keeps its first 10,000 and last 9,999, and a blank query or
`top_k: 0` returns an empty `data` list. A failure inside the service returns a retryable 503; so
does an Add answered pending (`X-Recall-Add-Pending: 1`, see `POST /v1/add`), which is not a failure,
and an Add answered `tenant_busy` (`Retry-After: 60`, see `POST /v1/add`), which stored nothing. From
`625b7138` (both Full 2 runs), a Search whose query embedding does not answer within the service's
Search time budget (90 seconds)
returns 503 `{"error": "search_deadline"}` with `Retry-After: 5`; it is safe to resend. Within that
budget a rerank or image search that runs out of time is skipped, and a query with text is served
its text ranking.
Authentication failures return HTTP 401.

Compiler attempts depend on the build, each attempt twelve seconds. C8 at `5d82b516` makes at most
three, each with the same payload. C9 at `3eb447c4` makes at most three and sends every evidence
anchor of the Add on the first; if that attempt fails and the encoded anchors exceed 300,000
characters, the remaining attempts send the session's first and last anchors that fit the budget
instead, because an over-long prompt is refused by the model's context window every time. From
`1f8666df` the anchors and the earlier records are fitted to the bounds given under Add before the
first attempt, so that refit never runs. Oversized Adds are fitted from their first and last
anchors; if fitting retains no anchor, compilation is skipped and the normal compiler fallback
handles the Add. A refusal (HTTP 400 or 413) of a payload that still carries earlier records is
resent once without them, which can make a fourth attempt. From `1f8666df`
(`compile_resend_truncated=False`) an answer cut off at the token limit is not resent: the compile
fails at the first such answer; at `3eb447c4` it is retried like any other failed attempt.
Within one Add, independent Voyage embedding requests are sent four at a time
(`RECALL_VOYAGE_PARALLEL_REQUESTS`, 1 to 16; 1 restores the sequential behaviour). The vectors are
the ones a sequential run returns; only the wall time of a large Add changes.

## Data handling

Exact text message content is sent through OpenRouter to `gpt-4o-mini` when the compiler is on,
and to the Voyage embedding service. Ordered text and image Data URIs are sent to Voyage's
multimodal embedding endpoint. Images above Voyage's 16 million pixel input limit are
proportionally resized only for that transient embedding request. RE-call stores and returns the
exact original image bytes. Search queries are sent to Voyage only, and in the Full 2 runs the
candidate text that `rerank-3` reorders is sent to Voyage as well. Evidence is stored in the
dedicated PostgreSQL evaluation database. Application logs contain only truncated user, request,
and query digests, counts, latency, fallback flags, and error classes. They never contain messages,
queries, model output, API keys, or database URLs.

The internal tenant is `aml_` plus the SHA256 digest of the exact `user_id`. Multimodal media and
vectors use deterministic `_media` and `_mm` tenant suffixes under the same user boundary.
PostgreSQL row level security applies the tenant inside each pooled transaction. The product stores
exact `session_id` only as evidence metadata so a result can identify its source session. The Delete
endpoint erases the primary tenant and both multimodal sidecars without affecting another user.

## Embedding cache

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
or availability repairs require a new fixed commit and are disclosed to AML.
