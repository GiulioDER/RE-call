# MemoraX and RE-call: Agent Memory Leaderboard coding audit

Audit date: 2026-09-11

MemoraX source inspected: `memorax-ai/memorax-code` at commit
`4b7fdcc8d413db9656f0c52fd3a1432043429f66`, package version `0.1.16`.

RE-call released baseline inspected: `origin/master` at commit
`f451faa30a2871c0da63ab2e3be7bf952093dcda`, package version `0.13.0`.

RE-call development delta inspected: the local `codex/performance-attribution-vps2` branch at
`7805d05796fbcffa13807cdd11922ded8470603f`, 59 commits ahead of the released baseline, plus the
current uncommitted graph answer validation work. No existing user change was modified for this
audit.

## Executive conclusion

RE-call reaching the Academic Methods top three is plausible, but it has not been demonstrated by
the existing benchmark record. The current coding board does not place MemoraX and RE-call in the
same competitive division. MemoraX is first on the Industry board at 62.00 percent. The Academic
board has eight methods tied at 52.67 percent. RE-call is open source and belongs on the Academic
board unless it is deliberately submitted as a commercial product.[^1]

The current Academic lead corresponds to 79 resolved tasks out of 150. A result of 80 out of 150,
or 53.33 percent, would exceed every current Academic entry if the second cycle keeps the same 150
task suite and scoring rule. MemoraX resolved 93 out of 150. Its category scores imply 36 of 51 new
feature tasks and 57 of 99 bug fix tasks. These task counts are inferred exactly from the published
score fractions and the official 150 task total. They are not separately disclosed by AML.[^2]

The principal finding is not that MemoraX has a visibly superior retrieval algorithm. Its winning
memory algorithm is not present in `memorax-code`. The repository is a carefully engineered client
integration layer around the proprietary `platform.memorax.net` memory service. It exposes enough
of the contract to prove hybrid retrieval controls, typed memory results, temporal rendering,
bounded context, lifecycle capture, scope isolation, redaction, and retries. It does not expose the
server side embedding model, memory extraction prompts, storage engine, deduplication, conflict
resolution, fusion equation, or final ranking policy.[^3]

The strongest public signal from MemoraX is therefore behavioral. It combines the highest task
resolution score with only 6,312 average returned characters and the lowest input token total among
the leading rows. That pattern is consistent with precise memory extraction and compact evidence
delivery. It does not prove that brevity alone causes the score. MemOS returns only 3,885 characters
and scores 52.00 percent, while LightMem returns about 325 characters and scores 51.33 percent.
MemoraX is distinctive because compactness and task relevance appear together.[^2]

RE-call already has the hard retrieval components needed for a strong entry: PostgreSQL plus
pgvector, dense and lexical hybrid retrieval with reciprocal rank fusion, optional learned sparse
retrieval, optional reranking, exact tenant isolation, temporal validity, supersession, durable
idempotency receipts, evidence selection, and unusually rigorous measurement discipline.[^4]
Its best public Bench'd result is encouraging but not directly transferable. The winning Bench'd
adapter used DeepSeek v4 pro at Search time to synthesize a short evidence digest. AML requires
Add/Search model use to be `gpt-4o-mini`, and Search may return only memory evidence rather than a
generated final answer. The Bench'd configuration therefore cannot be submitted unchanged.[^5]

The deadline strategy should concentrate on three seams:

1. Build a fully conformant synchronous AML adapter with exact `user_id` isolation, idempotent Add,
   stable session grouping, and the required Search response.

2. Compile raw historical engineering sessions at Add time into durable, evidence grounded coding
   memories. Preserve raw evidence alongside the compiled records. Use `gpt-4o-mini` if a model is
   used, subject to written confirmation from AML about whether the model restriction includes
   embedding and reranking models.

3. Retrieve from a wide candidate pool, then return a small, diverse, task shaped evidence pack.
   The pack should favor symptoms, root causes, paths and symbols, successful changes, failed
   attempts, constraints, and validation results. It should not generate an answer at Search time.

Do not make broad graph work the critical path for this admission. The current graph branch has
produced a real but bounded signal: the best LoCoMo structural arm improved complete evidence
coverage and answer citation coverage, while the newest authored `depends_on` mechanism rescued a
small synthetic set. It has not yet established coding task resolution improvement, and the served
corpus recorded on 2026-09-11 still lacked real typed dependency relations. Keep that work as a
controlled optional arm after the adapter, ingestion compiler, and evidence packer are working.

## 1. What the leaderboard actually measures

AML has two independent dimensions: evaluation track and participant division. The coding track
measures reuse of historical debugging experience, development experience, and repository context.
The public description reports 12 repositories, 150 base software engineering tasks, and 1,290
time constrained historical tasks with fine grained relevance annotations. The coding data,
protected annotations, orchestration, participant traces, and verifier materials are not public.[^6]

The participant controls only Add and Search. AML controls the answer model, answer prompt,
evaluator, Top K, dataset bundle, and aggregation. This is crucial. A good general purpose memory
system can lose if its returned evidence is hard for the fixed coding agent to use. Conversely, a
memory system can win without being the safest production memory layer if it reliably supplies the
specific experience needed to complete each task.[^7]

The current contract has several consequences for RE-call:

1. Add is synchronous. HTTP 200 may be returned only after every submitted message is persisted and
   immediately searchable. An asynchronous acceptance receipt is insufficient.

2. Add receives only `request_id`, ordered `messages`, `user_id`, and `session_id`. AML does not send
   arbitrary metadata, application identifiers, agent identifiers, or an asynchronous mode flag.

3. Large source sessions are split when they exceed 20 messages or 2,000 words. The implementation
   must preserve source session continuity across multiple Add calls.

4. Search receives the original query, optional choice options, the exact `user_id`, and
   `top_k=100` in formal evaluation. It does not receive filter, rerank, or keyword search flags.

5. Search output is a relevance ordered `data` array. Each item needs a stable nonempty `id` and
   nonempty `content`. `score` and `created_at` are optional. Undeclared metadata is ignored. The
   content is passed directly to the fixed answer model.

6. Returning fewer than 100 items is allowed. AML reads at most Top K and preserves the submitted
   order. This makes internal wide retrieval followed by narrow evidence packing admissible.

7. The formal Full mode may be used only once every three months, and an accepted Full submission
   cannot be replaced or withdrawn merely because its score is poor. Smoke is a compatibility test,
   not a quality test.[^8]

8. The second cycle is expected to open on September 20, 2026. The site describes that date as an
   expected opening, not as a submission deadline. Final timing and requirements remain subject to
   the official notice.[^1]

## 2. Current score landscape

The public coding data contained 58 entries on 2026-09-11: 43 Academic and 15 Industry. The
Academic median was 50.67 percent and the Industry median was 51.33 percent. The board is tightly
clustered from roughly 47 through 53 percent, except for MemoraX at 62 percent.[^2]

The relevant thresholds are:

1. Academic current lead: eight methods tied at 79 of 150, or 52.67 percent.

2. Academic score that clears the tie: 80 of 150, or 53.33 percent.

3. Industry current second place: MemOS at 78 of 150, or 52.00 percent.

4. Industry current third place: `hs v3` and `claude-mem` are also at 78 of 150. A clear Industry
   top three result would therefore require at least 79 of 150 if the board and tie handling remain
   unchanged.

5. MemoraX: 93 of 150, or 62.00 percent. Beating it requires at least 94 of 150, or 62.67 percent.

The immediate strategic target is not 62.67 percent. It is at least 53.33 percent in Academic,
with a safety target of 55 percent or higher because the second cycle may improve the field or
change the suite. One extra resolved task would clear the current Academic lead, but one task is too
small a margin to plan around.

## 3. What MemoraX publicly reveals

### 3.1 Architecture boundary

`memorax-code` integrates six coding clients into one local TypeScript backend. Native clients own
their models, tools, and transcripts. The backend owns client event parsing, turn correlation,
repository scope, buffering, chunking, redaction, retry behavior, local trace, and calls to the
remote MemoraX provider.[^3]

The normal retrieval path is explicit. Automatic retrieval at turn start is disabled by default.
The shared skill decides whether persistent memory is relevant, then invokes
`memorax-cli search`. That command resolves repository scope, calls the remote MemoraX Search API,
normalizes the result, and presents it to the coding client. Automatic writeback is a separate
lifecycle path.[^9]

This design is strong operationally. It keeps model ownership in the client, centralizes memory
semantics, validates exact session and repository identity, and does not guess transcript formats
across clients. None of those features reveal the server side memory algorithm.

### 3.2 Add behavior

The public provider adapter sends messages with role, content, timestamp, scoped `user_id`, language,
optional content type, chunk information, session identifier, bounded metadata, and
`async_mode:true` to `/v1/memories/add`. The source comment explicitly says that acceptance means
task submission, not completed memory extraction. The architecture says the backend records the
initial response and does not poll asynchronous Add status.[^10]

Automatic writeback has careful preprocessing. Defaults include an eight turn buffer, ten minute
age limit, 128,000 character buffer cap, 64,000 characters per message, 8,000 character chunks, and
five percent overlap. User and assistant text are bounded and redacted before hashing, buffering,
chunking, retries, or network dispatch. Timestamps and their source labels are preserved.[^11]

This public Add path is not AML compliant as written because AML requires synchronous persistence
and immediate searchability before HTTP 200. MemoraX's leaderboard entry is a submitted hosted API,
so it can use a different wrapper or server behavior. The public repository cannot establish how
the leaderboard's Add endpoint met that contract.

### 3.3 Search behavior

The provider request includes `query`, scoped `user_id`, `top_k`, `k_dense`, `k_sparse`, optional
filters, and an optional minimum semantic similarity. It calls `/v1/memories/search`. This proves
that the remote service accepts separate dense and sparse candidate widths and can disable sparse
retrieval by setting its width to zero. It does not prove which embedding, lexical algorithm,
fusion rule, or reranker is used.[^12]

Returned items can expose memory text through `memory`, `summary`, `content`, or `text`. The adapter
reads `metadata.memory_type` and recognizes typed buckets. Default output order is `core`,
`episodic`, `semantic`, `procedural`, then `unclassified`. Timestamps come from `updated_at` or
`created_at`. Each item is capped at 1,000 characters and total rendered context at 4,000
characters in the ordinary product configuration. The result is emitted as a small XML memory
block.[^13]

The typed result surface is important. A coding task benefits from different evidence roles. A
procedure, an episode describing a failed attempt, and a durable architectural fact should not be
treated as interchangeable chunks. MemoraX gives its client enough type information to order these
roles deliberately.

### 3.4 Scope and local repository memory

MemoraX derives an effective identity from a base user and repository or folder. Default chat
locations use a General namespace. Conflicting or unprovable scope fails closed. This is a robust
implementation of the same isolation property AML demands from exact `user_id` matching.[^9]

The `.repo_memory` subsystem is separate from remote provider results. It builds repository local
guidance from local structure, commits, and optional GitHub or GitLab facets. It should not be
credited as part of the remote Search algorithm without evidence that the leaderboard endpoint
used it. The leaderboard row contains no linked GitHub repository or run trace.

### 3.5 What remains unknowable from public sources

The following are not implemented or specified in the inspected repository:

1. Embedding model and vector database.

2. Lexical algorithm and tokenization.

3. Dense and sparse fusion equation.

4. Reranking model and policy.

5. Add time memory extraction prompt and output schema.

6. Semantic deduplication, consolidation, and forgetting.

7. Conflict resolution and temporal supersession rules inside the hosted memory service.

8. The exact MemoraX configuration used for AML.

9. Retrieved memories, task traces, and per task outcomes from the winning run.

10. A reproducible mapping from leaderboard service version `v0.5` to the public package version
    `0.1.16`.

The repository's test fixture includes a mocked result with `rank_method: "query_rrf"`. That is a
useful API compatibility clue, not proof of the production ranking algorithm. It would be an error
to reverse engineer a full server design from a mock response.

## 4. What RE-call currently provides

### 4.1 Storage and ingestion

Released RE-call is primarily a document and memo memory layer. It reads supported files, extracts
text and structure, chunks prose at a default target of 800 characters with 80 characters of
overlap, embeds batches, and writes them to a tenant scoped PostgreSQL and pgvector store. Source
replacement, pruning, immutable generations, migrations, and content fingerprints are mature.[^14]

RE-call does have model backed truth extraction, but it is designed to identify checkable claims
such as supersession, validity, status, and identity from authored memos. It is not a general coding
session compiler. The project README accurately says RE-call is not an automatic truth extractor.
No current endpoint accepts AML's ordered message array and turns it into debugging or development
memories.[^4]

### 4.2 Retrieval

`HybridRetriever` embeds the query, retrieves dense candidates from pgvector, retrieves lexical
candidates from PostgreSQL full text search, optionally retrieves learned sparse candidates, and
combines their ranked identifiers with reciprocal rank fusion. The default candidate pool is 20
per leg. The full fused pool is reranked before it is truncated to the requested result count.[^15]

This is a good base for AML. For the submission profile, candidate width should be independent of
returned memory count. A candidate pool near 100 per active leg can maximize discovery, while a
separate evidence packer returns only the few items that improve task completion. The existing
Bench'd adapter already applies this separation at a smaller output depth.

The optional fused query path retrieves both the current query and conversation history, combines
the two result sets with nested reciprocal rank fusion, and reranks once. It measured a small but
significant nDCG improvement and a larger Recall at 100 improvement on MTRAG human development data
only when reranking was enabled. This mechanism is not directly useful in AML Search unless Search
has meaningful query history. The Add histories are the corpus, not Search history.[^16]

### 4.3 Trust, temporal logic, and evidence

RE-call's differentiator is its trust layer. Every candidate can carry provenance, validity window,
calibrated confidence, supersession status, authority, dependencies, and an explicit verdict. The
trust layer can promote a retrieved successor over stale evidence and abstain when nothing is safe
to use. The evidence builder admits only `ok` hits and supports fixed item counts, exact token
budgets, document grouping, answer slots, and beam selection.[^17]

This is stronger than anything publicly verifiable in MemoraX. It is also potentially misaligned
with AML coding evaluation. Historical coding tasks are constructed to provide useful prior
experience. A strict confidence threshold that returns an empty array can turn an imperfect but
useful match into a guaranteed loss. RE-call's Bench'd run measured a 26.7 point penalty from its
default abstention behavior on an answerable by construction suite. The AML coding suite is not
documented as entirely answerable, so the exact optimal threshold is unknown, but an aggressive
abstention default should not be assumed.[^5]

For the submission profile, keep hard temporal invalidity, exact isolation, and explicit
supersession. Treat confidence abstention as an experimental knob. Compare a zero threshold against
a calibrated low threshold on a representative proxy. This is a benchmark profile choice, not a
recommendation to weaken production safety.

### 4.4 Idempotency and concurrency

RE-call already has durable idempotency receipts keyed by tenant and request key. A receipt records
the operation, request fingerprint, result, and expiration, and rejects conflicting reuse. This is
directly useful because AML retries transient Add failures and requires exact identifiers to be
echoed. The PostgreSQL store also supports a connection pool for concurrent server use and database
statement timeouts.[^18]

These capabilities make the adapter implementation lower risk than starting from a research
retriever. The missing work is wiring and workload qualification, not inventing transaction
semantics.

### 4.5 Evidence output weakness for AML

RE-call's default evidence path preserves retrieval order, performs no semantic deduplication, and
does no neighbor retrieval. The MCP surface normally returns five items and does not set a token
budget. That is defensible for citable research evidence, but it is not yet a task optimized coding
memory pack.[^19]

The Bench'd adapter solved the conversion problem by asking DeepSeek v4 pro to compress retrieved
chunks into two declarative sentences. It achieved 69.0 on LongMemEval 500 and 71.6 on LoCoMo 1,540
under that harness's own reported metric. The result shows that RE-call can retrieve useful evidence
and that output conversion matters. It does not validate AML compliance or AML coding task success.
For AML, conversion should happen when memories are created, or through deterministic selection and
formatting at Search time, not through a prohibited answer generating Search stage.[^5]

### 4.6 Current graph work

The development branch has moved semantic graph candidates earlier in the selection pipeline,
added score before truncation, typed `depends_on` projection, calibrated candidate ranking, and a
controlled one item tail replacement. Current measurements show:

1. A full LoCoMo structural sweep improved complete evidence coverage from 64.39 percent to 66.67
   percent in its best 8 direct plus 2 graph arm. A repeat remained positive but missed the
   preregistered two point promotion gate.

2. The answer citation replay improved complete gold citation coverage from 49.54 percent to 50.85
   percent for that arm. It did not measure factual answer correctness.

3. The authored `depends_on` mechanism rescued all eight synthetic dependent queries without
   changing eight controls. This establishes mechanism capability, not production quality.

4. The live semantic projection recorded 6,524 entities and 4,377 relations, but every relation was
   `references`. The typed relations needed for causal graph value were absent from the served
   corpus at that measurement time. The record explicitly requires remeasurement before reuse.

The correct interpretation is promising but unpromoted. Graph tail competition may become a useful
coding memory feature when the Add compiler authors genuine relationships. It should not displace
the compiler and evidence packer work that makes those relationships exist.

## 5. Direct comparison

### 5.1 Ingestion quality

MemoraX: Public client code captures validated user and assistant turns, preserves timestamps,
redacts secrets, groups and chunks content, and delegates memory creation to a remote service. The
actual extraction quality is unknown.

RE-call: File ingestion, structure extraction, generation management, and provenance are stronger
and fully inspectable. Raw coding session ingestion and reusable experience extraction are missing.

Assessment: MemoraX has the current product fit advantage. This is the largest RE-call gap for AML.

### 5.2 Retrieval quality

MemoraX: Separate dense and sparse widths are proven. Exact algorithms and ranking are unknown.

RE-call: Dense, lexical, optional learned sparse, reciprocal rank fusion, reranking, scope priors,
and measured candidate widening are implemented and open.

Assessment: RE-call is not visibly behind at the component level. The missing proof is end to end
coding task resolution under the AML contract.

### 5.3 Memory representation

MemoraX: Search results expose core, episodic, semantic, procedural, and unclassified memory types.
This is a strong fit for coding experience reuse.

RE-call: Chunks carry rich provenance and graph metadata, but ordinary indexed content remains
document shaped. It lacks a first class coding episode schema.

Assessment: MemoraX has a practical representation advantage. RE-call should add an adapter local
coding memory schema rather than redesign its core storage.

### 5.4 Temporal and conflict handling

MemoraX: The client preserves source timestamps and displays updated or created times. Server side
conflict behavior is unknown.

RE-call: Validity windows, known as of logic, supersession, successor promotion, dependency
invalidation, immutable evidence cards, and append only fact history are explicit.

Assessment: RE-call has the stronger verifiable design. The AML adapter must retain message times
and make the latest valid memory easy for the answer model to recognize.

### 5.5 Context quality

MemoraX: Type ordered, timestamped, per item truncated, globally bounded output. Leaderboard output
is exceptionally compact relative to most entries.

RE-call: Citable JSON evidence preserves full chunk text and retrieval order. It has item and token
budget mechanisms but no AML specific compact coding packer.

Assessment: MemoraX has the demonstrated advantage. This is the second largest RE-call gap.

### 5.6 Isolation and reliability

MemoraX: Strong repository identity checks, scope consistency, local redaction, buffering, retries,
and client specific transcript authority. The public Add path is asynchronous.

RE-call: Exact tenant scoping, row level database controls, generation binding, idempotency receipts,
connection pooling, timeouts, and source security policy. No AML HTTP wrapper exists.

Assessment: RE-call has sufficient primitives. The contract wrapper and load profile remain to be
implemented and tested.

### 5.7 Transparency and reproducibility

MemoraX: The integration layer is open and well documented. The winning memory engine and AML
configuration are closed, and leaderboard `v0.5` is not mapped to public package `0.1.16`.

RE-call: Core storage, retrieval, trust, adapters, benchmarks, preregistrations, and result artifacts
are open. The working tree has a substantial unreleased graph delta, so the submission version must
be frozen deliberately.

Assessment: RE-call can be materially more reproducible, which is valuable for Academic admission,
but reproducibility does not substitute for task score.

## 6. Root cause model for MemoraX's lead

The following is an inference ranked by confidence, not a claim about hidden source code.

### High confidence

1. The winning system does more than return raw historical transcripts. Its output size and typed
   API strongly indicate selection or compression into memory sized units.

2. It uses both semantic and lexical retrieval signals, because the public provider contract
   exposes independent dense and sparse candidate widths.

3. It preserves and renders time, which helps distinguish old attempts from current decisions.

4. It supplies much less context to the downstream coding agent than most Academic leaders.

### Medium confidence

1. Add time extraction probably separates durable procedures, facts, and episodes. The typed Search
   response requires those types to be produced somewhere.

2. Query relevant compression probably preserves code entities and actionable validation details.
   Brevity without those details would resemble the lower scoring compact systems.

3. The system probably consolidates redundant experiences. The returned size would otherwise grow
   rapidly across historical tasks.

### Low confidence or unknown

1. The exact embedding model.

2. Whether ranking is reciprocal rank fusion, learned fusion, or a reranked mixture.

3. Whether a knowledge graph materially contributes.

4. Whether its advantage comes mainly from bug tasks, feature tasks, or repository specific
   concentration beyond the published category totals.

5. Whether the current public `memorax-code` release corresponds to the leaderboard service.

## 7. Recommended RE-call submission architecture

### 7.1 Synchronous Add adapter

Implement a small HTTP service dedicated to AML rather than changing production MCP semantics.
Map the exact AML `user_id` to the RE-call tenant boundary without normalization. Keep `session_id`
as source grouping metadata and `request_id` as the idempotency key.

For each Add request:

1. Validate all required fields and preserve message order.

2. Check the durable idempotency receipt using `request_id` and a canonical request fingerprint.

3. Persist raw message evidence under stable source and chunk identifiers derived from `user_id`,
   `session_id`, `request_id`, message position, and content digest.

4. Compile structured coding memories from the messages. If the session arrives in several chunks,
   preserve chunk order and allow later calls to add new records without erasing prior records.

5. Embed and index both raw evidence and compiled records.

6. Verify the new records are queryable inside the same tenant transaction boundary.

7. Record the success receipt, then return HTTP 200 with the exact echoed identifiers.

The compiler should produce several narrow records rather than one broad summary. A practical schema
is:

1. `memory_kind`: symptom, root cause, failed attempt, successful repair, architectural decision,
   procedure, validation, constraint, or repository fact.

2. `task_shape`: short description of the bug or feature request.

3. `entities`: file paths, symbols, commands, error strings, services, configuration keys, and
   dependencies.

4. `action`: what was changed or attempted.

5. `outcome`: succeeded, failed, partial, reverted, or unknown.

6. `evidence`: compact verbatim excerpts or stable raw chunk references.

7. `validation`: exact tests, commands, exit status, and observed result when present.

8. `event_time`: source timestamp and session ordering.

9. `supersedes` and `depends_on`: only when directly supported by the source session.

The generated text should read as a reusable memory, not as a prediction for a future question. For
example: “In repository X, error Y came from configuration Z. Changing file A at symbol B fixed it;
test C passed. Attempt D did not help.” That is admissible stored evidence if produced from Add data.
It should retain links to raw excerpts so the representation remains auditable.

### 7.2 Retrieval and ranking

Use the current hybrid core before introducing new algorithms:

1. Dense retrieval with the strongest permitted embedding profile.

2. PostgreSQL lexical retrieval for exact error strings, paths, symbols, and configuration keys.

3. A candidate pool near 100 per active leg.

4. Reranking over the fused pool if AML confirms the chosen reranker is permitted. If the model rule
   covers all learned models, use `gpt-4o-mini` only, or run a deterministic fusion arm.

5. Prefer compiled coding memories, but retain raw transcript evidence as a rescue source.

6. Apply diversity constraints so one verbose historical task cannot occupy the entire pack.

7. Preserve hard `user_id` isolation in every retrieval leg.

Do not use Search time generation to manufacture a direct answer. If `gpt-4o-mini` is used at Search,
limit it to selecting or ordering existing memory identifiers and return the stored evidence text.
Confirm this design with AML before relying on it.

### 7.3 Compact evidence packer

The packer should transform ranked stored records into a short evidence sequence without adding new
claims. A reasonable first profile is 6,000 through 8,000 characters, chosen because it brackets
MemoraX's published 6,312 character average and fits the coding agent's input budget. The exact
budget must be preregistered and measured rather than copied as dogma.

Recommended allocation:

1. Two high confidence task analogues, with successful outcomes preferred.

2. One failure or negative lesson when it prevents repeating a known bad approach.

3. One durable architectural or procedural record.

4. One raw evidence excerpt when compiled memories omit an exact code token needed for the task.

Each returned item should contain a stable ID, a compact content block, a comparable relevance score,
and the original or persistence time. Put the most actionable record first because AML preserves
order.

### 7.4 Benchmark specific trust profile

Retain these production safety properties:

1. Exact tenant isolation.

2. Invalid time window rejection.

3. Explicit supersession and successor preference.

4. Source provenance and traceability.

5. Prompt injection neutralization in stored content and output framing.

Treat these as submission profile experiments:

1. Confidence threshold at zero versus a low calibrated threshold.

2. Entailment demotion on or off.

3. Strict empty result abstention versus best available evidence.

4. Graph tail replacement on or off after real typed relations exist.

The likely AML profile suppresses similarity based abstention while retaining hard invalidity and
isolation. That is a hypothesis, not a result.

## 8. Experiment plan before Full submission

The public AML smoke cannot validate quality, and the private coding dataset cannot be reconstructed.
Use a proxy that measures the same outcome: executable coding task success after retrieval from
historical engineering sessions.

Reuse the existing `agent-memory-bench` task success design rather than inventing a prose judged
benchmark. For the admission sprint, take a bounded subset with at least 12 distinct tasks, balanced
between bug fixes and new features. Each task needs a restored repository state, several relevant
historical sessions, many distractor sessions, and an executable checker. Avoid any repository known
or suspected to belong to AML.

Predeclare three sequential comparisons:

1. Raw session indexing versus Add time compiled coding memories. Hold retrieval and output budget
   fixed. This tests the largest architectural hypothesis.

2. Full ranked chunks versus the compact diverse evidence pack. Hold ingestion and candidate
   retrieval fixed. This tests conversion without confounding discovery.

3. Baseline hybrid retrieval versus reranked wide retrieval. Hold ingestion and evidence packing
   fixed. Run only after the first two comparisons establish a useful representation.

Primary endpoint: task checker success.

Secondary endpoints: retrieval of the annotated useful historical task, context precision, negative
transfer count, returned characters, input tokens, Add latency, Search latency, provider errors, and
idempotent retry correctness.

Required controls:

1. Tasks with no useful prior memory, to detect harmful irrelevant context.

2. Duplicate histories, to test consolidation and pack diversity.

3. Superseded or reverted repairs, to test temporal handling.

4. Exact string tasks containing error messages and symbols, to ensure lexical retrieval remains
   valuable.

5. Semantic analogues with different identifiers, to ensure dense retrieval contributes.

Do not infer an AML percentage from this proxy. Use it to reject bad designs and freeze a plausible
one before the scarce Full run.

## 9. Work order through September 20

### September 11 and 12: contract and architecture

1. Ask AML whether the `gpt-4o-mini` rule covers embedding and reranking models or only generative
   model calls.

2. Ask whether stored Add time summaries and structured coding memories are accepted as memory
   evidence, provided Search does not generate final answers.

3. Confirm that the second coding cycle keeps 150 tasks, Top K 100, and the same division rules.

4. Choose Academic Methods and the platform deployed Docker route unless operating a public endpoint
   is clearly easier.

5. Freeze an adapter design and create its preregistration before any quality measurement.

### September 12 and 13: conformant service

1. Implement `/health`, synchronous Add, and Search.

2. Wire exact `user_id` tenant isolation and durable `request_id` idempotency.

3. Add contract tests for split sessions, retries, duplicate requests, conflicting duplicates, empty
   results, Top K enforcement, and ignored metadata.

4. Run the official public compatibility smoke as soon as access is available.

### September 13 through 15: memory compiler

1. Implement raw transcript persistence first.

2. Add `gpt-4o-mini` coding memory extraction with source grounded records.

3. Preserve exact paths, symbols, commands, errors, outcomes, and validation.

4. Add deterministic normalization and deduplication. Never merge records with conflicting outcomes
   without retaining time and source evidence.

### September 15 through 17: evidence packing

1. Widen candidate retrieval.

2. Add type and source diversity.

3. Add the fixed character budget and stable serialization.

4. Compare raw and compact packs under the same task checker.

### September 17 through 18: reliability

1. Load test Add concurrency from 16 through 64 and Search concurrency from 16 through 256 because
   those are the ranges exposed by the evaluation interface.[^1]

2. Verify that HTTP 200 follows persistence, not queue acceptance.

3. Exercise retry storms, provider timeouts, PostgreSQL reconnection, and process restart.

4. Confirm that no request body or credential is logged unnecessarily and document 30 day data
   deletion behavior.

### September 19: freeze

1. Select the winning preregistered proxy arm.

2. Pin commit, Docker image, dependency versions, embedding profile, model, prompts, pool widths,
   output budget, and environment variables.

3. Publish complete run instructions and prior work attribution.

4. Run the container from a clean machine or CI worker.

5. Do not change the version after requesting the formal Full evaluation.

### September 20: admission opening

1. Recheck the official notice because September 20 is currently an expected opening date.

2. Submit the fixed Academic coding method and complete the smoke path.

3. Inspect smoke failures only for compatibility and operations. Do not treat a smoke pass as a
   quality result.

4. Start Full only after the fixed version, load behavior, and data handling obligations are ready
   for publication.

## 10. Priority ranking

### Priority zero: admission blockers

1. AML HTTP contract.

2. Exact `user_id` isolation.

3. Synchronous and idempotent Add.

4. Docker reproduction and clean startup.

5. Written clarification of model restrictions.

### Priority one: probable score drivers

1. Add time coding memory compiler.

2. Compact evidence packer.

3. Wide hybrid retrieval followed by narrow output.

4. Lexical preservation of code entities.

5. Temporal outcome and validation representation.

### Priority two: conditional improvements

1. Reranking, after model eligibility is confirmed.

2. Query and option aware retrieval, if the coding track actually sends options.

3. Raw transcript rescue items.

4. Low threshold versus zero threshold calibration.

### Priority three: after admission core is stable

1. Graph tail replacement with real typed relations.

2. Broader graph traversal.

3. New learned sparse models.

4. Fine tuning.

The last group may improve RE-call generally, but it has lower expected value for this admission
because it adds mechanisms before the benchmark native memory representation exists.

## 11. Risks and decision gates

### Risk: cross benchmark optimism

RE-call's Bench'd scores demonstrate capability under a different protocol. They do not predict AML
coding task resolution. Gate: require executable task success on the coding proxy before making any
top three forecast.

### Risk: copying MemoraX's visible shell

MemoraX's public code is not its server algorithm. Copying XML formatting, memory type names, or a
4,000 character default would imitate the wrapper rather than the cause. Gate: every copied design
choice needs a paired proxy result or a direct AML contract reason.

### Risk: prohibited Search generation

The successful Bench'd digest can look like an answer rather than stored evidence. Gate: create
summaries during Add, retain raw evidence, and obtain AML clarification before any Search time model
selection.

### Risk: model rule ambiguity

Voyage embeddings and Voyage reranking powered the Bench'd result, while AML says the model used in
Add and Search must be `gpt-4o-mini`. Gate: written organizer confirmation before freezing any
non OpenAI retrieval model.

### Risk: one Full every three months

A weak Full run cannot be cheaply replaced. Gate: contract smoke, load test, proxy outcome test, and
clean Docker reproduction all pass before Full.

### Risk: overinvestment in graph work

Graph retrieval has a modest current signal but lacks coding task outcome evidence and real typed
relations in the measured served corpus. Gate: no graph work on the admission critical path until
the memory compiler and compact packer pass their proxy gates.

### Risk: leaderboard movement

The second cycle may add stronger entries or change the evaluation contract. Gate: target at least
55 percent rather than the mathematical 53.33 percent minimum, and recheck the board and official
notice before submission.

## 12. Final assessment

The case for attempting the Academic coding board is strong. The current lead is only one resolved
task above 52 percent, RE-call has competitive retrieval machinery, and its engineering quality is
well suited to a reproducible Academic submission. The case for claiming top three today is not
strong. No current RE-call run uses the AML coding contract, the protected coding tasks are not
available, and MemoraX's causal advantage cannot be recovered from its public repository.

The most likely route to a top result is not a new graph algorithm. It is to turn RE-call from a
high quality document retriever into a coding experience memory system at the AML boundary:

1. Store raw sessions safely and synchronously.

2. Compile them into small, typed, evidence grounded engineering memories.

3. Retrieve broadly across semantic and exact code signals.

4. Return a short, diverse, temporally clear pack that the fixed coding agent can act on.

5. Validate with executable task success before spending the scarce Full run.

That plan borrows the publicly evidenced strengths of MemoraX without pretending its closed server
has been reverse engineered, and it builds directly on the strongest verified parts of RE-call.

## Sources

[^1]: Agent Memory Leaderboard, competition, participation, evaluation gate, and schedule, accessed
      2026-09-11: [official site](https://agentmemoryleaderboard.ai/).

[^2]: Agent Memory Leaderboard coding row data, accessed 2026-09-11:
      [official coding data asset](https://agentmemoryleaderboard.ai/static/coding-data.js?v=coding-hs-v3-submitted-api-20260818-no-memory-8000).
      Task counts are arithmetic inferences from the published percentages and the official 150 task
      total.

[^3]: MemoraX architecture at the inspected commit:
      [ARCHITECTURE.md](https://github.com/memorax-ai/memorax-code/blob/4b7fdcc8d413db9656f0c52fd3a1432043429f66/ARCHITECTURE.md).

[^4]: RE-call product and architecture at the released baseline:
      [README.md](https://github.com/GiulioDER/RE-call/blob/f451faa30a2871c0da63ab2e3be7bf952093dcda/README.md#L32-L44).

[^5]: RE-call's signed Bench'd campaign and adapter configuration:
      [Bench'd writeup](https://github.com/GiulioDER/RE-call/blob/f451faa30a2871c0da63ab2e3be7bf952093dcda/benchmarks/benchd/WRITEUP.md),
      [adapter documentation](https://github.com/GiulioDER/RE-call/blob/f451faa30a2871c0da63ab2e3be7bf952093dcda/benchmarks/benchd/README.md#L253-L280).

[^6]: AML public repository, coverage and unpublished coding materials:
      [official README](https://github.com/AML-memory/agent-memory-leaderboard#evaluation-at-a-glance).

[^7]: AML controlled evaluation boundary:
      [official documentation](https://agentmemoryleaderboard.ai/#documentation).

[^8]: AML Add/Search contract and evaluation quotas:
      [official API guide](https://agentmemoryleaderboard.ai/#guide).

[^9]: MemoraX manual search, lifecycle, and scope flow:
      [architecture sections 3.2 and 3.3](https://github.com/memorax-ai/memorax-code/blob/4b7fdcc8d413db9656f0c52fd3a1432043429f66/ARCHITECTURE.md#32-hook-and-retrieval-data-flow).

[^10]: MemoraX public provider payload and endpoints:
       [adapter source](https://github.com/memorax-ai/memorax-code/blob/4b7fdcc8d413db9656f0c52fd3a1432043429f66/packages/ts/memorax-code-backend/src/provider/memorax/adapter.ts#L84-L108),
       [Add payload](https://github.com/memorax-ai/memorax-code/blob/4b7fdcc8d413db9656f0c52fd3a1432043429f66/packages/ts/memorax-code-backend/src/provider/memorax/adapter.ts#L378-L425).

[^11]: MemoraX writeback configuration:
       [configuration documentation](https://github.com/memorax-ai/memorax-code/blob/4b7fdcc8d413db9656f0c52fd3a1432043429f66/docs/configuration.md#automatic-writeback).

[^12]: MemoraX Search payload construction:
       [adapter source](https://github.com/memorax-ai/memorax-code/blob/4b7fdcc8d413db9656f0c52fd3a1432043429f66/packages/ts/memorax-code-backend/src/provider/memorax/adapter.ts#L221-L262).

[^13]: MemoraX retrieval defaults and context rendering:
       [configuration documentation](https://github.com/memorax-ai/memorax-code/blob/4b7fdcc8d413db9656f0c52fd3a1432043429f66/docs/configuration.md#memorax-retrieval),
       [renderer source](https://github.com/memorax-ai/memorax-code/blob/4b7fdcc8d413db9656f0c52fd3a1432043429f66/packages/ts/memorax-code-backend/src/provider/memorax/adapter.ts#L511-L576).

[^14]: RE-call indexing and chunk defaults:
       [index.py](https://github.com/GiulioDER/RE-call/blob/f451faa30a2871c0da63ab2e3be7bf952093dcda/recall/index.py#L38-L40).

[^15]: RE-call hybrid retrieval implementation:
       [retriever.py](https://github.com/GiulioDER/RE-call/blob/f451faa30a2871c0da63ab2e3be7bf952093dcda/recall/retriever.py#L383-L672).

[^16]: RE-call measured query and history fusion:
       [retriever.py](https://github.com/GiulioDER/RE-call/blob/f451faa30a2871c0da63ab2e3be7bf952093dcda/recall/retriever.py#L674-L703).

[^17]: RE-call trust and evidence assembly:
       [trust.py](https://github.com/GiulioDER/RE-call/blob/f451faa30a2871c0da63ab2e3be7bf952093dcda/recall/trust.py),
       [evidence.py](https://github.com/GiulioDER/RE-call/blob/f451faa30a2871c0da63ab2e3be7bf952093dcda/recall/evidence.py#L286-L417).

[^18]: RE-call server connection modes and idempotency receipts:
       [store.py](https://github.com/GiulioDER/RE-call/blob/f451faa30a2871c0da63ab2e3be7bf952093dcda/recall/store.py#L785-L850),
       [receipt implementation](https://github.com/GiulioDER/RE-call/blob/f451faa30a2871c0da63ab2e3be7bf952093dcda/recall/store.py#L1461-L1539).

[^19]: RE-call evidence policy and selection behavior:
       [evidence.py](https://github.com/GiulioDER/RE-call/blob/f451faa30a2871c0da63ab2e3be7bf952093dcda/recall/evidence.py#L43-L70),
       [bundle builder](https://github.com/GiulioDER/RE-call/blob/f451faa30a2871c0da63ab2e3be7bf952093dcda/recall/evidence.py#L286-L417).
