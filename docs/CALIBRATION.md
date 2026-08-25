# Generation-bound calibration

> New to this? [docs/FIRST_CALIBRATION.md](FIRST_CALIBRATION.md) walks the whole path from an
> indexed folder to a trusted answer, including why indexing alone reports `INDEX_NOT_READY`
> and why `recall calibrate`'s file is never read back. This page is the reference.

RE-call v1 stores calibration as immutable evidence for one tenant and one exact index generation.
It never treats a threshold measured on another corpus, model revision, chunker configuration, or
labelled query set as applicable merely because the vector dimensions match.

## Binding and lifecycle

`CalibrationArtifactV2` records the tenant and generation IDs, full embedder identity, pipeline and
corpus fingerprints, canonical labelled-query-set digest, measured scores, threshold, scale,
separability interval, sample counts, certification decision, creation metadata, and artifact
checksum. The labelled questions are stored separately from their measurements. This permits reuse
of human labels while requiring retrieval to be measured again for every new generation.

Artifacts move through these states:

```text
certified measurement: draft -> published -> superseded
failed certification:  rejected
legacy JSON import:    legacy_unbound
```

Publication is serialized with a PostgreSQL advisory lock. At most one artifact is published for a
tenant and generation, while superseded artifacts and their audit events remain available.
Uncertified evidence is retained as `rejected`; publication refuses it without deleting it.

## Labelled query set

The input is a JSON array. Each entry requires a nonempty query and a boolean `answerable` field.
`relevant_ids` is optional evidence and is canonicalized when supplied.

```json
[
  {"query": "What is the current API limit?", "answerable": true},
  {"query": "Which habitat is used on Mars?", "answerable": false}
]
```

Array order and `relevant_ids` order do not change the digest. Duplicate labelled entries are
refused. Certification still applies the existing minimum sample and separability confidence rules;
a thin or overlapping set produces a preserved rejected artifact.

## Create and publish

The generation must be `ready`, `active`, or `retired`, and the selected embedder implementation
must match its stored model and dimension.

```bash
recall --tenant acme --serving-dsn "$RECALL_SERVING_DSN" \
  calibration calibrate --generation gen_... --queries queries.json

recall --tenant acme --serving-dsn "$RECALL_SERVING_DSN" \
  calibration calibrate --generation gen_... --queries queries.json --publish
```

Publishing is explicit. Creating a certified draft does not make it active. Creating with
`--publish` first stores the measurement, then attempts publication, so failed certification still
leaves inspectable evidence.

## Carry a threshold across a rebuilt generation

A calibration binds to one `generation_id`. Rebuild the generation to absorb new documents and the
child has no artifact, so `resolve` returns `STALE`, strict policy refuses every query and
development mode degrades every query. Without a way to re-verify, a corpus that grows at all costs
a full manual recalibration, and the practical result is that live indexes stop being rebuilt.

```bash
recall --tenant acme --serving-dsn "$RECALL_SERVING_DSN" \
  calibration carry-forward --generation gen_child --publish
```

**This is not a tolerance and it does not loosen the gate.** Nothing is carried because the corpus
delta looked small. The parent's own stored labelled query set is re-scored against the child
generation, and the inherited threshold must pass on those fresh scores. What is inherited is the
threshold, not the certification.

Two conditions must both hold, and the second exists because the first cannot see the failure that
matters here.

| Check | Question it answers |
|---|---|
| separability CI lower bound `>= MIN_SEPARABILITY` | are the two classes still ordered? |
| false-abstain and false-confirm rates `<= --max-error` | is the fixed threshold still between them? |

⛔ **Separability alone is not sufficient, and this is the whole reason carry-forward needs a rule
`calibrate` does not.** Separability is threshold-free. Documents that lift every unanswerable
score by the same amount leave the ordering perfect, so AUC stays at 1.00 and certification passes,
while the entire unanswerable class slides above a threshold that is no longer allowed to move. A
refit cannot be fooled that way because it puts the cut back between the classes. Carry-forward
holds the cut still, so it has to check the cut. `tests/test_calibration_carry_forward.py` builds
exactly that corpus and asserts the refusal.

Three refusals happen before any embedding work:

- **a different pipeline fingerprint**, refused outright at any delta. A threshold is a property of
  an embedder's cosine regime: 2026-08-17 measured `voyage-4` at 0.269 to 0.413 on one corpus where
  `voyage-code-3` returned 0.480 to 0.834 on another. No delta is small enough to make that safe.
- **a delta above `--max-corpus-delta`** (default 0.25). Re-scoring a query set says nothing about
  the queries nobody labelled, so past some point the labelled set describes a corpus that no
  longer exists.
- **an unchanged corpus fingerprint**, which means the caller named the generation the calibration
  is already bound to.

An uncertified or unpublished parent is refused too, so a draft threshold cannot acquire a
certification by being copied forward.

⚠️ **Both defaults are ceilings on the mechanism, not measured safe distances.** The only delta
this has been measured at is recorded in
`docs/preregistrations/2026-08-20-calibration-carry-forward.md`. Lower them per tenant; do not read
the defaults as evidence.

The artifact records where its threshold came from, inside the checksum, so provenance cannot be
edited away afterwards. `calibration list` reports `threshold_was_measured_here`,
`carried_forward_from` and `corpus_delta`, because after a chain of rebuilds an operator's real
question is which of these thresholds anyone actually measured.

The output also prints `refit_threshold`: what a fresh fit on these scores would have chosen. It
**changes nothing**. It is there so that an operator watching the inherited number drift away from
the data gets the warning before the drift is large enough to fail.

## Watch for drift between rebuilds

Carry-forward answers the drift question *after* a rebuild, and `resolve` answers it as a yes/no
about identity. Neither can be asked the question an operator actually has: **the corpus on disk has
moved on, is the threshold serving it still deciding anything?**

```bash
# a rebuilt generation: compared and re-scored
recall --tenant acme calibration drift --generation gen_child

# a live directory: compared only, because nothing has indexed it
recall --tenant acme calibration drift --path ./docs --strict
```

Two tiers, and which one produced the verdict is always visible in the output.

| Tier | Cost | What it establishes |
|---|---|---|
| **screen** `corpus_delta` over `(uri, sha256)` | a manifest comparison, no embedding, no retrieval, no model load | an upper bound on how much *could* have moved |
| **probe** the calibration's own stored labelled query set, replayed | one retrieval per labelled query | what the frozen threshold costs now, per class |

⛔ **The screen firing is never reported as a verdict.** Where the probe cannot run, the strongest
verdict is `recalibrate_recommended`, and the report names the check that was not made. A directory
therefore can never reach `recalibrate_required`, however total its delta, because nothing has
scored it.

### Why there is no delta at which recalibration is demanded outright

An earlier draft demanded it past `--max-corpus-delta`, reasoning that the labelled set had stopped
describing the corpus. **Measured 2026-08-21 over 57 snapshots of three real corpus histories**
(`docs/preregistrations/2026-08-21-calibration-drift-trigger.md`,
`results/calibration_drift_2026-08-21.json`), that reasoning is wrong:

- the frozen threshold first crossed the 0.10 error bound at a corpus delta of **0.945**, and never
  below it;
- a delta-only rule at 0.25 fires on **56 of 57** snapshots and is right about **5**, a precision of
  **0.09**;
- the labels were durable where the argument assumed rot. At delta 0.981 only **27.5%** of the
  answerable queries' original evidence chunks still existed, and false abstains were **0.025**;
- what moved was the **false-confirm** rate, tracking corpus **growth** rather than change (Spearman
  0.95 against growth on `docs`, where growth and delta are collinear at 0.98, so this measurement
  cannot tell them apart). A top-1 cosine is a max over the index: added documents can only raise an
  unanswerable query's score, and how much of a corpus was rewritten predicts nothing about that.

⚠️ **This does not license raising `--max-corpus-delta` on carry-forward.** That bound governs
whether a threshold may be *inherited*, on one repository, one embedder, generated queries and exact
rather than approximate search. What the numbers above establish is that a delta is a poor alarm,
not that a large delta is safe.

### Automatic recalibration

`RECALL_AUTO_CALIBRATE` decides how far a build is allowed to act on what it finds.

| value | after a generation build |
|---|---|
| `off` | nothing, including opening a connection |
| `warn` (default) | measure drift, print the report, leave the decision to the operator |
| `auto` | additionally re-establish the calibration |

Under `auto`, two paths are tried cheapest first, and **neither loosens certification**:

1. **carry the threshold forward**, keeping an operating point the operator has already seen;
2. **refit on the same stored labelled evidence** when the threshold has to move, which is exactly
   the case carry-forward refuses.

```bash
recall --tenant acme calibration auto --generation gen_child
```

It **will not invent a first calibration**. A tenant with no published artifact reports `skipped`,
because deciding what the labelled questions should be is not a decision to make unattended, and a
post-build hook that failed on every fresh install is a hook that gets removed. For the same reason
a missing calibration reports `unknown` rather than `stable`: an uncalibrated tenant has zero
measured drift by every arithmetic definition, and calling that stable would say a threshold which
does not exist is holding up fine.

## Inspect and transfer

```bash
recall --tenant acme calibration list
recall --tenant acme calibration show cal_...
recall --tenant acme calibration export cal_... --output calibration-v2.json
recall --tenant acme calibration import calibration-v2.json
```

Exports include the versioned artifact, reusable labels, artifact checksum, and a checksum over the
whole bundle. Import verifies both checksums, the tenant, generation lineage, embedder identity, and
query-set digest. An imported v2 artifact returns to `draft` or `rejected`; import never publishes
it automatically.

An old `calibration.json` can be imported for audit continuity. It is stored as `legacy_unbound`
because it cannot prove tenant, generation, pipeline, corpus, or query-set identity. Library, CLI,
and MCP search paths never select that file automatically.

## Search resolution

`GenerationStore` pins the tenant's active generation for the whole search and resolves calibration
inside that snapshot. A match is usable only when tenant, generation, pipeline fingerprint, corpus
fingerprint, stored query-set contents, query-set digest, checksum, and certified publication state
all agree. Python and MCP results carry these identities in-band. The compatibility property
`calibrated` is true only for that exact certified match.

A privacy erasure changes the effective corpus fingerprint of every affected generation. This
immediately makes prior calibration stale, prevents it from being republished, and carries the
same tombstone-derived fingerprint into replacement generations built from the old manifest.

🔁 **Corrected 2026-08-25.** This paragraph read *"Strict refusal before returning corpus text is
still deferred: a missing or stale artifact produces a result marked uncalibrated rather than an
error."* That was true before `TrustPolicy` landed and has not been true since, and it contradicted
the section **"Why the refusal happens before retrieval"** forty lines below it in this same
document. The stale sentence came first, so a reader who stopped there concluded that strict mode
did not protect them.

**Strict refusal has landed, and it happens before any corpus text is fetched.** The gate sits
above `retriever.search(...)` in `trusted_search`, so a refusal cannot leak chunk text, source
names or previews, because none were ever retrieved. `TestStrictRefusesBeforeAnyCorpusRead` in
`tests/test_strict_trust_search.py` asserts this structurally, with a store whose retrieval methods
raise if they are reached at all, rather than by inspecting the refusal's message.

What the old sentence described is now **development mode only**: it retrieves, marks the result
`trust_state=degraded` with `calibrated=false`, and names the reason in `failure_code`. Strict is
the default for both the library and the MCP service, and omitting a policy cannot open the gate.

Refusal at **promotion** time has landed. For a tenant served under production (see
`GenerationManager.certification_required`, which reads the serving environment rather than the
build one), `generation promote` resolves the generation's calibration and raises `UnsafePromotion`
unless the status is CERTIFIED,
so an absent, stale, mismatched, rejected or superseded artifact cannot be made active. The check
runs inside the promotion transaction, after the generation's existence and state are established,
and a calibration that cannot be resolved fails closed.

⛔ **`generation rollback` is exempt, on purpose.** It is the incident path, and a gate there blocks
recovery exactly when recovery is needed. It activates the previous generation whatever the status,
and records that status plus the operator's `provisional_reason` in `generation_rolled_back`. The
argument is in `docs/UNCALIBRATED_FIRST_RUN_DESIGN.md` section 6, decision 2, and its shortest form
is: prevented, no; hidden, never.

## Operations and privacy

Apply schema migrations before starting the serving process. Startup performs only compatibility
checks. Grant the serving role DML on `recall_calibrations` and
`recall_calibration_query_sets`, without schema ownership or DDL privileges, as shown in
[MIGRATIONS.md](MIGRATIONS.md).

Calibration exports contain questions and raw retrieval scores. Treat them as corpus data. Creation,
rejection, publication, import, and supersession are appended to `recall_audit_events`; administrative
tools should use a meaningful actor identifier.

## Strict trust policy: what happens when calibration is absent, stale or uncertified

Before this, an absent or uncertified calibration was not an error. `trusted_search` fell back to
the library's 0.50 default and answered anyway, so the caller received a normal-looking result
whose verdicts came from a threshold nobody had certified for that corpus.

That fallback is gone. `TrustPolicy` decides what happens instead, and **strict is the default**
for the library and for the MCP service. Omitting a policy cannot open the gate.

### The six failure codes

| Code | Meaning | Remedy |
|---|---|---|
| `INDEX_NOT_READY` | No active generation for this tenant | Build and promote a generation |
| `LINEAGE_MISMATCH` | Generation pipeline or corpus fingerprint is not what the artifact bound | Recalibrate against the current generation |
| `CALIBRATION_MISSING` | No artifact bound to this tenant and generation | Calibrate against a labelled query set and publish |
| `CALIBRATION_UNCERTIFIED` | An artifact exists but was never certified | Certify and publish a replacement; the rejected evidence is retained |
| `CALIBRATION_STALE` | A certified artifact no longer binds to the current lineage | Try `calibration carry-forward` first: after an ordinary rebuild it re-verifies the existing threshold against the new generation and needs no new labels. Recalibrate if it refuses, or after a privacy erasure |
| `DEPENDENCY_UNAVAILABLE` | A dependency the gate needs was unreachable | Retry once healthy |

These are an API. Automating on them is the intended use, so their spelling is pinned by test.

`DRAFT` maps to `CALIBRATION_UNCERTIFIED` deliberately. A draft artifact is certified in the
statistical sense but has not been published, so it is not the artifact an operator chose to serve.

### Why the refusal happens before retrieval

The gate sits above `retriever.search(...)`. A refusal raised before any `query_dense` call cannot
leak chunk text, source names or previews, because none were ever fetched. The alternative,
filtering the payload afterwards, would leave a sanitiser that someone eventually forgets to call.
The regression test asserts this structurally, using a store whose retrieval methods raise if they
are reached at all, rather than by inspecting the refusal's message.

### An outage is not an empty answer

`DEPENDENCY_UNAVAILABLE` exists because "the gate ran and found nothing" and "the gate could not
run" are the same shape on the wire and opposite in meaning. Collapsing them is how a downstream
agent concludes there is no prior decision on a question that was in fact settled. Every code's
`advice` states that no trustworthy decision was possible; none of them says an answer was not
found.

### Development mode

```python
from recall.trust_policy import TrustPolicy

result = trusted_search(store, embedder, "…", policy=TrustPolicy.development())
```

Development mode retrieves, but it cannot claim anything:

- `trust_state=degraded`, and `calibrated` is `False` regardless of what the status says
- `calibration_status` names the specific reason, and `failure_code` carries the stable code
- with no threshold at all, every hit is `verdict=unverified` and `abstained` is forced `False`,
  since an abstention is itself a trustworthy decision
- if you pass an explicit `Calibration`, the verdicts it produces are kept (this is what abstention
  benchmarks measure), but the result is still `degraded` and still not `calibrated`

The CLI **and the MCP server** both read `RECALL_TRUST_MODE`. The value is matched after
`strip().lower()`, so `development`, `Development` and `  DEVELOPMENT  ` all relax the gate and
anything else is strict. A **misspelling** such as `developmnet` therefore cannot open it, which is
the property this is here for; a capital letter is not a typo, and refusing it would only produce a
strict server its operator believes is relaxed. In development the CLI prints the uncertified
threshold it is using rather than inheriting one silently, and the server logs at ERROR on every
start that its gate is open.

### What this does not protect against

Strict mode constrains what the library will *claim*. It does not verify the corpus, and it cannot
help if an operator publishes a certified artifact measured on a query set that does not represent
production traffic. A certified calibration means the binding is exact and the statistics were
computed on a labelled set; it does not mean the labelled set was a good one.
