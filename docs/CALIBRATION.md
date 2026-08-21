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

Strict refusal before returning **corpus text** is still deferred: a missing or stale artifact
produces a result marked uncalibrated rather than an error.

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
| `CALIBRATION_STALE` | A certified artifact no longer binds to the current lineage | Recalibrate (a rebuild or privacy erasure changed the corpus fingerprint) |
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
