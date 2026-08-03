# Generation-bound calibration

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
  calibrate --generation gen_... --queries queries.json

recall --tenant acme --serving-dsn "$RECALL_SERVING_DSN" \
  calibrate --generation gen_... --queries queries.json --publish
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

Strict refusal before returning corpus text is intentionally deferred to the strict-trust session.
Until then, a missing or stale artifact produces a result marked uncalibrated; production generation
promotion remains disabled.

## Operations and privacy

Apply schema migrations before starting the serving process. Startup performs only compatibility
checks. Grant the serving role DML on `recall_calibrations` and
`recall_calibration_query_sets`, without schema ownership or DDL privileges, as shown in
[MIGRATIONS.md](MIGRATIONS.md).

Calibration exports contain questions and raw retrieval scores. Treat them as corpus data. Creation,
rejection, publication, import, and supersession are appended to `recall_audit_events`; administrative
tools should use a meaningful actor identifier.
