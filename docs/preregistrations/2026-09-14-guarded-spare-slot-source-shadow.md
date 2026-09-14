# Pre-registration: guarded spare-slot source shadow

Date: 2026-09-14.

## Question

Can the post hoc guarded spare-slot rescue be implemented as a private same-vector shadow that
preserves the existing alpha `0.08` evidence prefix, adds no retrieval or embedding call, and earns
fresh human-reviewed evidence of higher fact support without increasing false answers?

## Prior evidence and contamination boundary

The inspected 72-query trace and the rule derived from it are development data. The fixed replay,
its thresholds, and its result are recorded in
`docs/results/2026-09-14-source-admission-spare-slot-replay.md`. Those queries may be used only for
implementation parity. They cannot contribute to the fresh quality decision.

Global alpha `0.15` is closed by
`docs/results/2026-09-13-live-source-conditioning-alpha015-screen.md`. This experiment does not
reopen it and does not replace or reorder any alpha `0.08` item.

## Frozen candidate rule

The treatment starts from the exact alpha `0.08` shadow selection and runs only when fewer than
five items were selected. It considers unrepresented sources whose candidate chunks have verdict
`ok` or `low_confidence`.

A source enters the dual-leg lane when its best dense rank and best sparse rank are each at most
`5` and its maximum cosine is at least `0.35`. A source enters the lexical-dominant lane when its
best sparse rank is `1`, it contributes at least two sparse top `10` chunks, and its maximum cosine
is at least `0.28`. Eligible chunks use the corresponding cosine floor. Sources are ordered by
lane, combined rank, descending fitted support, and stable source identity. Items are appended
round robin until the existing five-item budget is full.

No threshold, rank bound, ordering rule, metric, or gate in this record may be edited after the
first measurement. Corrections must be appended.

## Implementation arm

Add an opt-in shadow policy whose default remains disabled. It must reuse the main request's exact
dense, sparse, trusted-pool, query-vector, generation, and calibration trace. It may not call the
embedder, dense store, sparse store, reranker, or trust gate again. It must never alter public
evidence.

The private diagnostic may expose aggregate counts, lane names, latency, lineage, and SHA256
identifier hashes. It must not expose raw candidate identifiers, source paths, query text, or
candidate text.

## Implementation parity sample

Use the immutable 72-query captured trace only to establish code parity with the frozen replay.
The shadow implementation must satisfy all of these conditions:

1. All 72 alpha `0.08` base selections match their captured hashes.
2. All 72 candidate selections match the frozen replay hashes.
3. All 72 candidate selections preserve the complete base prefix.
4. Public evidence is byte-identical with the policy disabled and enabled.
5. Shadow errors are zero and every sampled request records a timing span.
6. Instrumented doubles prove zero additional embedding, dense, sparse, rerank, and trust calls.

Failure of any condition gives `REPAIR_SHADOW`. Passing gives `READY_FOR_FRESH_VALIDATION`; it is
not a quality result.

## Fresh blinded validation

Draw queries without inspecting treatment outcomes. The primary cohort contains fresh queries for
which alpha `0.08` returns fewer than five items and at least one frozen rescue lane fires. Stop
when the cohort contains at least 20 answerable and 20 unanswerable queries, or after 500 eligible
candidate queries have been screened. If either quota is not reached, report `INSUFFICIENT`.

Before arm identity is revealed, a human reviewer records whether each returned item supports the
question, which required facts it covers, whether the evidence is sufficient to answer, and whether
an unanswerable query would be falsely answered. Source and query authorship must be disjoint from
the inspected 72-query development set.

Report activation prevalence separately. It is descriptive and is not a promotion gate.

## Predictions and decision rule

I predict the candidate will add supported facts on at least one answerable query, lose no facts or
complete answers, preserve every base prefix, keep context precision within `0.05` absolute of the
base, and produce no more false answers than alpha `0.08`.

Return `PROMOTION_CANDIDATE` only if the cohort quotas and integrity conditions hold and every
prediction passes. Return `NO_QUALITY_GAIN` if integrity holds but no supported fact is gained.
Return `REJECT_RESCUE` for any fact loss, complete-answer loss, false-answer increase, precision
failure, public evidence difference, or extra retrieval call. No result in this protocol directly
licenses active serving; a promotion candidate still requires a separate staged rollout decision.

## Planned verification

```powershell
python -m pytest tests/test_source_conditioning.py tests/test_live_source_conditioned_admission.py tests/test_guarded_spare_slot_shadow.py -q
python -m ruff check recall/source_conditioning.py recall_mcp/service.py tests/test_guarded_spare_slot_shadow.py
python -m mypy --explicit-package-bases --follow-imports=skip recall/source_conditioning.py recall_mcp/service.py
python tools/architecture_map.py --check
git diff --check
```

<!-- frozen_above -->

## Implementation result

Measured 2026-09-14. The implementation verdict is `READY_FOR_FRESH_VALIDATION`, not a quality
result. The production helper matched the frozen replay on all 72 rows, preserved all 72 base
prefixes, and reproduced the earlier artifact SHA256 exactly. The service boundary tests prove
public evidence parity, a timing receipt, private diagnostics, and zero additional embedding,
dense, sparse, reranking, or trust work. The complete result and reproduction commands are in
`docs/results/2026-09-14-guarded-spare-slot-shadow-implementation.md`.
