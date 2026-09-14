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

## Fresh pool freeze

Frozen before treatment screening on 2026-09-14. The pool contains 500 unique queries, split into
250 answerable memory source queries and 250 matched absent project controls. Its SHA256 is
`af7c74d4d2b232cb79de72d5fdfd67e1ccf1d6ad814fb634ba61f98999632c75`.

The answerable sources have zero overlap with the 880 unique sources in the inspected 72 row
trace. The query text has zero exact normalized overlap with that trace. The negative prefix
`ZXQMEM` was checked absent from every candidate source before pool construction.

Rebuild and verify from the repository root:

```powershell
python scripts/build_guarded_spare_slot_fresh_pool.py --source-root "recall=C:\Users\gde00\.claude\projects\C--Users-gde00-Documents-recall\memory" --source-root "sentiment-agent=C:\Users\gde00\.claude\projects\C--Users-gde00-Documents-progetto-sentimental\memory" --trace docs/results/2026-09-13-live-source-admission-trace-capture.json --output docs/preregistrations/2026-09-14-guarded-spare-slot-fresh-pool.json
```

## Fresh screen attempt one repair

Measured 2026-09-14. The first screen stopped before scoring at query 214. It had completed 213
queries and observed 18 answerable plus 12 unanswerable activations. The harness then required a
separately repeated benchmark audit to reproduce the live shadow base order, and that duplicate
trace differed. No capture artifact had been written because the first harness wrote only at the
end.

This is a harness failure, not a treatment result. The repaired runner resolves the live shadow
hashes against the private benchmark pool without recomputing the treatment and checkpoints every
completed query. The frozen pool, treatment rule, lineage, quotas, and decision rule are unchanged.

The failed command was:

```powershell
$env:RECALL_BENCHMARK_REMOTE_CODE_ROOT='/home/sentiment/recall-repos/guarded-spare-slot-89caaf43'
$env:RECALL_SOURCE_COMMIT='ec0cabf3e2dfc778b33e214fdbfe98218cb7be62'
python -u scripts/run_live_guarded_spare_slot_fresh_screen.py --query-pool docs/preregistrations/2026-09-14-guarded-spare-slot-fresh-pool.json --artifact docs/results/2026-09-13-source-conditioning-model.json --output C:\Users\gde00\.codex\evals\guarded-spare-slot-2026-09-14\capture.json --review-output C:\Users\gde00\.codex\evals\guarded-spare-slot-2026-09-14\blind-review.json --generation-id gen_98c6f34508384ee2badffc14fcf47c4c --calibration-id cal_2cc3192509b64524a71aa949918267d8 --pipeline-fingerprint 57ee96893adaff493879a6893b9a4707675b2462dd914c51984f01813de2ba86 --corpus-fingerprint a31ac76d107eb3705ea26460f10b59edd7710df52a6088aad3f67fa3c8371085
```

## Fresh screen result

Measured 2026-09-14. The repaired screen reached `READY_FOR_BLIND_REVIEW` after 270 requests, with
23 answerable and 20 unanswerable triggered queries. All 270 timing receipts were present, and no
triggered candidate changed its alpha `0.08` prefix. This is not a quality verdict. The arms remain
unrevealed pending the preregistered human review.

The result, artifact hashes, private artifact paths, and exact reproduction command are recorded in
`docs/results/2026-09-14-guarded-spare-slot-fresh-screen.md`.

## Blind review retirement and source audit boundary

Registered 2026-09-14 before opening the private capture artifact. The human reviewer reported
that the task was not answerable from memory and that the volume and presentation of source links
made any labels unreliable. Inspection of the still-blinded review artifact confirmed that every
query had an empty `required_facts` list. The review therefore supplied evidence without an
independent fact-level answer key and asked the reviewer to manufacture that key after seeing the
retrieved material.

No response from the spreadsheet will be collected or used. The original promotion decision is
unresolvable and this run must return `INVALID_REVIEW_INSTRUMENT`. It cannot return
`PROMOTION_CANDIDATE`, `NO_QUALITY_GAIN`, or `REJECT_RESCUE` from the replacement analysis below.

After this correction is committed, the private capture may be opened only for a descriptive source
audit against labels that were already frozen in
`docs/preregistrations/2026-09-14-guarded-spare-slot-fresh-pool.json`. For each triggered answerable
query, report whether the base and candidate contain a chunk from the frozen gold source, whether an
added chunk comes from that source, and how many additions come from other sources. For each
triggered unanswerable query, report every addition as non-gold because its query contains a unique
`ZXQMEM` identifier that was verified absent from the corpus before screening. Also recheck capture
and review digests, cohort counts, and base-prefix preservation.

This audit is diagnostic only. It may establish whether the rescue reaches the intended source and
how much known non-gold material it adds. It cannot establish required-fact coverage, answer
sufficiency, or false-answer rate.

The next confirmatory quality experiment must use a new untouched holdout. Each answerable item must
freeze a canonical source, source digest, source ordinal, question, and exact answer span before any
retrieval run. Evidence quality is then the presence of the frozen answer span in admitted evidence.
An extractive answer stage must return that exact span or `NOT_FOUND`; absent-identifier controls
have `NOT_FOUND` as their frozen answer. This replaces recollection and post-treatment human
judgment with an answer key recorded before treatment.
