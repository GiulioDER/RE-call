# Pre-registration: K-2 v3, same-subject adjacency by mutual nearest neighbours within the list

**Date:** 2026-09-25   **Status:** predicted, not yet measured

Branch `claude/mm1-mm3`. Third attempt at K-2. Both earlier versions stopped at the mechanism gate
on LoCoMo, before any answer was paid for, and both results stand unrevised:

- **v1** (`docs/preregistrations/2026-09-25-aml-c9-relative-dates-and-conflict-adjacency.md`):
  subject-word Jaccard at 0.35 never fired (0 of 720; best cross-day Jaccard at most 0.250).
- **v2** (`docs/preregistrations/2026-09-25-aml-c9-same-subject-adjacency-v2.md`): voyage-code-4
  cosine at τ = 0.641, calibrated on LongMemEval, fired on 720 of 720, because cross-day windows
  inside one LoCoMo top 30 sit at a median cosine of 0.699.

The shared cause (memory `absolute-similarity-thresholds-do-not-transfer-into-one-conversation`):
an absolute threshold carries the calibration corpus's notion of "unrelated", which does not exist
inside one conversation. v3 therefore decides links **relative to the list being reordered**, and
calibrates on **lists shaped like a top 30 from one user**, not on pairs from across a haystack.

Baseline: served C9 at `3eb447c4`. Local directional experiment over public data.

## The question

With links decided by mutual nearest neighbours inside the top 30, filtered by how much a pair
stands out from that list's own similarity distribution, does placing same-subject, different-day
items side by side, newest first, raise BEAM contradiction_resolution plus knowledge_update against
H, without lowering overall accuracy or Coding MRR?

## The link rule, fixed now

Inside the top 30, using the same `voyage-code-4` document vectors as v2 (read from the index in
production; embedded from stored text offline, which v2 showed agrees with the index to cosine
0.9996 or better):

1. **Cross-day pairs only.** Pairs of items whose `created_at` fall on different UTC days and that
   both have a vector. Everything below is computed over these pairs of this list.
2. **Standing out:** for each cross-day pair, z = (cosine − mean) / standard deviation, the mean
   and standard deviation taken over all cross-day pairs of this list.
3. **Mutual nearest neighbours:** item j is i's nearest cross-day neighbour (highest cosine among
   i's cross-day partners), and i is j's.
4. **Link** i and j when 3 holds and z ≥ **z₀**. Because each item has one nearest neighbour,
   links form disjoint pairs; each pair is moved to the rank of its better-ranked member, newest
   first. Unchanged from v1 and v2: top 30 only, ranks 31 to 100 untouched, no text added, dropped
   or rewritten, default off behind a flag.

## How z₀ is fixed, before anything is scored

On LongMemEval (`xiaowu0162/longmemeval-cleaned` at `98d7416c24c778c2fee6e6f3006e7a073259d48f`),
which none of this record's measurements score, with lists built to look like one user's top 30:

1. **Questions:** the 67 eligible knowledge-update questions of v2's amendment 1 (evidence marked in
   at least two sessions on different days).
2. **One list per question, 30 windows:** the evidence windows of the earliest and the latest
   evidence session, plus 28 windows drawn from that question's own haystack (its non-evidence
   sessions), each window carrying its session's date. Drawn once with `random.Random(20260925)`.
   C9's content-only rendering and `word_windows` at 160/120, as in v2.
3. **Embed** with `voyage-code-4-v1`, passage mode. Windows already embedded for v2 are re-embedded,
   not reused, so every vector in a list comes from one run.
4. **Negatives:** every cross-day mutual-nearest-neighbour pair in a list other than the evidence
   pair. **Positive:** the evidence pair, which links only if it is itself a mutual-nearest-neighbour
   pair.
5. **z₀ is the smallest value at which spurious links average at most 0.10 per list** (at most one
   false link in ten lists), counting negatives that are mutual nearest neighbours with z ≥ z₀.
6. **Stop rule at calibration:** if fewer than 20% of the 67 evidence pairs link at z₀, v3 stops
   there.

z₀, the positive recall, the spurious-link rate and the z distributions are committed before any
LoCoMo or BEAM mechanism metric is computed.

## What I predict

Written before any v3 list is built. v2's τ came out below its band and its reorder share far above
it; these bands are wide for that reason.

**Calibration (Voyage only):**

| Metric | Predicted |
|---|---|
| z₀ | 2.5 to 4.0 |
| Evidence pairs that are mutual nearest neighbours at all (before z₀) | 0.50 to 0.90 |
| Evidence pairs linked at z₀ | 0.30 to 0.70 |

**Mechanism (no model call):**

| Metric | Predicted |
|---|---|
| LoCoMo questions (the committed 720) whose top 30 v3 reorders | 0.10 to 0.60 |
| Linked pairs per reordered LoCoMo question, median | 1 to 2 |
| BEAM questions whose top 30 v3 reorders | 0.10 to 0.60 |

The LoCoMo band sits well below v2's 1.000 because a list-relative z discounts the conversation's
general likeness. It could still land low: inside one conversation, a real old-and-new pair may not
stand out from windows that are all about the same two people, which is the risk that makes this
version inert.

**Answers**, paired on identical retrieval, read and judged as in the T-1/K-2 record; bands
unchanged from v1 and v2:

| Contrast | Set | Predicted |
|---|---|---|
| K2v3 − H | BEAM contradiction_resolution + knowledge_update (80) | **+0.02**, band −0.02 to +0.06 |
| K2v3 − H | BEAM contradiction_resolution alone (40) | +0.01, band −0.02 to +0.05 |
| K2v3 − H | BEAM 10-type mean | −0.005, band −0.02 to +0.01 |
| K2v3 − H | LoCoMo, the 720 | −0.3 points, band −1.5 to +0.5 |
| K2v3 − K0 | Coding K3 MRR | −0.01, band −0.04 to 0.00 |

## What would falsify this

- **Calibration:** fewer than 20% of evidence pairs linked at z₀ (stop rule).
- **Mechanism:** a reorder share below 0.05 (inert, as v1) or above 0.90 (links everything, as v2)
  on LoCoMo or BEAM stops the answer stage for that set.
- **Answers:** K2v3 − H on BEAM contradiction plus knowledge update at or below 0.

## Decision rule

Unchanged from v1 and v2: recommend if K2v3 − H on BEAM contradiction plus knowledge update is at
least +0.03, LoCoMo 720 at least −0.5, BEAM 10-type mean at least −0.01, and Coding MRR at least
−0.01; any change to C9 needs an explicit user decision and never during an AML job.

## How it will be measured

1. **Calibrate** on VPS3 (steps above); commit z₀ and the report.
2. **Build** the v3 rule into `same_subject_adjacent` (a `rule` argument; v1 and v2 unchanged), with
   unit tests seen red against deliberate mutations: a mutual pair above z₀ on different days
   links; a one-way nearest neighbour does not; a mutual pair below z₀ does not; a same-day pair
   never enters the computation.
3. **Mechanism** on the committed LoCoMo 720, reusing v2's item vectors
   (`k2v2/locomo-vectors.json.gz` on VPS3, already checked against the index).
4. **Answers, LoCoMo**, only if the gate passes: joined to the queued T-1 LoCoMo run if it has not
   started, otherwise its own run with a fresh H arm interleaved beside it, never compared with an
   H answered in another session.
5. **BEAM** after the VPS2 hands-off rule is lifted or a VPS3 re-collect; **Coding** after a
   re-collect that keeps item content.

**Spend.** Voyage only for calibration (about 2,000 windows). OpenRouter only for answers, inside
the T-1/K-2 cap: about USD 4 to 6 for the LoCoMo arm, USD 2 for BEAM. Pinned DeepSeek V4.1 Flash,
the USD 40 floor, two workers.

## Apparatus checks, fixed now

1. The calibration builds at least 90% of the 67 planned lists, each with 30 windows, the evidence
   pair on different days, and at least 20 cross-day pairs; otherwise z₀ is not fixed.
2. Render-only on every stored row, as for v1 and v2.
3. The reported mutual-nearest-neighbour pairs are recomputed by a second, independent loop in the
   report (brute force over all pairs) and must match exactly.
4. The unit tests of step 2 above, each seen red before green.

## What I already know

- v1 and v2 results above; v2's item vectors agree with the index to 0.9996.
- Voyage document vectors are close to deterministic at that level; query vectors are not (memory
  `voyage-query-embeddings-are-not-deterministic`), and only document vectors are used here.
- Reordering has a price when it is not gated (chronological order: temporal −0.13, extraction
  −0.17 on BEAM).

## Confounds I can name now

- **LongMemEval lists are not LoCoMo lists.** Their 28 filler windows come from unrelated synthetic
  sessions, so a real pair stands out there more than it will inside a two-person conversation. A
  list-relative z reduces this gap; it does not remove it. The mechanism gate exists for this.
- **The evidence pair can fail to be mutual nearest neighbours** even when it is the only
  same-subject pair, if a filler window happens to be closer to one side; this lowers positive
  recall and makes v3 conservative.
- **LongMemEval is one of AML's Textual sources**, used here only to fix one number.

<!-- FROZEN PREREGISTRATION ENDS HERE. APPEND RESULTS BELOW. -->

## Results

No measurement had run when this record was committed.

## Result, calibration (2026-09-25): stopped by the stop rule

**Status:** K-2 v3 stops at calibration. No LoCoMo or BEAM mechanism, no answers, no OpenRouter.

`python scripts/aml_k2v3_calibrate.py` at `509426a7` on VPS3, 67 one-user lists of 30 windows,
`voyage-code-4-v1` passage vectors. Report: `results/aml-k2v2/calibration-v3.json`.

| Metric | Measured | Predicted | Gap |
|---|---|---|---|
| Lists built and passing shape (check 1) | 67 of 67 | at least 90% of 67 | passes |
| MNN pairs recomputed by an independent loop (check 3) | identical on every list | required | passes |
| Spurious MNN pairs per list, before z₀ | 4.37 | | |
| **z₀** (at most 0.10 spurious links per list) | **3.961** (0.090 at z₀) | 2.5 to 4.0 | inside, at the top |
| Evidence pairs that are MNN at all | **0.761** (51 of 67) | 0.50 to 0.90 | inside |
| **Evidence pairs linked at z₀** | **0.164** (11 of 67) | 0.30 to 0.70; stop below 0.20 | **below the band; stop rule met** |

**What it means.** The old-and-new pair usually finds each other: three in four evidence pairs are
mutual nearest neighbours. What fails is standing out. The evidence pairs' z has a median of 2.78;
the chance MNN pairs between unrelated windows have a median of 2.29 and a 99th percentile high
enough that keeping false links to one in ten lists needs z₀ = 3.96, which only 16% of evidence
pairs reach. This is the easy case, with 28 filler windows from unrelated synthetic sessions; inside
one two-person conversation the evidence pair would stand out less, not more.

**Where K-2 stands after three versions.** v1 (word overlap) never fires, v2 (absolute cosine)
fires on everything, v3 (list-relative mutual neighbours) cannot separate a real pair from a chance
one at a tolerable false-link rate even on easy lists. The retrieval embedding does not carry
"same fact, updated" strongly enough to isolate such pairs among a list of related windows. The
idea behind K-2 (show the reader an old and a new statement side by side) is not refuted; this
signal cannot find the pairs. A next attempt would need a signal that names the fact, such as an
Add-time extracted (entity, attribute) key, which is the design the round-two plan first described
and which costs a model call per Add.

The v3 rule was drafted in `recall_aml/conflict_order.py` while calibration ran; it was reverted
unmerged, since nothing measures it and its tests had not been red-proved.
