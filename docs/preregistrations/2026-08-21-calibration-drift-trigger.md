# Pre-registration: can a manifest-level corpus delta decide when a calibration needs refitting?

**Date:** 2026-08-21   **Status:** MEASURED 2026-08-21 — see the Result section at the foot.
Every prediction below is unedited; the result was appended, never merged in.

## The question

Held fixed while its corpus evolves, how far can a certified abstention threshold be carried before
it stops separating, and does the manifest-level `corpus_delta` predict that point well enough to
be the trigger on its own?

Answerable by three numbers: the Spearman correlation between `corpus_delta` and the worst per-class
error of the frozen threshold; the smallest delta at which that error first crosses
`DEFAULT_MAX_CARRY_FORWARD_ERROR = 0.10`; and the precision of a delta-only trigger tuned to 90%
recall of the crossings.

## Why this is being asked

`recall/calibration_v2.py` now carries a threshold across a rebuild by re-scoring the parent's
stored query set (`carry_forward`), and it bounds how far it will try with
`DEFAULT_MAX_CORPUS_DELTA = 0.25`. Its own comment says the number is deliberately not tuned, and
the memory note `calibration-carry-forward-re-verifies-rather-than-rebinds` states the evidence base
in full: one tenant, one delta (9.31%), one embedder. So the repository has a mechanism whose
operating point is a guess.

There is also nothing that looks at a corpus **before** a rebuild. `resolve` answers `STALE` on an
exact fingerprint mismatch, which is a yes/no about identity rather than a statement about
magnitude, and `carry_forward` only runs once a new generation already exists. Nothing tells an
operator that the corpus under a live, still-passing calibration has drifted far enough to be worth
recalibrating. That is the feature this measurement is for.

## What I already know, and where it lives

- **`corpus_delta` and `threshold_error_rates`** exist and are the primitives this uses:
  `recall/calibration_v2.py:210` and `recall/calibration_v2.py:161`. Delta is
  `(added + removed + modified) / |union of source URIs|`, compared on `(uri, sha256)`.
- **One measured point.** `docs/preregistrations/2026-08-20-calibration-carry-forward.md`: at a
  10.8% delta on the `memory` tenant, separability CI low 0.9528 and false abstains **9.09%**
  against the 10% bound. Certified, and close to the edge.
- **Separability cannot see a shifted class**
  (`memory/separability-cannot-see-a-shifted-class.md`). A change that lifts every unanswerable
  score equally keeps AUC at 1.00 and slides the whole class over a fixed threshold. So the outcome
  here must be the per-class error of the fixed cut, never the AUC.
- **The classes overlap in 4 of 4 measured corpora**
  (`memory/calibrated-thresholds-and-the-overlap.md`), min(answerable) below max(unanswerable) every
  time. A zero-error bar is unreachable; only movement relative to the baseline is meaningful.
- **Additions and modifications move the error in opposite directions.** A top-1 cosine is a max
  over the indexed set, so additions can only raise scores (false confirms up) while modifications
  can lower them (false abstains up). The carry-forward record predicted false abstains could only
  fall and was falsified for exactly this reason.
- **Offline-generated query sets score higher than real questions**
  (`docs/preregistrations/2026-08-16-generated-calibration-query-sets.md`), because answerable
  queries are derived from chunk text and share its vocabulary.
- **I over-predict effect magnitudes**, eleven of twelve predictions falsified, every one too high
  by two to four times (`memory/i-over-predict-effect-magnitudes.md`). The bands below are set
  deliberately low, and each names a mechanism metric beside the outcome.

## What I predict

**P1. The association is real but weak: Spearman between `corpus_delta` and `max_error` lands in
0.30 to 0.60.** The mechanism is why. Top-1 cosine is a max over the index, so a change only moves a
query's score when it lands near that query. The off-topic class is drawn from subjects checked
disjoint from the corpus, so most additions cannot touch it, and most of a delta is therefore
invisible to the error. A delta is an upper bound on how much could have moved, not an estimate of
how much did.

**P2. Direction splits by corpus, and I name which way before looking.**

| corpus | dominant change | predicted dominant error |
|---|---|---|
| `docs/**/*.md` from git history | mixed add and edit | **false abstain** |
| `recall/**/*.py` from git history | edit heavy | **false abstain**, and the larger of the three |
| memory store, mtime ordered | almost pure addition | **false confirm**, with false abstains flat or falling |

Falsified if the memory corpus's dominant error is false abstain, which would mean something other
than the max-over-index argument governs.

**P3. `DEFAULT_MAX_CORPUS_DELTA = 0.25` is not conservative enough, and the code corpus breaks it
first.** Predict at least one snapshot with `corpus_delta < 0.25` and `max_error > 0.10`, and that
between **5% and 20%** of all snapshots below 0.25 are already over the error bound. The point is
not that 0.25 is wrong as a ceiling on the mechanism, which is what its comment claims for it, but
that it cannot be read as a safe distance, and today nothing in the tree measures the difference.

**P4. Delta alone is a poor standalone trigger: precision at 90% recall lands in 0.20 to 0.50.**
This is the prediction the design rests on. If it holds, the cheap manifest delta is a **screen that
decides whether to spend the probe replay**, not a substitute for it, and the feature has to be two
tiered. If precision comes out above 0.70 the two-tier design is unnecessary complexity and a single
delta threshold should ship instead.

**P5. The first crossing of `max_error > 0.10` happens at a delta between 0.10 and 0.30 on `docs`,
and does not happen at all on the memory corpus within its history.** The memory store grew from 2
files to 194 in fourteen days, which is a delta near 1.0 by this definition, so a null there would
be the sharpest possible statement that delta is the wrong unit.

## What would falsify this

- Spearman below 0.20 (delta carries no signal at all, and the screen in P4 is unbuildable) or above
  0.80 (delta is a good standalone trigger, and P4's two-tier design is over-engineering).
- No snapshot anywhere with `corpus_delta < 0.25` and `max_error > 0.10`. P3 is then wrong and 0.25
  looks defensible on this evidence.
- Precision at 90% recall above 0.70. P4 falsified, ship one threshold.
- The memory corpus's dominant error being false abstain. P2's mechanism is then wrong.
- Any corpus where `max_error` **falls** monotonically as delta grows. That would mean the harness is
  measuring something other than what it claims and the apparatus check below failed to catch it.

## How it will be measured

```bash
python -m benchmarks.calibration_drift --out results/calibration_drift_2026-08-21.json
```

**Corpora, three, chosen because their change modes differ:**

| id | source | snapshots | change mode |
|---|---|---|---|
| `docs` | `docs/**/*.md` at commits of this repository | ~24, evenly spaced over 432 commits | add and edit |
| `code` | `recall/**/*.py` at the same commits | ~24 over 573 commits | edit heavy |
| `memory` | the recall memory store, files revealed in mtime order | ~24 | append only |

**Per corpus:** chunk with the same function the index would use (`chunk_text` for prose,
`chunk_code` for Python), embed with `fastembed BAAI/bge-small-en-v1.5`, take the **exact** top-1
cosine per query in numpy. At the baseline snapshot t0, generate a labelled query set with
`recall.wizard.queryset.generate_offline(seed=0, per_class=40)` and fit the threshold with
`recall.calibration.from_samples`. At every later snapshot t, hold that threshold fixed and record:

- `threshold_error_rates(answerable_t, unanswerable_t, theta_0)`, each rate over its own class;
- `separability` and `separability_interval`, for the P3 comparison against certification;
- the refit threshold, which changes nothing and shows the drift before it is fatal;
- `corpus_delta(objects_0, objects_t)` on `(uri, sha256)` pairs, the branch's own definition;
- `evidence_survival`, the fraction of answerable queries whose originating chunk text is still
  present verbatim, so label rot can be told apart from retrieval drift.

**Primary metric:** `max_error = max(false_abstain_rate, false_confirm_rate)`, a rate over its own
class, thresholded at `DEFAULT_MAX_CARRY_FORWARD_ERROR = 0.10`. **n** is 3 corpora times roughly 23
non-baseline snapshots, so about 69 paired observations, 40 answerable and 40 unanswerable queries
behind each.

**Apparatus check, run before the corpora and reported with the result.** Two cases whose answer is
already known: re-scoring the baseline against itself must return `corpus_delta` exactly 0.0 and the
same error rates the fit reported in sample; and a synthetic snapshot with every source removed but
one must return a delta near 1.0. Exit code 0 is not a measurement.

## Confounds I can name now

- **Exact search, not HNSW.** The real path uses an ANN index whose builds are nondeterministic, and
  issue #26 measured coverage swinging 0.40 to 0.84 across rebuilds on one host. Exact top-1 removes
  that noise, so every number here is a **lower bound on the drift a real deployment sees**. Stated
  as a limitation, not corrected for.
- **Generated queries are easier than real ones.** Absolute rates will be optimistic. The claim is
  about the shape of the delta-to-error relation, not about the absolute error of any deployment.
- **Label rot is inside the false-abstain arm.** When the chunk an answerable query was generated
  from is edited away, the query is arguably no longer answerable, and the frozen threshold is
  blamed for it. `evidence_survival` is recorded so the two can be separated after the fact; it is
  not subtracted out, because an operator's query set rots the same way and that is part of what
  "the calibration needs refitting" means.
- **mtime ordering is a reconstruction.** The memory corpus has no git history, so a memo edited
  later appears later. It is a plausible growth order, not the true one.
- **One embedder.** bge-small only. Thresholds do not transfer across embedders and neither,
  necessarily, does the shape of this curve.
- **The three corpora are all this project's own.** Every one is technical prose or Python by the
  same authors, so the off-topic pool is disjoint from all three in the same way.

## A note on line numbers below this rule

Every `path:line` citation **above** this line is part of the registered prediction and is never
updated, even when the tree moves under it. `docs/citation-policy.toml` carries the exemption that
stops a gate demanding the edit. Citations below this rule are maintained normally.

## Result (2026-08-21)

**Status:** measured. Nothing above this line has been edited.

Command, from this worktree, at commit `c4c37b67`:

```bash
python -m benchmarks.calibration_drift --snapshots 24 \
  --memory-root ~/.claude/projects/C--Users-gde00-Documents-recall/memory \
  --out results/calibration_drift_2026-08-21.json
python -m benchmarks.calibration_drift --analyze results/calibration_drift_2026-08-21.json
```

Apparatus check passed on all three cases. **n = 57** paired observations (20 `docs`, 19 `code`,
18 `memory`), 40 answerable and 40 unanswerable queries behind each.

### What was measured

| corpus | baseline | snapshots | delta range | over the 0.10 bound | smallest delta over it |
|---|---|---:|---|---:|---:|
| `docs` | 13 sources, 364 chunks, threshold 0.673, AUC 0.9869 | 20 | 0.375 to 0.981 | **4 of 20** | **0.945** |
| `code` | 29 sources, 165 chunks, threshold 0.651, AUC 0.9938 | 19 | 0.579 to 0.979 | **1 of 19** | **0.979** |
| `memory` | 43 sources, 175 chunks, threshold 0.680, AUC 0.9838 | 18 | 0.157 to 0.778 | **0 of 18** | never |

### Scoring the predictions

| | predicted | measured | verdict |
|---|---|---|---|
| **P1** Spearman, delta vs `max_error` | 0.30 to 0.60 | **0.384** pooled | **confirmed** |
| **P2** dominant error, `docs` | false abstain | false **confirm** leads 17 of 20 | **falsified** |
| **P2** dominant error, `code` | false abstain, largest of the three | false **confirm** leads 14 of 19; smallest of the three | **falsified** |
| **P2** dominant error, `memory` | false confirm, abstains flat or falling | false confirm leads 11 of 18 | **confirmed** |
| **P3** a snapshot under delta 0.25 over the bound | yes; 5% to 20% of those below 0.25 | **0**, and only **1 of 57** observations is below 0.25 at all | **untestable, not falsified** |
| **P4** precision at 90% recall | 0.20 to 0.50 | **0.714 in sample**, **0.500 out of sample** | see below |
| **P5** first crossing on `docs` | delta 0.10 to 0.30 | **0.945** | **falsified** |
| **P5** `memory` never crosses | yes | 0 of 18, reaching delta 0.778 | **confirmed** |

### P4, and a flaw in the analysis I registered

Taken at face value P4 is falsified: the pooled best cut at 90% recall is delta **0.945** with
precision **0.714**, above the 0.70 line I registered as "ship one threshold instead".

**That number is not a measurement, and the fault is in the analysis I specified rather than in the
result.** The cut is chosen and scored on the same 57 points, so its precision is an in-sample
optimum. It is the same defect `recall/calibration.py` documents at length for `best_threshold`
("the cheapest way to keep every answerable sample above the boundary is to put the boundary
exactly ON the lowest one"), and the same one FINDINGS section 2b retracted a published number for.
I should have registered a held-out rule and did not.

Leave-one-corpus-out, fitting the cut on two corpora and scoring on the third:

| held out | cut fitted elsewhere | fires | true | positives | precision | recall |
|---|---:|---:|---:|---:|---:|---:|
| `docs` | 0.979 | 1 | 1 | 4 | 1.000 | **0.250** |
| `code` | 0.945 | 3 | 1 | 1 | **0.333** | 1.000 |
| `memory` | 0.945 | 0 | 0 | 0 | never fired | nothing to catch |
| **pooled** | | 4 | 2 | 5 | **0.500** [0.15, 0.85] | **0.400** |

Out of sample the precision is **0.500**, at the top of P4's predicted band, and the recall
collapses to **0.400** against the 0.90 the cut was tuned for. The `docs` fold is the one that
matters: a cut fitted on the other two corpora catches **one of four** real failures. So a
delta-only trigger does not transfer between corpora, and P4's conclusion stands on evidence its
own registered number did not supply.

### What this changed in the code

- **The delta-only `RECALIBRATE_REQUIRED` route was removed from `recall/drift.py`.** A rule at
  0.25 fires on **56 of 57** snapshots and is right about **5**: precision **0.09**. It would have
  demanded recalibration on twenty consecutive states of `docs/` where the threshold was measurably
  fine, several with a *lower* error than the day it was fitted. Where a probe can run the probe
  decides; where it cannot, the strongest verdict is a recommendation.
- **`DRIFT_SCREEN_DELTA` stays at 0.05, restated as what it is:** a cost decision with a nineteen
  fold margin under the smallest observed failure, not a tuned optimum. Firing costs one probe;
  staying quiet costs a silent failure.
- **`DEFAULT_MAX_CORPUS_DELTA` was NOT changed.** This measurement holds a threshold frozen, which
  is carry-forward's semantics, and it says 0.25 is conservative by a wide margin. It is left alone
  anyway: one repository, one embedder, generated queries, exact rather than approximate search,
  and only one observation below 0.25. What is established is that a delta is a poor alarm, not
  that a large delta is safe.

### The mechanism, which is the part that generalises

False **confirm** is what moves, in all three corpora (17 of 20, 14 of 19, 11 of 18), and it moves
with corpus **growth**:

| corpus | Spearman, delta vs growth | Spearman, growth vs false confirm | growth over the history |
|---|---:|---:|---|
| `docs` | 0.979 | 0.953 | 1.4x to 7.6x |
| `memory` | 1.000 | 0.879 | 1.2x to 6.3x |

A top-1 cosine is a max over the indexed set, so an added document can only raise an unanswerable
query's score, never lower it. How much of a corpus was *rewritten* predicts nothing about that.

⛔ **All three corpora grew several-fold over their histories, so this measurement cannot separate
"the corpus changed" from "the corpus got bigger".** That is the sharpest limitation here and it is
not fixable by re-analysis: it needs a corpus that churns without growing. Until somebody measures
one, read every delta number in this record as a growth number wearing a delta's name.

### The labels were far more durable than the argument for a delta bound assumed

`evidence_survival` is the fraction of answerable queries whose original top-scoring chunk still
exists verbatim.

| corpus | survival at the last snapshot | false abstain there |
|---|---:|---:|
| `docs` | **0.275** | 0.025 |
| `code` | **0.400** | 0.000 |
| `memory` | 1.000 | 0.050 |

Three quarters of `docs`'s original evidence was gone and the false-abstain rate was 0.025. The
questions kept working long after the specific text that first answered them did, which is the
direct refutation of "past some delta the labelled set describes a corpus that no longer exists".

### Deviations from the registered method, all disclosed

1. **The `code` arm did not run on the first attempt.** `generate_offline` refused it: until
   2026-08-18 the off-topic subject pool lived in `recall/eval/synthetic.py` as source, so a code
   corpus rooted at this repository contains the word list its own unanswerable queries are drawn
   from. That is the collision the generator's own refusal predicts, and its own advice is to
   exclude the path. `recall/eval/synthetic.py` is excluded and the arm re-ran. No result was
   discarded, because the arm had produced none.
2. **`excess_max_error` was added before the corpora ran, not after.** The smoke run put the
   baseline's own in-sample false-abstain rate at 0.100, exactly on the bound, before any drift.
   That is an apparatus fact rather than a result, and it means an absolute bar can be met by a
   calibration on the day it is fitted. Both outcomes are reported; the registered one is first.
3. **The baseline is the first snapshot with at least 160 chunks**, not the first snapshot. Both
   git histories begin at a commit holding one file, and a threshold fitted to a 15-chunk corpus is
   not a calibration anybody would deploy. 3, 4 and 5 snapshots were dropped respectively, and the
   counts are in the result file.
4. **Leave-one-corpus-out was not registered.** It was added because the registered analysis fits
   and scores on the same points, which is a flaw in the registration. The registered number is
   reported first and in full.

### Confounds that survived, restated

Exact search rather than HNSW, so these are lower bounds on the drift a real deployment sees.
Generated queries score higher than real ones, so the absolute rates are optimistic. One embedder.
The memory corpus's order is an mtime reconstruction. And the growth confound above, which is the
one that most limits what any of this supports.
