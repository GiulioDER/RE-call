# Pre-registration: can a manifest-level corpus delta decide when a calibration needs refitting?

**Date:** 2026-08-21   **Status:** predicted, not yet measured

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
