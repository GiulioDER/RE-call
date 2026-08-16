# Pre-registration: can a generated labelled query set certify a calibration on a real corpus?

**Date:** 2026-08-16   **Status:** MEASURED 2026-08-16 — see the Result section at the foot.
Prediction below is unedited.

## The question

Does a labelled query set produced automatically from a corpus, with no human labelling, yield a
**certified** calibration artifact on that corpus, and does the offline structural generator differ
from the LLM generator in the threshold it produces?

Answerable by: certified yes/no per arm, plus the AUC 95% lower bound and the fitted threshold per
arm.

## Why this is the load-bearing measurement for the wizard

Calibration binds to a generation and requires ≥20 answerable and ≥20 unanswerable labelled
queries. Nothing in the repository generates them; `recall setup` asks for a path to a file the user
must have written. That is the step that makes calibration too hard to be worth doing, and the
Windows wizard is not deliverable unless this step can be automated honestly.

## What I predict

**Both arms certify.** Neither is close to the floor.

| | offline structural | LLM (`openai/gpt-5.6-luna`) |
|---|---|---|
| certified | yes | yes |
| AUC 95% lower bound | **0.97 to 1.00** | **0.92 to 0.98** |
| fitted threshold | **0.70 to 0.80** | **0.55 to 0.70** |

**The offline generator scores higher, and that is a defect rather than a win.** Its answerable
queries are derived mechanically from chunk text, so they share surface vocabulary with the chunk
they came from and retrieve it at an inflated cosine; its unanswerable queries come from a fixed
maximally-disjoint domain list. Both ends are easier than reality. A threshold fitted to that set is
set too high, and at serving time it will abstain on genuine user questions that a correctly fitted
threshold would answer.

**Specific prediction, and the one I care about: `threshold_offline − threshold_llm ≥ 0.05`.**

If that holds, the wizard must prefer the LLM generator wherever a key or a local endpoint exists,
and must say plainly that the offline fallback buys privacy at the cost of an over-abstaining
threshold. If it does not hold, the offline generator is a first-class path and the API key page
becomes optional convenience rather than a quality lever.

## What would falsify this

- Either arm fails certification (AUC 95% lower bound < 0.90, or fewer than 20 usable per class
  after canonicalization). That would mean generated sets cannot replace hand labelling at all.
- `threshold_offline − threshold_llm < 0.05`, including the case where the LLM threshold is the
  higher of the two. That falsifies the inflation claim and the product decision resting on it.
- The two AUC lower bounds are indistinguishable, i.e. their intervals overlap substantially.
- The LLM arm's "unanswerable" queries turn out to be answerable by the corpus often enough to
  depress its AUC. That is a falsification of the *method*, not of the comparison, and is called
  out as a confound below because it would produce a low LLM AUC for the wrong reason.

## How it will be measured

Corpus: this repository's `docs/`, 50 markdown files (measured 2026-08-16 by
`recall manifest inventory docs`, corpus fingerprint `f26a39a6…`).

Embedder: `fastembed` bge-small, 384 dim. Chosen because it is deterministic; Voyage query
embeddings are not (42.5% of repeat calls differ), which would make no fixture reproducible.

```bash
eval "$(scripts/session-db.sh up)"
recall --migration-dsn "$DSN" schema --dim 384 apply
recall manifest inventory docs --output inventory.json
recall --tenant docs manifest create --corpus-version 2026-08-16 \
  --objects inventory.json --output manifest.json
RECALL_LOCAL_ALLOWLIST="$PWD/docs" recall --tenant docs generation build manifest.json \
  --chunker text --embedder-revision <pinned bge-small sha>
recall --tenant docs generation validate <gen>
# arm A and arm B, 40 per class each
recall --tenant docs calibration calibrate --generation <gen> --queries <arm>.json
```

n = **40 answerable and 40 unanswerable per arm**, twice the floor of 20, chosen so the
Hanley–McNeil lower bound is not the binding constraint on a set that is genuinely separable.

Metrics, by name:
- **certified**: the boolean `Calibration.certified`, which is `AUC_lower_bound ≥ 0.90` AND
  `≥20 samples per class`.
- **AUC 95% lower bound**: the lower bound of the two-sided Hanley–McNeil interval, not the point
  estimate. This is the number certification actually tests.
- **threshold**: midpoint of `q05(answerable top cosine)` and `q95(unanswerable top cosine)`.
- Rates are over the 40 queries of the relevant class, stated as `k/40`.

## Apparatus verification, before trusting either arm

Predicting an outcome does not reveal a broken harness, and exit code 0 is not a measurement.

1. **A known-answer case.** Run the shipped `recall/eval/queries.json` (20 answerable / 20
   unanswerable, hand-labelled, exactly at the floor) against the same generation. It is a
   hand-built set on a corpus it was not written for, so it is not required to certify; what is
   required is that it *runs*, reports 20 per class, and produces a threshold in a plausible range.
   If it errors or reports different counts, the harness is wrong and neither arm means anything.
   Note: its 6 `trust` entries have no `answerable` key and v2's loader refuses them, so they must
   be stripped first. That stripping is itself a thing to verify rather than assume.
2. **A negative control.** Feed a set whose two classes are drawn from the same distribution
   (40 answerable queries split arbitrarily and half relabelled unanswerable). It MUST fail
   certification. If a set with no real signal certifies, the certification path is not testing
   what it claims and every number here is void.
3. **Determinism.** Run arm A twice. The query set is generated deterministically and fastembed is
   deterministic, so the two artifacts must agree on threshold to within floating-point noise.

## What I already know

- `recall/eval/synthetic.py:69-75` records the measurement that constrains the offline design.
  Building an unanswerable query by suffixing a nonsense token onto an answerable one produced a
  set that was **not separable at all**: median top cosine **0.830** against answerable **0.923**,
  and **0%** of it fell below the weakest answerable query. Genuinely off-topic questions sit at
  median **0.570** with **78%** below the answerable floor. This is why the offline generator draws
  unanswerable queries from a disjoint domain rather than perturbing corpus queries, and it is the
  basis for predicting a high offline AUC.
- `MIN_CALIBRATION_SAMPLES = 20` per class and `MIN_SEPARABILITY = 0.90`
  (`recall/calibration.py:44,61`), applied to the interval's lower bound.
- The shipped `recall/eval/queries.json` has exactly 20/20 — zero margin — and the repository's
  root `calibration.json` is a preserved **rejected** artifact (1 sample per class).
- Memory `calibration-belongs-at-install-time`: this is the user's stated direction from
  2026-08-08. `DEFAULT_GAP_THRESHOLD = 0.50` is untuned and sits at the 0th percentile of five of
  six measured top-1 distributions, and the 16th of the sixth. So any fitted threshold materially
  above 0.50 is already an improvement on the shipped constant, for either arm.
- `docs/CALIBRATION.md:169-176`: "A certified calibration means the binding is exact and the
  statistics were computed on a labelled set; it does not mean the labelled set was a good one."
  That sentence is precisely the risk this measurement exists to quantify, and it is why
  certification alone is not the outcome being predicted.

## Apparatus note, written 2026-08-16 before any measurement

Recorded here rather than in a commit message because it changes how the offline arm's result
should be read, and it must be on the record before the number exists.

The offline generator's first implementation ranked candidate terms by document frequency alone.
Run against this repository's `docs/` (1809 chunks) it produced:

```
why was doubly sources_not_found fell decided this way
what is the behaviour of id_rsa promo_ uuid8
how is launching test_bench_conversation_indices advertised handled here
```

Every one of those tokens is genuinely rare and none is a topic. A question built from an
identifier retrieves the single chunk containing that symbol, which would have produced excellent
separability while measuring string matching rather than retrieval. Measuring that would have
measured my implementation, not the approach.

Changed before measuring: terms are ranked by `tf * log(N/df)` rather than rarity alone, so a word
used five times in a chunk beats a word used once; symbols are excluded (`_`, digits, length > 24);
and a markdown heading, where the chunk has one, is preferred over any inferred term. Output now
reads like `what does this project say about truncated registry conforming` and
`where is retrieval profiles budget bounded cost described`.

**This is where tuning stops.** The remaining awkwardness is the genuine limit of a semantics-free
generator on prose, and it is the thing the prediction above is about. Further polishing would be
choosing the result.

Also measured while checking the apparatus: only **13 of the 25** shipped off-topic subjects are
disjoint from this corpus (penguins, espresso, coral, harpsichord and eight others appear in it),
giving **65** distinct gap questions. So 65 is this corpus's hard ceiling on `per_class`, the
default of 40 sits below it with modest headroom, and 66 is refused with both numbers named. A
corpus overlapping more of the pool caps lower and is refused rather than served a gap class that
shares its vocabulary — which is the right failure, but it means **the size of the shipped
off-topic pool, not the corpus, is what bounds how large a generated query set can be.** Widening
that pool is the obvious lever if a thin set turns out to be what limits certification.

Sampling reliability, measured over 300 seeds on the same corpus after the fix: 0/300 failures at
`per_class=20` and 0/300 at `per_class=40`, against 1.1% and 2.3% before it. Output is identical
across a re-run, so the determinism the comparison depends on holds.

**A third apparatus defect, caught before measuring and worth recording because it would have
invalidated the offline arm outright.** The oversampling fix above iterated the pool in sorted
order and stopped at `per_class`, so only its lowest-indexed quarter was ever examined. Measured on
`docs/`: the 40 answerable questions came from **3 of 51 files**, all alphabetically first, over
chunk indices 2..513 of 1813. That is the head-of-corpus bias the sampling exists to prevent,
reintroduced by the fix for a different problem, and a threshold fitted to three documents would
have been reported as a threshold for the corpus. Iterating in sampled order instead: **21 of 51
files**, indices 82..1741. The offline arm's number should be read as covering the corpus only
because of this change.

## Confounds I can name now

- **The LLM writes an "unanswerable" question the corpus does answer.** recall's `docs/` covers
  retrieval, calibration, trust, generations, MCP and deployment, so a plausible-sounding
  off-topic question can land inside it. This depresses arm B's AUC for a labelling reason rather
  than a generator-quality reason, and would look identical to the LLM generator being worse.
  Mitigation: report, for each unanswerable query, whether its top cosine exceeded the answerable
  5th percentile, and inspect those by hand before drawing the comparison.
- **The LLM quotes the chunk verbatim.** If arm B's answerable questions copy chunk phrasing, arm
  B becomes arm A and the contrast collapses to noise. Mitigation: measure mean token overlap
  between each answerable query and its source chunk in both arms, and report it alongside.
- **The offline generator's advantage may be corpus-specific.** `docs/` is one technical corpus
  with heavy shared vocabulary. A conclusion drawn here may not transfer to a user's prose corpus.
  This measurement bounds the claim to this corpus and does not license a general one.
- **n = 40 per class is small.** The Hanley–McNeil interval is wide at this n, so two arms can
  differ in point estimate while their intervals overlap. The threshold difference, not the AUC
  difference, is the primary comparison for exactly this reason.
- **A generation built in development is marked unverified unless a revision is pinned.** If the
  pinned bge-small revision is wrong the build refuses, which is a loud failure; but if it silently
  falls back to `--unverified-development` the artifact still certifies while binding to something
  production would refuse. Check the generation's `verified` flag before reading any result.

---

## Result (2026-08-16)

**Status:** measured

Corpus: this repository's `docs/`, 52 objects, 1793 chunks in generation
`gen_af48372b781d40f5b6c8db5b8cefb26e`, `fastembed` bge-small, 384 dim, n = 40 per class per arm.

| | ARM A offline | ARM B LLM | predicted A | predicted B |
|---|---|---|---|---|
| certified | **yes** | **yes** | yes | yes |
| threshold | **0.7050** | **0.6620** | 0.70–0.80 | 0.55–0.70 |
| AUC | 0.9806 | 1.0000 | — | — |
| AUC 95% lower bound | **0.9496** | **1.0000** | 0.97–1.00 | 0.92–0.98 |
| median cosine, answerable | 0.751 | 0.784 | — | — |
| median cosine, gap | 0.629 | 0.554 | — | — |
| gap below the weakest answerable | 36/40 | 40/40 | — | — |

### The headline prediction is falsified, in magnitude and in direction

Predicted: `threshold_offline − threshold_llm ≥ 0.05`. **Measured: +0.043.** The sign is right and
the size is not, so the claim as registered fails.

The more interesting failure is the reasoning behind it. I predicted the offline generator would
score **higher** on separability, because its answerable queries are built from its own chunk's
words and its gap class is maximally disjoint, and that this apparent strength would be an artefact
inflating the threshold. **The opposite happened on every metric.** The LLM arm separated perfectly
(AUC 1.000, all 40 gap queries below the weakest answerable) while the offline arm did not
(0.9806, 36/40), and the offline arm's AUC lower bound came in **below** its predicted range while
the LLM's came in **above** its own.

Why the reasoning was wrong: a bag of three distinctive terms is not semantically close to the
chunk it came from. Offline answerable queries median **0.751** against the LLM's **0.784**, so the
model's fluent questions retrieve their own material *better* than term-bags do, not worse. And the
offline gap class is drawn from a 13-subject list that survives the corpus filter, so it is
narrower and closer to the corpus (median 0.629) than the LLM's freely-written questions (0.554).
The offline generator is weaker at both ends, not artificially strong at either.

### What this changes

The product decision is unchanged but for a different reason than registered. **Prefer the LLM
generator whenever a key or a local endpoint exists**, and say plainly that the offline fallback
buys privacy at the cost of a thinner, noisier set — not, as predicted, at the cost of an
over-abstaining threshold.

Both arms certify, so the wizard's central claim holds: **a generated set can replace hand
labelling on a real corpus.** That is the question this was written to answer, and the answer is
yes for both generators.

### Apparatus checks, all three as registered

1. **Known-answer.** The shipped `recall/eval/queries.json` ran after 6 `trust` entries were
   stripped, reported 20/20, and produced threshold 0.707 with AUC 0.868 [0.752, 0.983]. Rejected
   for falling under 0.90, which is the expected outcome for a set written against a different
   corpus, and the point is that it ran and reported plausible numbers.
2. **Negative control.** Forty offline answerable queries split arbitrarily, half relabelled
   unanswerable: AUC **0.495 [0.314, 0.676]**, rejected. The certification path does not certify
   noise, which is what makes the two arms above meaningful.
3. **Determinism.** `generate_offline(seed=0)` produced byte-identical output across a re-run.

### Confounds, measured rather than asserted

- ⚠️ **Arm B's perfect separation is partly manufactured by my own filter.** Of 80 gap questions
  the model produced, **40 were dropped** for reusing the corpus's subject vocabulary, leaving
  exactly 40. The dropped ones are by construction the *closest* to the corpus, so the surviving
  gap class is the easy half. AUC 1.000 should be read as "1.000 after the borderline gap questions
  were removed", not as a property of the model's output. A fair rerun would keep every gap
  question the model labelled and measure what that costs.
- **This also validated the margin.** `asked = per_class * 2` left exactly 40 survivors. The
  earlier `per_class + max(5, per_class // 2)` would have yielded 60 asked, ~30 surviving, and the
  arm would have failed outright rather than measured.
- **Answerable-side overlap is not what distinguished the arms.** Corpus-word overlap was 1.00
  (offline) against 0.95 (LLM), essentially the same, so the LLM did not win by quoting less. Mean
  question length 8.0 words against 11.8.
- **An AUC of exactly 1.000 makes its own interval degenerate.** The reported bound [1.000, 1.000]
  is what Hanley–McNeil gives for perfect separation at n=40/40; it is not evidence that separation
  would be perfect on unseen questions.
- **The generation is `verified: false`** (`unverified_reason: "explicit development build"`,
  `revision: null`). There is **no pinned revision for bge-small anywhere in the tree**, so
  `--unverified-development` was the only way to build. Production would refuse this generation.
  That is a real gap for the wizard, not a detail of this run: as things stand the wizard cannot
  produce a verified generation with the default embedder.
- **One corpus, one embedder.** Nothing here licenses a claim about a prose corpus or another model.
