# Pre-registration: does annotating evidence with self-declared successor markers change the answer?

**Date:** 2026-08-15   **Status:** predicted, not yet measured

Filed at `results/enterprise_rag/` rather than the `docs/preregistrations/` default, following this
repository's established convention: a pre-registration lives beside the results it predicts, as
`results/truth_extraction/PREREGISTRATION-prose-extraction.md` and
`results/enterprise_rag/PREREGISTRATION-library-parity.md` both do.

## Registration

```yaml
registration_commit: 4f0a8c83a199367f1db9eb4ffd257902a7eb8573
registration_authored: 2026-08-15T20:14:54+00:00
frozen_evidence_digest: 70715fcd64de564ac1fea1ffe54d90458265efaa0543be9f1a56fe425464d2f0
frozen_anchors_digest: dcba3d7338d40937eee56c393fd0cb44de7792e8afedbe68860f59d6254eb560
```

The two digests are the apparatus half. The evidence bundles and the fact anchors are frozen
BEFORE either arm runs, and both arms read the same bytes; if either digest moves between arms the
comparison is void.

## The question

On the four EnterpriseRAG rows whose conflict is a superseded value against its replacement, does
annotating each evidence item as `current` or `superseded` change the **fact-anchor hit rate** of
the generated answer?

## What I predict

**Metric.** `fact_anchor_hit_rate` = anchors present in the answer ÷ anchors defined for that row.
**The denominator is anchors, not rows**, and it is roughly 40 anchors across 11 rows.

**Arms.** Both read the same frozen evidence and use the same cheap model at temperature 0 and the
same unmodified `SYSTEM_PROMPT`.

- **C**, control: evidence items rendered as they are today.
- **S**, supersession: each item additionally carries a library-authored
  `supersession_status` field and the marker phrase that decided it, inside the delimited data
  region.

| # | quantity | point | interval |
|---|---|---|---|
| S1 | C arm hit rate, the 4 supersession rows | 0.45 | 0.20 to 0.70 |
| S2 | **Δ hit rate, S minus C, the 4 supersession rows** | **+0.15** | 0.00 to +0.35 |
| S3 | **Δ hit rate, S minus C, the 6 coverage rows (negative control)** | **0.00** | −0.05 to +0.05 |
| S4 | A pairs the annotator marks | **3 of 4** | exactly 3 |
| S5 | B rows the annotator marks | **0 of 6** | exactly 0 |
| S6 | answer length ratio, S ÷ C | 1.15 | 1.0 to 1.4 |

**S4 names `qst_0419` as the expected miss**, in advance. Its successor says only "the updated
requirements" while the OLDER document carries `legacy` and `replace`. Naming it now is what stops
a 3-of-4 result being retold afterwards as a clean success.

**Ordering predictions**, harder to hit by luck than the levels:

- **O1.** Every A row whose hit rate improves is a row where the annotator FIRED. If a row improves
  without having been annotated, the effect is not supersession and S2 is measuring something else.
- **O2.** The S arm loses no anchor that the C arm found, on any A row. An annotation that trades
  old facts for new ones is not an improvement, it is a different answer.

## What would falsify this

- **S3 fails**, the six coverage rows move by more than 0.05 either way. They contain one document
  each and have no sibling to supersede, so a change there means the annotation is acting
  non-specifically, most likely by making answers longer. **Publish no claim about supersession.**
- **S5 fails**, the annotator marks a single-document row. That is a detector bug and the run is
  void until it is fixed.
- **O1 fails.** The aggregate could move for reasons unrelated to the mechanism, and the per row
  attribution is the only thing that separates the two.
- **S2's interval includes 0 after the run**, which at n=4 rows it very well may. Then the honest
  report is "could not tell from four rows", not "no effect".

⚠️ **The most likely uninformative outcome, declared live rather than hypothetical:** four rows and
roughly sixteen anchors cannot separate +0.15 from 0.00 with any confidence. **This is a mechanism
probe, not an effect size.** If it shows the mechanism firing and moving the right rows in the
right direction, that earns a larger run. It cannot itself justify shipping anything.

## How it will be measured

- **Dataset.** EnterpriseRAG-Bench v1.0.0, 11 questions: A = `qst_0418`, `qst_0419`, `qst_0420`,
  `qst_0425`; B = `qst_0310`, `qst_0320`, `qst_0325`, `qst_0332`, `qst_0333`, `qst_0336`.
  `qst_0413` is run and reported but **not predicted**: it is an attribution conflict, a third
  mechanism, and one row.
- **Evidence.** Retrieved ONCE from the `ber_voy_lex_12k_full` index on VPS2 under the submitted
  configuration, then frozen to a fixture and digested. Every subsequent iteration reads the
  fixture, so the index leaves the loop and the probe is reproducible from the fixture alone.
- **Model.** A cheap model, pinned by id and recorded in the artifact, temperature 0, one call per
  row per arm. 22 calls total.
- **Scoring.** Mechanical. No LLM judge. Anchors are literal strings extracted by hand from
  `answer_facts`, frozen and digested before the run; matching is case-insensitive over
  whitespace-normalised text, and that normalisation is fixed now rather than tuned later.
  Negative facts ("must not invent `/v1/capacity/migration/start`") score as absence checks.
- **Human read.** At n=11 I will also read all 22 answers, because the one fact the anchors cannot
  score is framing: "must not present the legacy endpoint as primary" is about emphasis, not
  presence.

### Apparatus checks, run before the arms

Predicting the outcome does not reveal a broken harness, so each of these has a known answer:

| # | check | known answer |
|---|---|---|
| A1 | the scorer on a hand-written answer containing every anchor for a row | 1.0 |
| A2 | the scorer on an empty answer | 0.0 |
| A3 | the scorer on the C arm's *previously judged wrong* answers | > 0.0 and < 1.0, since these rows are wrong but not empty |
| A4 | the annotator on `probe_reasoning_reach.MEMO_SHAPED` | fires |
| A5 | the annotator on a single-item bundle | does not fire, there is no sibling |
| A6 | the frozen evidence digest, before and after both arms | unchanged |

## What I already know

Searched before predicting, so this is not a re-measurement.

- 🔑 **`benchmarks/probe_reasoning_reach.py`**, measured today: the SHIPPED contradiction detector
  returns zero proposals on an EnterpriseRAG-shaped conflict, and when it does fire it
  `_fail_closed`s to an abstention. It cannot win a correctness row, which is why this experiment
  proposes a new mechanism rather than wiring the old one.
- 🔑 **Step 0**, in `SCOPE-conflict-resolving-reasoning.md`, measured today: all four A pairs carry
  in-text supersession markers, and the markers are DIRECTIONAL, with the successor announcing
  itself. The signal is in the text, not in metadata, which is why the mechanism goes in the answer
  layer.
- **`FINDING-where-the-deficit-actually-is.md`**: all 11 rows are answer-control failures with the
  gold documents FULLY retrieved. Retrieval is not the constraint on these rows, which is what makes
  them the right target.
- **All 11 are currently judged WRONG** under `gpt-5.4 medium`. So the direction of "better" is
  known before the run, which is itself a bias worth naming.
- **The shipped nine-word status vocabulary matches 8 of 8 documents positively.** Any mechanism
  keyed on it would fire everywhere; this one is keyed on directional successor language instead.

## Confounds I can name now

1. **Prompt volume.** The S arm's prompt is strictly longer, and a longer prompt can produce a
   longer answer that hits more anchors without any reasoning having occurred. **S3 and S6 exist
   for this**, and S3 is the sharper of the two: the coverage rows get no annotation, so if they
   move, volume is the explanation.
2. **I wrote the anchors after reading the gold facts.** Unavoidable, since the anchors come from
   `answer_facts`. Mitigated by freezing and digesting them before either arm runs, and by never
   deriving an anchor from any arm's output.
3. **Literal matching flatters neither arm equally.** If the S arm's annotation nudges the model
   toward quoting the evidence verbatim, it hits literal anchors more often for a formatting reason.
   The human read is the check on this.
4. **Single run, n=4.** No variance estimate. A second seed would not fix n=4.
5. **I already know these rows are wrong**, so I am predicting an improvement on rows selected for
   being failures. Regression to the mean is available as an explanation for any small positive.

## What this does not settle

- **The six coverage rows.** They are a different mechanism and are here only as a negative control.
- **Anything about the benchmark score.** Eleven rows, no judge, no aggregate.
- **Whether the trust layer should change.** It demotes; these rows need retention with a label.
  Untouched by design.

---

## Apparatus note, 2026-08-15: the freeze, and two deviations it forced

**Appended before either arm ran. No prediction above is edited.** These are facts about the
apparatus, discovered while building it, and the pre-registration's own rule is that predictions
are never revised after measuring; recording what the instrument turned out to be is the opposite
of that.

`frozen_evidence_digest` is now filled: **`70715fcd…`**, 11 rows, 8 hits each. It verifies on a
second machine with no index and no network, which is what apparatus check A6 needs.

### Deviation 1: SPLADE runs on CPU, the submitted run used CUDA

VPS2 has no GPU. `prune_to_top_k` takes a hard top-k over SPLADE weights, so device differences in
the last bits flip terms across the pruning boundary and the candidate pool changes. **Forced, not
chosen.**

### Deviation 2: `--rerank-document-chars` is 3900, the submitted run used 4000

At 4000 the Voyage reranker refuses the batch: the fused pool is 604,210 tokens against a 600,000
ceiling, 0.7% over. I swept the setting and measured its effect on whether the gold documents
survive into the frozen evidence, which is the property the experiment actually depends on:

| `rerank_document_chars` | qst_0418 | qst_0419 | qst_0420 | qst_0425 |
|---|---|---|---|---|
| 2000 | 2/2 | 1/2 | 1/2 | **0/2** |
| 3500 | 2/2 | 1/2 | 2/2 | 2/2 |
| **3900, adopted** | **2/2** | **1/2** | **2/2** | **2/2** |

⚠️ **At 2000 the fixture would have been useless and it would have looked fine.** `qst_0425` had
NO gold document at all, so both arms would have scored near zero on it and the null would have
been read as "the annotation does not help" when the truth is "the evidence was not there". The
check that caught this cost nothing and is the reason it is worth doing before the arms, not after.

### The consequence for `qst_0419`, stated now

**`qst_0419` carries 1 of its 2 gold documents at every setting tried.** It cannot be recovered by
tuning, so the remaining cause is deviation 1 or ordinary reranker nondeterminism. This matters
beyond detection: with one of the two documents absent, **the conflict may not be present in the
frozen evidence at all**, and a row with no conflict cannot demonstrate conflict resolution.

`qst_0419` was already the named expected miss for S4, on the separate grounds that its successor
says only "the updated requirements". It now has two independent reasons to miss. **S4's
prediction of "exactly 3 of 4" is unchanged**, and the analysis will report `qst_0419` separately
so the two reasons are not conflated with each other or with a genuine null.

### What this does not damage

Both arms read the same frozen bytes, so the A/B comparison is unaffected by either deviation. What
the deviations cost is the claim that these rows reproduce their ORIGINAL failure: three of four
do, on gold presence, and `qst_0419` does not.

---

## Apparatus note 2, 2026-08-15: the anchors are frozen and three checks have run

`frozen_anchors_digest` is filled: **`dcba3d73…`**. **48 scorable anchors** across the 11 rows,
39 positive and 9 negative, plus **12 facts recorded as unanchorable** with a reason each. The
pre-registration estimated "roughly 40"; the denominator is 48, and it is fixed now.

The anchor file is **not committed**, on the same rule as the evidence fixture: 48 literal strings
drawn from a live benchmark's answer key are answer-key material. The digest is published and
`benchmarks/fact_anchors.py` carries the schema, so it can be rebuilt.

### Checks A1, A2, A3, A6: passed

| check | expected | measured |
|---|---|---|
| A1, complete answer | 1.0 | 1.0 |
| A2, empty answer | 0.0 | 0.0 for positives; the row rate is 0.25 because an absent NEGATIVE anchor is vacuously satisfied, which is stated rather than tuned away |
| A3, previously judged-wrong answers | strictly between 0 and 1 | **0.583 overall** (28/48) |
| A6, fixture digest recomputed elsewhere | unchanged | unchanged on a second machine, no index, no network |

A4 and A5 test the annotator, which does not exist yet. They run before the arms.

### What A3 shows beyond passing, and it is the encouraging part

The misses land exactly where the judge said the answers were wrong, which is the strongest
evidence available that the anchors measure the intended thing:

- **`qst_0418`, 5/9.** Every v2 threshold present, and `old_t1_80`, `old_t2_60_79`, `old_t3_30_59`
  all MISSING. The submitted answer gave the new values and omitted the superseded ones. That is
  precisely the supersession-synthesis failure the annotation is meant to address.
- **`qst_0420`, 1/5.** `gib_based`, `egress_rate` and `sampled_bytes` all missing, which is the
  answer having reported the OLD token-based model.
- **`qst_0425`, 3/5.** `integrity_values` and `integrity_ref` missing.

⚠️ **These are apparatus numbers, not S1.** The control arm is a fresh generation from the frozen
evidence under `SYSTEM_PROMPT` and a cheap model; the submitted answers came from a different
prompt, a different model and a different retrieval. The A-row figure of 0.542 is a prior, not a
measurement of the C arm, and S1 stays as registered.

---

## Result (2026-08-15)

**Status: measured.** 22 calls, `openai/gpt-4o-mini`, temperature 0, both arms over the frozen
evidence `70715fcd…` and the frozen anchors `dcba3d73…`. No judge. **No prediction above is
edited.**

### The verdict: the probe cannot tell, and it says so for a reason it also measured

🔑 **Two of seven UNANNOTATED rows returned different answers between the arms.** On those rows the
annotator did not fire, so both arms sent a byte-identical prompt to the same model at temperature
0. **Temperature 0 is not determinism.** One of those rows, `qst_0320`, moved 0.33 to 0.00, which
is a larger swing than the entire measured effect.

So the measured S2 delta of **+0.042** sits inside a noise floor this run demonstrated at ±0.33 on
a single row. Re-running the same arm twice would produce a different delta. This is the
uninformative outcome the pre-registration declared live rather than hypothetical, and it is worse
than anticipated: the obstacle is not only n=4, it is that the instrument is not repeatable.

### Score

| # | registered | measured | verdict |
|---|---|---|---|
| S1 | C hit rate, A rows: 0.45, [0.20, 0.70] | **0.875** | **FALSIFIED**, far above |
| S2 | Δ A rows: +0.15, [0.00, +0.35] | **+0.042** | inside the interval, but see the ceiling below |
| S3 | Δ B rows: 0.00, [−0.05, +0.05] | **−0.050** over 6; **−0.056** over the 5 unannotated | at/over the boundary, and caused by nondeterminism |
| S4 | A pairs annotated: exactly 3 of 4 | **3 of 4** | **CORRECT**, and the miss is `qst_0419`, named in advance |
| S5 | B rows annotated: exactly 0 of 6 | **1 of 6** | **FALSIFIED**, and the premise was mine |
| S6 | length ratio S÷C: 1.15, [1.0, 1.4] | **0.907** | **FALSIFIED**, answers got SHORTER |
| B3 | validation failures: 0% to 5% | **0 of 22** | correct |
| O1 | only annotated rows move | only `qst_0420` moved, and it fired | **HELD** |
| O2 | S loses no anchor C found | none lost on any A row | **HELD** |

### What was wrong, and why, taken one at a time

**S1 is the most informative failure.** I predicted the control would score 0.45 because these rows
were judged wrong. It scored **0.875**, against the SUBMITTED answers' 0.542 on the same anchors.
The library path over frozen evidence is far better on anchored facts than the harness that
produced the judged-wrong answers. ⚠️ **That undercuts the probe's premise**: if the control already
carries 87.5% of the anchored facts, then whatever made these rows wrong is mostly NOT a missing
anchored fact, and an instrument built from those anchors cannot see the thing that is broken.

**S2's interval was impossible.** With C at 0.875 the maximum achievable delta is +0.125, so the
registered interval's upper half could never have been reached. A prediction should not be able to
be unreachable by construction; the fix is to predict against the HEADROOM, not the raw rate.

**S5's premise was mine, not the detector's.** I wrote that the B rows "contain one document each
and have no sibling to supersede". That is their GOLD count. Every retrieved bundle holds about
eight documents, so the annotator can fire on any of them, and on `qst_0310` it did. Apparatus
check A5, which tests a genuinely single-document bundle, passes. The detector obeyed its rule; the
prediction rested on a confusion between gold-document count and bundle-document count.

**S6 went the wrong way**, as O1 in the parity pre-registration also predicted wrongly for a
different arm. Twice now I have predicted that removing length guidance lengthens answers, and
twice the answers got shorter.

### The one encouraging signal, stated with its width

**O1 held.** The only row whose score moved is `qst_0420`, and it is a row where the annotator
fired, and it moved **+0.20** in the predicted direction. It is also the row whose successor
document says, in prose, *"This is a working doc intended to supersede the older Confluence page"*.
The other two annotated rows were already at 1.00 and had no room to move.

⚠️ **One row is not evidence of an effect.** A single unannotated row moved −0.33 on an identical
prompt in this same run. O1 holding is consistent with the mechanism working and equally consistent
with one lucky draw.

### What I would change before spending anything further

1. **Repeat each arm n times and measure the nondeterminism directly**, rather than assuming
   temperature 0 gives repeatability. That is now a known property of this apparatus and it should
   be a pre-registered quantity, not a discovery.
2. **Predict against headroom**, never against a raw rate that may already sit near 1.0.
3. **Pick rows where the control actually fails the anchors.** S1 says the four A rows are mostly
   already anchored-correct under this substrate, which makes them the wrong rows for this
   instrument.
4. ⚠️ **Reconsider the target entirely.** `ANALYSIS-where-reasoning-could-help.md`, written after
   this was registered, segments all 500 rows and finds this cell is worth **3.2 aggregate points**
   while retrieval-side reasoning is worth **21.6**. This probe was scoped before that analysis
   existed, it has now run, and its result does not argue for extending it.
