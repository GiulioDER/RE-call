# Pre-registration: does annotating evidence with self-declared successor markers change the answer?

**Date:** 2026-08-15   **Status:** predicted, not yet measured

Filed at `results/enterprise_rag/` rather than the `docs/preregistrations/` default, following this
repository's established convention: a pre-registration lives beside the results it predicts, as
`results/truth_extraction/PREREGISTRATION-prose-extraction.md` and
`results/enterprise_rag/PREREGISTRATION-library-parity.md` both do.

## Registration

```yaml
registration_commit: PENDING
registration_authored: PENDING
frozen_evidence_digest: PENDING
frozen_anchors_digest: PENDING
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
