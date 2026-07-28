# Prediction — §9c entailment ROC sweep, re-measured post-#81/#84

Written and committed **before** the run starts, per [`docs/RESEARCH_PROTOCOL.md`](../../docs/RESEARCH_PROTOCOL.md).
Scored afterwards for whether it was right, and right *for the right reason*.

## Why this run exists

`FINDINGS.md` §9c is the two-judge ROC sweep. Its retained artifact,
`results/locomo_entailment_sweep.json`, carries `_provenance.generation: "pre-#81/#84"` and
`superseded_by: null` — it measured an effectively **dense-only** configuration (the sparse leg
ANDed every query term; the unfiltered dense scan capped near `hnsw.ef_search=40`). §9b was
re-measured post-fix; §9c was not. This run produces the successor.

## Diagnosis of the previous attempt, before predicting anything

An earlier post-fix attempt died and left no artifact. `FINDINGS.md` records §9c as "not
re-measured", which is true and undersells it. What actually happened, from
`results/locomo/postfix_entailment_sweep.log` (gitignored, worktree-local, so this is the only
place the diagnosis is retained):

- It completed **9 of LOCOMO's 10** conversations — conv-26, 30, 41, 42, 43, 44, 47, 48, 49 — and
  died on **conv-50**, the last one.
- Only **one** judge banner appears (`qnli-distilroberta`), against `DEFAULT_JUDGES` of two. So it
  died inside the *first* judge's pass and never reached `qnli-electra-base`.
- stderr was captured (the HF Hub warning is in the log) and there is **no traceback**.
- `recall/eval/locomo_entailment_sweep.py` has no `try`/`except` around the conversation loop, so a
  raised exception could not have been swallowed — it would have printed and exited.

**Conclusion: the process was killed externally, not failed internally.** A killed process leaving
no JSON is correct behaviour, not a bug.

**The defect it does expose is different: the sweep has no incremental persistence.** It ran ~90%
of a multi-judge pass and lost everything, including the nine conversations it had already scored.
`benchmarks/salvage.py` exists on master for exactly this on the head-to-head harness; this runner
has no equivalent. Recorded as a finding; **not** fixed in this cycle, which adds no capability.

## Prediction

**Best separation** (adversarial-abstain − answerable false-abstain, the only quantity that
matters since either column alone is gameable) stays within **±0.03** of the recorded values:

| judge | recorded (pre-fix) | predicted post-fix |
|---|---|---|
| `qnli-distilroberta` (shipped) | 0.197 | 0.167 – 0.227 |
| `qnli-electra-base` (stronger) | **0.240** | 0.210 – 0.270 |

And the ordering holds: electra > distilroberta.

## Reasoning

Separation is a property of the **judge's** ability to tell "Caroline realized X" from "Melanie
realized X", not of which candidates reach it. A better candidate pool hands the judge a
*better-retrieved* on-topic-but-wrong turn, which scores **higher**, not lower. §9b measured
precisely this and it did not move: post-fix discrimination came in at **0.154** against a pre-fix
**0.157**, while both columns rose together.

Two independent corroborations that this is a judge bound rather than a retrieval bound:
§10b measured the same judge's answerability AUC at **0.648**, *below* plain cosine (0.753); and
the newly-published §9o reports on a different corpus that "the entailment guard does not
discriminate".

## Decision rule, fixed in advance

- **Within ±0.03 on both judges** → §9c's conclusion stands. Update the configuration note, mark
  the row `current`, and register the successor.
- **Outside ±0.03 in either direction** → a retrieval fix moved an answerability judgement, which
  contradicts §9b. **Publish neither number** until the disagreement is resolved; a result that
  contradicts a measured sibling is a signal about the apparatus, not a finding.
- **Ordering flips (distilroberta > electra)** → treat as apparatus failure, not as a result. The
  two judges are scored in the same pass on the same data; a flip means the pass is not measuring
  what it claims.

## The invariants this run asserts in code, recorded before it runs

So they cannot be adjusted afterwards to match the outcome:

1. `conversations == 10` in the emitted JSON. The previous attempt reached 9; a run that reports 9
   is the same failure wearing a summary line.
2. `corpus_rows` is present and non-zero (this branch's provenance block), and equals the count a
   clean single run produces.
3. Both judges appear in the report. One-judge output is a partial run, not a sweep.

**Exit code 0 is not a measurement.** All three are checked before any number is read.
