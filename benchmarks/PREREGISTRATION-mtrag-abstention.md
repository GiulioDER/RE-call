# Pre-registration — MTRAGEval: can an abstention detector recover the oracle bound?

**Date:** 2026-08-05 · Written **before** any measurement. The git history of this file is the
evidence. Predecessor: [`PREREGISTRATION-mtrag-rbalg.md`](PREREGISTRATION-mtrag-rbalg.md), verdict
in [`results/mtrag/PROBE_VERDICT.md`](../results/mtrag/PROBE_VERDICT.md). Neither is edited.

**Prior work searched** — `docs_search(source_type="memory")` on "abstention detector unanswerable
classifier threshold false abstention retrieval score entailment signal". `gap_warning: false`,
top-3 cosine 0.671 / 0.666 / 0.645. **The top hit did not decorate this design, it deleted its
first draft.**

- 🛑 [[project-recall-abstention-bounded-domain-2026-07-24]] (PR #79, `fe5979c`): **six candidate
  signals were already measured on the same 500 questions and all failed.** AUC: dense cosine
  **0.753** (what ships), cross-encoder rerank 0.742, RRF fusion 0.739, QNLI entailment 0.648,
  margin 0.579, ratio 0.545. Best in-sample balanced error (0.271) does not beat the shipped
  threshold's 0.305. Two **named dead ends, not to be re-proposed without new evidence**:
  retuning the threshold (already at its ROC ceiling) and using a stronger judge (moves along the
  curve, does not lift it). Independently reproduced 2026-07-28 on a third corpus (cosine 0.78,
  entailment 0.59).
- [[project-recall-entailment-supersession-phase0-done-2026-07-18]]: the entailment stage is
  judge-bounded and must stack on the threshold, not replace it.
- [[project-recall-campaign-db-and-calibration-wiring-gap-2026-08-05]]: the promotion harness
  **cannot consume a calibration at all** (`StoreSearch` built without `calibration=`, no
  `--calibration` flag, `generation=None`).

**What that ruled out.** The obvious preregistration here was "build an abstention detector from
retrieval features and measure how much of the +0.0595 oracle bound it recovers". That is the
**closed question**, on three benchmarks. It is not in this file. Also removed: any dependence on
RE-call's certified-trust path, because a calibration authored today is an artifact nothing reads.

**What survives, and why it is not a re-litigation.** The same memo names the one remedy it did
*not* test: *"The only remaining remedy is a genuine reading step (an LLM call)"* — explicitly an
evidenced claim rather than an assumption, "since six cheaper signals were tried first". It was
never run because RE-call's product constraint is **$0 and no LLM in the retrieval path**. **That
constraint does not apply to MTRAGEval**, which the user has scoped unconstrained, and MTRAGEval is
the first setting where the payoff for getting abstention right is exactly measurable.

## 0. The claim this file tests

From the probe, on MTRAG's 842 tasks:

**On an UNANSWERABLE task the conditioned score is binary.** I first inferred this from mean
`rb_agg` equalling each model's correct-IDK rate exactly, for all nine models. **A mean is
consistent with that claim without proving it** — a spread of partial values averages to the same
number — so the **joint** distribution was checked before this file was committed
(`probe/probe_binary_check.py`). Marginals would not have been enough either: three counts of
72/423 are equally consistent with three *disjoint* sets of 72.

> 495 UNANSWERABLE model cells. The tuple `(rl_f, rb_llm, rb_agg)` takes exactly **two** values:
> **(1.0, 1.0, 1.0) on 72 cells and (0.0, 0.0, 0.0) on 423.** Nothing else.

There is no partial credit, and the three metrics move together rather than coincidentally. A
correct IDK scores **exactly 1.0** on all three simultaneously; anything else scores **exactly
0.0**.

⚠️ **Qualifier, because the unqualified sentence would be false.** This is a property of the
answerability-**conditioned composite**, which is the metric MTRAGEval ranks on. The underlying raw
components are continuous on the very same cells (RougeL is strictly between 0 and 1 on 478 of the
495). The binariness is the conditioning gate, not the similarity.

Perfect abstention is therefore worth **+0.0595** mean harmonic mean on MTRAG. That is an **oracle**
bound.

> **How much of it survives a detector that does not get to see the answerability label?**

## 1. Two arms, in this order, and the first one is allowed to end the experiment

**Arm R — which regime is MTRAG?** One signal only: dense cosine separability between the
answerable and unanswerable classes, computed with the **shipped** `from_samples` /
`separability` machinery (Mann-Whitney AUC, threshold-free, so it cannot be inflated by fitting and
scoring on the same samples).

⛔ **The other five signals are NOT re-run.** They are closed. Running them would be the 2026-07-28
mistake again: four hours and a re-derivation of a result already written down.

Arm R is not a re-measurement, because the same memo establishes that **the near-miss regime is a
property of how the questions were built** ("BEAM's inversion is adversarial, not general"). Which
regime a *new* corpus sits in is therefore an open question about that corpus, not a re-run.

**Arm L — the reading step**, run **only if Arm R lands in the near-miss regime.** An LLM is asked,
per task, from the conversation and the retrieved passages alone, whether the passages answer the
question. This is the remedy the prior work named and did not test.

## 2. Predictions, committed before measuring

**P1 — MTRAG is the near-miss regime.** Dense-cosine separability AUC between answerable and
unanswerable is **≤ 0.80**, i.e. materially closer to the 0.753 / 0.78 near-miss figures than to the
far-gap regime where the distributions are disjoint (accuracy 1.00 / 0.89).
*Falsified if* AUC ≥ 0.90, which is also the shipped `certified` threshold. **A falsified P1 is the
good outcome**: it would mean the cheap shipped signal already works here and Arm L is unnecessary.

**P2 — the reading step clears the bar the cheap signals could not.** Arm L reaches balanced error
**< 0.271**, the best in-sample figure any of the six cheap signals achieved.
*Falsified if* ≥ 0.271, which would mean the named remedy is not a remedy either and the whole
abstention lever is unreachable, not merely unbuilt.

**P3 — recovered fraction of the oracle bound ≥ 50%.** Applying Arm L's decisions at the derived
threshold (§3) recovers **≥ +0.0298** of the +0.0595, measured as the actual change in mean
harmonic mean, not inferred from classifier metrics.
*Falsified if* < +0.0149 (25%).

**P4 — false abstention is the binding cost, and it will bite.** Arm L's false-abstention rate on
ANSWERABLE tasks is **> 0**, and the net gain in P3 is smaller than the gain computed from true
positives alone. The prior work's near-miss failure was a **0.481 false-abstain rate while hit@5
was 0.970** — it refused half the questions it had just answered correctly.
*Falsified if* false abstention is exactly 0, which would be surprising enough to warrant
suspecting the harness before believing it.

## 3. The threshold is DERIVED, not tuned

Retuning a threshold is dead end #1 in the prior work. This file does not tune one. It computes the
break-even from the measured payoffs:

```
E[abstain] = p · 1.0     + (1-p) · 0          # correct IDK scores 1.0; false abstain scores 0
E[answer]  = p · 0.0     + (1-p) · a          # answering an unanswerable scores exactly 0
break-even: p* = a / (1 + a)
```

with `a` = the mean score on the tasks where answering pays, **measured only on cells where the
model actually answered**. That qualifier is the whole derivation: a cell where the model abstained
is the *other* branch of the decision, and the conditioning gate scores it exactly 0.0, so folding
it into `a` contaminates the answer branch with the abstain branch.

| `a` measured over | value | p\* |
|---|---|---|
| **ANSWERABLE + PARTIAL, answering** (**primary**) | **0.4199** (n=6675) | **0.2957** |
| ANSWERABLE only, answering | 0.4265 (n=6146) | 0.2990 |
| ~~ANSWERABLE only, including abstentions~~ | ~~0.4108~~ (n=6381) | ~~0.2912~~ |

**The struck row is what this file said in its first draft**, and it was wrong for the quantity §3's
own words define. A `bug-auditor` pass caught it before commit; the corrected figure was already
sitting in [`PROBE_VERDICT.md`](../results/mtrag/PROBE_VERDICT.md) §1 as subset (d). The error
biased `p*` **low**, which would have made the detector abstain more readily than break-even and
biased the run *against* P3.

**PARTIAL is in the primary population on evidence, not by assumption**: abstaining on a PARTIAL
task scores exactly 0.0 on all 83 of its abstained cells, exactly as on ANSWERABLE, so answering is
what pays there too. **CONVERSATIONAL (90 cells, 1.2%) is excluded**: every one of its cells scores
(1.0, 1.0, 1.0) regardless of the response, so it is payoff-neutral and would only dilute `a`.

The threshold will be computed this way per metric and on the harmonic mean, from the probe's own
artifacts, and **reported before the detector's outputs are scored against it**.

⚠️ **Base-rate hazard, stated now.** UNANSWERABLE is **6.5%** of MTRAG and **19.1%** of MTRAG-UN, a
2.93× difference. A detector whose operating point is chosen at the dev base rate is at the wrong
point on the test set. The threshold will be **recomputed for the target base rate**, and any
transfer claim must state both.

## 4. Leakage firewall

`Answerability` (MTRAG) and `answerability` (MTRAG-UN `reference.jsonl`) are **ground truth for
scoring only**. They must not reach Arm L's prompt, its features, or any selection step. The
MTRAGEval task page states this metadata was withheld from participants; it is in the public
release, which makes the firewall a code property to enforce rather than a courtesy.

Two further label-side fields found in `RAG.json` and **equally barred from inference**:
`Retriever Performance Characteristics` (takes the value `NO_RELEVANT_PASSAGES_EXIST`, which is the
answer to the question being asked) and `# Relevant Passages`.

If Arm L is ever *trained* rather than prompted, the training split must be disjoint from the
scored split, and the split must be declared here before it is used.

## 5. Scope and what this cannot claim

- **Dev is MTRAG (842 tasks). MTRAG-UN is held out and is not touched by this file.**
- Nothing here produces a Task C submission. It measures one component in isolation.
- `RAG.json` licenses **paired within-file comparison only**; no number here may be stated as parity
  with a published leaderboard figure (see the predecessor's §0).
- **No dependence on the certified-trust path.** It is blocked on a harness feature that does not
  exist, and this experiment is designed to run without it. If that changes, a strict-trust arm is a
  separate pre-registration.
- The prior work's conclusion is **not** re-opened. If P2 falsifies, the correct entry in
  `closed_hypotheses_index.md` is that the reading step failed too, and the abstention lever on this
  workload is closed rather than pending.
