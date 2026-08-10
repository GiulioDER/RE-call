# Pre-registration — MTRAGEval: is RB_alg length-bound?

**Date:** 2026-08-05 · Written **before** any measurement. The git history of this file is the
evidence. Related: [`benchmarks/mtrag/README.md`](../../mtrag/README.md) (Task A adapter, frozen arms),
archived Task A baseline `2026-08-04-mtrag-symmetric-baseline` on VPS2. Neither is edited.

**Prior work searched** — `docs_search(source_type="memory")` on "MT-RAG generation RB_alg RougeL
response length calibration answer verbosity benchmark". `gap_warning: false`, top-3 cosine
0.588 / 0.584 / 0.573. Two load-bearing hits, and the first **changed this design rather than
decorating it**:

- [[project-recall-token-f1-harness-offset-2026-07-29]]: on LOCOMO, "mean answer 7.8 (RE-call) vs
  9.0 (Mem0) words against gold 4.9, median 3 both". So answer-length-versus-gold is already an
  established axis in this repo, it was already found to be a *checkable confound*, and RE-call's
  own outputs already run ~1.6x the gold length. That is a prior in favour of H1, and it is also
  the reason §4's causal step exists: the same memo treated verbosity as something to *rule out*,
  not to exploit, so nothing here can lean on it as evidence for a lift.
- [[project-recall-mtrag-symmetric-baseline-2026-08-04]]: the Task A six-arm baseline. Retrieval
  only. Says nothing about generation metrics.

**Nothing was found on RB_alg, on MTRAGEval generation, or on length calibration against a
reference answer.** This is new ground.

## 0. The claim this file tests

MTRAGEval ranks Tasks B and C by the harmonic mean of `RL_F`, `RB_llm` and `RB_alg`. From
MTRAG-UN Table 4, in the RAG (Task C) setting, across thirteen baseline models:

| | human reference | best model | worst model |
|---|---|---|---|
| RL_F | 0.69 | 0.60 | 0.46 |
| RB_llm | 0.92 | 0.65 | 0.52 |
| **RB_alg** | **0.88** | **0.38** | **0.23** |

A harmonic mean is dominated by its minimum, and RB_alg is the minimum for every model in the
table. The published SemEval-2026 Task 8 leaderboard tops out at **0.586** (GenAIus, Task C), and
every top-3 system was zero-shot.

`run_algorithmic.py:rb_agg` computes RB_alg as the harmonic mean of three terms:

```
recall         = (BertScoreR + 1) / 2                # rescale_with_baseline, deberta-xlarge-mnli
extractiveness = (max(BertKPrec) + 1) / 2            # same rescaling
rouge          = RougeL_stemFalse                    # raw LCS F-measure
```

Two of the three are BERTScore values rescaled against a baseline and then mapped from [-1, 1]
into [0, 1]. Rescaling centres typical values near 0, so `(x+1)/2` puts them near 0.5. RougeL is a
raw F-measure and is not rescaled. RougeL is an LCS F-measure, so it is penalised by *both*
directions of length mismatch.

> **Is RB_alg low because the models are wrong, or because they are long?**

If it is length, the metric is recoverable by re-expression at no cost in correctness, and the
field's 0.586 ceiling is an artefact of nobody having looked.

## 1. The apparatus, and the invariant that validates it

Source: `mtrag-human/evaluations/RAG.json` from the MT-RAG repo at revision
`cc5b1d481b391181b89f7ced860308482e785463`. 842 tasks × 10 models = 8420 published evaluations,
each carrying `model_response` and per-metric `system` (raw) and `composite` (conditioned) values.

**These are IBM's own published baseline outputs.** No generation is required for §2 and §3, so
those cost nothing and introduce no model of mine into the measurement.

**Invariant, asserted before running.** Recomputing the per-model harmonic mean of the composite
`rl_f`, `rb_llm` and `rb_agg` from `RAG.json` must reproduce the published MTRAG Task C table to
within ±0.01:

| model | published Task C harmonic mean |
|---|---|
| Target (reference) | 0.81 |
| GPT-4o | 0.53 |
| Llama-3.1-405B-Instruct | 0.53 |
| Qwen-2.5 (72B) | 0.52 |
| Llama-3.1-70B-Instruct | 0.52 |
| GPT-4o-mini | 0.51 |
| Command-R+ (104B) | 0.51 |
| Qwen-2.5 (7B) | 0.51 |
| Mixtral-8x22B-Instruct | 0.48 |
| Llama-3.1-8B-Instruct | 0.45 |

**If this does not reproduce, the apparatus is wrong and nothing below counts.** Exit code 0 is not
a measurement. I am checking the artefact, which is the reproduced table, not that the script ran.

## 2. Predictions, committed before measuring

**P1 — models are long.** Median token-length ratio (prediction / target) across non-target models
is **> 2.0**.
*Falsified if* the median is ≤ 1.5.

**P2 — RougeL is the binding term.** RougeL is the minimum of the three RB_alg components in
**> 70%** of unconditioned instances.
*Falsified if* ≤ 50%.

**P3 — RB_alg falls with length.** Mean RB_alg among instances in the top quartile of length ratio
is **at least 0.10 lower** than among instances whose length ratio is nearest 1.0.
*Falsified if* the gap is < 0.03, or has the opposite sign.

**P4 — a correct IDK scores exactly 1.0.** For an UNANSWERABLE task answered with an IDK response,
the composite `rb_agg`, `rb_llm` and `rl_f` are each exactly **1.0**, not merely high.
*Falsified if* any correct-IDK composite is < 0.99.

**P5 — the causal claim, and the one that matters.** Re-expressing a model's response at the
reference length while preserving its content raises mean RB_alg by **≥ +0.10 absolute**.
*Falsified if* < +0.05.

**What P5 would be worth.** Starting from gpt-oss-120b's published Task C row (RL_F 0.59,
RB_llm 0.65, RB_alg 0.37, harmonic mean 0.505), and holding the other two metrics fixed, RB_alg
must reach **0.53** for the harmonic mean to reach 0.586 and clear the leaderboard's top score.
0.53 is still 0.35 below what the human reference scores on the same metric.

## 3. What P3 does NOT establish, stated before I see the number

P3 is **observational and confounded**. A long answer may be long because the question demanded a
long answer, in which case length is a proxy for question type and not a cause of the score. A
confirmed P3 raises the hypothesis; it does not license the claim.

**Only P5 is causal**, because it holds the question, the passages and the content fixed and varies
only the expression. If P3 confirms and P5 falsifies, the honest conclusion is that RB_alg tracks
question type, the exploit does not exist, and this line closes. I am recording that outcome as a
real possibility now so that it cannot be reinterpreted later.

## 4. Method for P5

Stratified sample from the 842 tasks, balanced across the four domains and across answerable versus
unanswerable. For each sampled instance, take the **published** response of one fixed baseline
model, then produce a re-expressed variant under an instruction that constrains length and register
to the reference distribution and **forbids adding, removing or altering any claim**. Score both
variants with the unmodified `run_algorithmic.py`. Paired comparison, same instances, bootstrap CI.

**Content preservation is the load-bearing assumption and it will be checked, not asserted.** If
the re-expression silently drops or invents claims then a RB_alg rise is measuring damage, not
calibration. `RL_F` is a reference-less faithfulness judge against the passages, so it is the
control: **RL_F must not fall.** A rise in RB_alg accompanied by a fall in RL_F is a failed probe,
not a win, and will be reported as such.

## 5. Leakage firewall, declared before any code runs

`mtragun-human/generation_tasks/reference.jsonl` ships a per-task `answerability` label
(ANSWERABLE 285, UNANSWERABLE 97, UNDERSPECIFIED 78, PARTIAL 47) together with `Question Type` and
`Multi-Turn`. The MTRAGEval task page states this metadata was **withheld from participants**.

It is used in this probe for **stratification and diagnostics only**. It must never reach an
inference path, because the IDK conditioning gate is exactly what that label answers, and a system
that reads it is measuring nothing. Any later abstention work must obtain answerability from the
conversation and passages alone, and the firewall must be a property of the code rather than a
claim in a document.

Related, and equally binding: the SemEval-2026 evaluation window closed in February 2026 and the
MTRAG-UN labels are public. Everything measured here is post-hoc with the test set visible. **MTRAG
(the 842-task human set) is the dev set for this probe. MTRAG-UN is held out and is not touched.**

## 6. Scope

This file covers the probe only. It does not preregister a pipeline, a submission or a paper. If
P5 falsifies, the result is published as a null and this line closes.
