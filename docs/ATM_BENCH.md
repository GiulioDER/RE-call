# ATM-Bench: personal memory QA, scored by the benchmark's own evaluator

> **The one-line summary: the official evaluator scores this run at QS 68.4264 <!--@ atm/atm_bench_full_20260821.json # official_score.qs_percent -->
> and Recall@10 92.8924 <!--@ atm/atm_bench_full_20260821.json # retrieval.recall_at_10_percent --> on the full 1,013 <!--@ atm/atm_bench_full_20260821.json # question_count -->-question split, which would be first on both columns of the public
> board. Three things stop that from being a clean "state of the art" claim, and all three are
> below: the answer model is not matched to the baselines, the judge ran over a non-official
> transport, and the leaderboard pull request is open rather than merged.**

This document is under the claim gate, so every RE-call figure in it resolves to a committed
artifact and CI checks the digits against it. That artifact is
[`results/atm/atm_bench_full_20260821.json`](../results/atm/atm_bench_full_20260821.json), which
also carries the checksums of the offline package the figures come from.

Three classes of number here cannot resolve, and each says so in the source rather than passing
quietly: the **maintainers' own leaderboard rows** in §3, which are their published figures
and not a RE-call measurement; the **judge-transport comparison**, whose per-question output is not
retained here; and the **replay behind §5**, which is reproducible at zero cost from the
archived package but which no committed artifact yet holds.

---

## 1. What ATM-Bench is

[ATM-Bench](https://github.com/JingbiaoMei/ATM-Bench)
([arXiv:2603.01990](https://arxiv.org/abs/2603.01990), leaderboard at
[atmbench.github.io](https://atmbench.github.io)) is a long-horizon **personal referential memory**
QA benchmark: 1,013 <!--@ atm/atm_bench_full_20260821.json # question_count --> questions over 11,034 <!--@ atm/atm_bench_full_20260821.json # corpus_items --> personal memory items drawn from email, images and video,
asked in the first person about the owner's own past ("How much did I pay in total for my
accommodation for BMVC 2024?").

Two properties make it the right third opinion for a memory layer.

1. **The corpus is a person's memory, not a document collection.** Answering needs the right item
   out of eleven thousand near-identical receipts, itineraries and confirmations, which is the
   failure mode a memory layer exists to handle and which a topical IR benchmark does not test.
2. **Half of the score never touches an LLM judge.** The headline metric, **QS**, is three
   different metrics stacked by question type:

   | type | n | how it is scored |
   |---|---:|---|
   | `number` | 360 <!--@ atm/atm_bench_full_20260821.json # official_score.by_qtype.number.count --> | exact multiset equality against the gold value |
   | `list_recall` | 139 <!--@ atm/atm_bench_full_20260821.json # official_score.by_qtype.list_recall.count --> | Jaccard overlap over gold evidence IDs |
   | `open_end` | 514 <!--@ atm/atm_bench_full_20260821.json # official_score.by_qtype.open_end.count --> | an LLM judge, officially `gpt-5-mini` |

   So **499 <!--@ atm/atm_bench_full_20260821.json # deterministic_question_count --> of 1,013 <!--@ atm/atm_bench_full_20260821.json # question_count --> questions are scored deterministically**. That half is free to re-measure and
   noiseless to compare, which is what made the diagnosis in §5 possible at zero API cost.

The board reports two columns: **QS** and **Recall@10**, the fraction of questions for which at
least one gold evidence item is in the retrieved top 10 <!--@ atm/atm_bench_full_20260821.json # config.retrieval_k -->.

---

## 2. The result

Full split, 1,013 <!--@ atm/atm_bench_full_20260821.json # question_count --> questions, run 2026-08-21. QS and the per-type rows are the official evaluator's
`atm_openai_gpt-5-mini_summary.json`, unedited.

| Measure | Result |
|---|---:|
| **Official QS** | **68.4264** <!--@ atm/atm_bench_full_20260821.json # official_score.qs_percent --> |
| QS, `number` | 72.7778 <!--@ atm/atm_bench_full_20260821.json # official_score.by_qtype.number.qs_percent --> |
| QS, `list_recall` | 59.8270 <!--@ atm/atm_bench_full_20260821.json # official_score.by_qtype.list_recall.qs_percent --> |
| QS, `open_end` | 67.7043 <!--@ atm/atm_bench_full_20260821.json # official_score.by_qtype.open_end.qs_percent --> |
| **Recall@10** | **92.8924** <!--@ atm/atm_bench_full_20260821.json # retrieval.recall_at_10_percent --> |
| Recall@10GT (all gold evidence in the top 10 <!--@ atm/atm_bench_full_20260821.json # config.retrieval_k -->) | 86.9694 <!--@ atm/atm_bench_full_20260821.json # retrieval.recall_at_10gt_percent --> |
| Joint@10 (QS times Recall@10) | 63.5629 <!--@ atm/atm_bench_full_20260821.json # joint_at_10_percent --> |
| Questions answered | 1,013 <!--@ atm/atm_bench_full_20260821.json # answer_count --> of 1,013 <!--@ atm/atm_bench_full_20260821.json # question_count --> |
| Blank answers | 0 <!--@ atm/atm_bench_full_20260821.json # blank_answers --> |

The evaluator's separate LLM-only summary reads 694 <!--@ atm/atm_bench_full_20260821.json # llm_only_diagnostic.weighted_correct --> of 1,013 <!--@ atm/atm_bench_full_20260821.json # question_count --> = 68.5094 <!--@ atm/atm_bench_full_20260821.json # llm_only_diagnostic.qs_percent -->. It is a diagnostic and is
**not** substituted for the QS score, which is the board's metric.

Both retrieval figures above were **recomputed for this document** from `retrieval.jsonl` against
the released ground truth, independently of the harness that produced them, and reproduce the
submitted values to four decimal places.

### Configuration

| | |
|---|---|
| Embedder | `voyage:voyage-4-large` |
| Reranker | `voyage:rerank-2.5` |
| Sparse leg | lexical (Postgres FTS) |
| Candidate pool, then final | 25 <!--@ atm/atm_bench_full_20260821.json # config.candidate_k -->, then 10 <!--@ atm/atm_bench_full_20260821.json # config.retrieval_k --> |
| Evidence budget | 8,192 <!--@ atm/atm_bench_full_20260821.json # config.evidence_chars --> characters |
| Answer model | `deepseek/deepseek-v4-pro` through OpenRouter, reasoning requested `medium` |
| Answer accounting | 1,082 <!--@ atm/atm_bench_full_20260821.json # usage.calls --> provider calls, 69 <!--@ atm/atm_bench_full_20260821.json # config.truncation_retry_count --> truncation retries, 2,886,847 <!--@ atm/atm_bench_full_20260821.json # usage.total_tokens --> tokens |
| Index | isolated table and tenant; the production index was not touched |

The candidate pool is 25 <!--@ atm/atm_bench_full_20260821.json # config.candidate_k --> rather than larger because larger Voyage pools exceeded the observed
rerank token-per-minute project limit, not because 25 <!--@ atm/atm_bench_full_20260821.json # config.candidate_k --> measured best.

---

## 3. What this may and may not be compared against

The board's own rows, full split, memory and RAG systems:

| type | harness | answer model | QS | Recall@10 |
|---|---|---|---:|---:|
| RAG | **RE-call** | `DeepSeek V4 Pro` | **68.4264** <!--@ atm/atm_bench_full_20260821.json # official_score.qs_percent --> | **92.8924** <!--@ atm/atm_bench_full_20260821.json # retrieval.recall_at_10_percent --> |
| Memory | Memexa (plus `Qwen3.6-27B` captions) | `DeepSeek-V4-flash` | 68.04 <!--@ citation-pending: a figure published on the ATM-Bench leaderboard by its maintainers, not a RE-call measurement; no artifact in this repository retains it --> \* | 79.09 <!--@ citation-pending: a figure published on the ATM-Bench leaderboard by its maintainers, not a RE-call measurement; no artifact in this repository retains it --> |
| Memory | Memexa | `DeepSeek-V4-flash` | 65.28 <!--@ citation-pending: a figure published on the ATM-Bench leaderboard by its maintainers, not a RE-call measurement; no artifact in this repository retains it --> \* | 78.93 <!--@ citation-pending: a figure published on the ATM-Bench leaderboard by its maintainers, not a RE-call measurement; no artifact in this repository retains it --> |
| Memory | MemPalace | `Qwen3-VL-8B-Instruct` | 56.80 <!--@ citation-pending: a figure published on the ATM-Bench leaderboard by its maintainers, not a RE-call measurement; no artifact in this repository retains it --> | 76.40 <!--@ citation-pending: a figure published on the ATM-Bench leaderboard by its maintainers, not a RE-call measurement; no artifact in this repository retains it --> |
| Memory | ScrapMem (No-Forget) | `Qwen3-VL-8B-Instruct` | 52.50 <!--@ citation-pending: a figure published on the ATM-Bench leaderboard by its maintainers, not a RE-call measurement; no artifact in this repository retains it --> | 70.30 <!--@ citation-pending: a figure published on the ATM-Bench leaderboard by its maintainers, not a RE-call measurement; no artifact in this repository retains it --> |
| RAG | ATM-RAG | `Qwen3-VL-8B-Instruct` | 51.00 <!--@ citation-pending: a figure published on the ATM-Bench leaderboard by its maintainers, not a RE-call measurement; no artifact in this repository retains it --> | 68.70 <!--@ citation-pending: a figure published on the ATM-Bench leaderboard by its maintainers, not a RE-call measurement; no artifact in this repository retains it --> |
| RAG | Self-RAG | `Qwen3-VL-8B-Instruct` | 50.30 <!--@ citation-pending: a figure published on the ATM-Bench leaderboard by its maintainers, not a RE-call measurement; no artifact in this repository retains it --> | 68.70 <!--@ citation-pending: a figure published on the ATM-Bench leaderboard by its maintainers, not a RE-call measurement; no artifact in this repository retains it --> |
| Memory | Mem0 | `Qwen3-VL-8B-Instruct` | 43.50 <!--@ citation-pending: a figure published on the ATM-Bench leaderboard by its maintainers, not a RE-call measurement; no artifact in this repository retains it --> | 61.90 <!--@ citation-pending: a figure published on the ATM-Bench leaderboard by its maintainers, not a RE-call measurement; no artifact in this repository retains it --> |

\* the board marks Memexa's QS as measured with a `DeepSeek-V4-flash` judge rather than
`gpt-5-mini`, so it is shown for reference and is not directly comparable. Its Recall is.

⛔ **Four boundaries, and none of them is a footnote.**

1. **The answer model is not matched.** Most baselines answer with `Qwen3-VL-8B-Instruct`; this run
   answers with `DeepSeek V4 Pro`. So the QS column compares *systems as configured*, not retrieval
   quality holding the reader fixed, and a share of the QS lead is bought by a stronger answerer.
   The board's own **Oracle** rows price that share: handed the gold evidence with no retrieval at
   all, `Qwen3-VL-8B-Instruct` scores 78.19 <!--@ citation-pending: a figure published on the ATM-Bench leaderboard by its maintainers, not a RE-call measurement; no artifact in this repository retains it --> and the strongest listed answerer scores 86.00 <!--@ citation-pending: a figure published on the ATM-Bench leaderboard by its maintainers, not a RE-call measurement; no artifact in this repository retains it -->. A
   controlled claim needs this run repeated on the baselines' answerer, and that has not been done.
2. **Recall@10 is the clean column.** It depends on neither the judge nor the answer model, so
   **92.8924 <!--@ atm/atm_bench_full_20260821.json # retrieval.recall_at_10_percent --> against 79.09 <!--@ citation-pending: a figure published on the ATM-Bench leaderboard by its maintainers, not a RE-call measurement; no artifact in this repository retains it -->** is like-for-like in a way the QS comparison is not. It is also the
   result this project would defend first, because retrieval is what RE-call is.
3. **The judge ran over a non-official transport.** The official evaluator, its prompt and the
   `gpt-5-mini` judge identity were all kept; only the HTTP endpoint was OpenRouter rather than
   OpenAI directly. This is disclosed in the pull request and **the maintainers have not ruled on
   it**. A same-prompt comparison of the two routes on 60 <!--@ citation-pending: measured 2026-08-20 by re-judging an existing answer file over both routes; the per-question output is not retained in this repository --> already-judged `open_end` questions
   agreed on 56 <!--@ citation-pending: measured 2026-08-20 by re-judging an existing answer file over both routes; the per-question output is not retained in this repository --> of 59 <!--@ citation-pending: measured 2026-08-20 by re-judging an existing answer file over both routes; the per-question output is not retained in this repository --> verdicts and ran about 1.7 <!--@ citation-pending: measured 2026-08-20 by re-judging an existing answer file over both routes; the per-question output is not retained in this repository --> points lower
   on that type, which is roughly 0.86 <!--@ citation-pending: measured 2026-08-20 by re-judging an existing answer file over both routes; the per-question output is not retained in this repository --> QS at its population share. That spread cannot be separated from the judge's own run-to-run
   variance, so treat it as an upper bound on route fidelity rather than a correction.
4. **The submission is open, not accepted.** The row exists as
   [pull request #2](https://github.com/atmbench/atmbench.github.io/pull/2) against the leaderboard,
   opened 2026-08-21 and unreviewed as of 2026-08-22. Until it merges, "first on the board" is a
   claim about arithmetic, not a placement.

Two smaller scope limits: the 31 <!--@ citation-pending: a property of the ATM-Bench dataset as published by its maintainers, not a RE-call measurement; no artifact in this repository retains it -->-question **ATM-Bench-Hard** split was not run, and `index_time` is
not reported because the run reused an existing index rather than building one.

---

## 4. Retrieval is not the bottleneck, and the gap says so

| type | Recall@10 | Recall@10GT | QS | gap |
|---|---:|---:|---:|---:|
| `number` | 93.0556 <!--@ atm/atm_bench_full_20260821.json # retrieval.by_qtype.number.recall_at_10_percent --> | 86.6667 <!--@ atm/atm_bench_full_20260821.json # retrieval.by_qtype.number.recall_at_10gt_percent --> | 72.7778 <!--@ atm/atm_bench_full_20260821.json # official_score.by_qtype.number.qs_percent --> | 20.2778 <!--@ atm/atm_bench_full_20260821.json # retrieval_to_qs_gap.number --> |
| `list_recall` | 89.9281 <!--@ atm/atm_bench_full_20260821.json # retrieval.by_qtype.list_recall.recall_at_10_percent --> | 83.4532 <!--@ atm/atm_bench_full_20260821.json # retrieval.by_qtype.list_recall.recall_at_10gt_percent --> | 59.8270 <!--@ atm/atm_bench_full_20260821.json # official_score.by_qtype.list_recall.qs_percent --> | 30.1011 <!--@ atm/atm_bench_full_20260821.json # retrieval_to_qs_gap.list_recall --> |
| `open_end` | 93.5798 <!--@ atm/atm_bench_full_20260821.json # retrieval.by_qtype.open_end.recall_at_10_percent --> | 88.1323 <!--@ atm/atm_bench_full_20260821.json # retrieval.by_qtype.open_end.recall_at_10gt_percent --> | 67.7043 <!--@ atm/atm_bench_full_20260821.json # official_score.by_qtype.open_end.qs_percent --> | 25.8755 <!--@ atm/atm_bench_full_20260821.json # retrieval_to_qs_gap.open_end --> |
| **overall** | **92.8924** <!--@ atm/atm_bench_full_20260821.json # retrieval.recall_at_10_percent --> | **86.9694** <!--@ atm/atm_bench_full_20260821.json # retrieval.recall_at_10gt_percent --> | **68.4264** <!--@ atm/atm_bench_full_20260821.json # official_score.qs_percent --> | **24.4660** <!--@ atm/atm_bench_full_20260821.json # retrieval_to_qs_gap.overall --> |

Retrieval finds the evidence for roughly nine questions in ten and the score lands at roughly seven
in ten. Whatever is left is **answer selection, synthesis and formatting**, not search.
`list_recall` is the weakest arm by a wide margin despite retrieval on it being only three points
below the others.

---

## 5. Where the remaining loss actually is

Measured 2026-08-22 by replaying the archived run package against the official scorer, **zero API
calls**: the replay reproduces QS 0.6843 <!--@ citation-pending: recomputed 2026-08-22 by replaying the archived run package against the official scorer; reproducible at zero cost but no committed artifact retains it --> and all three per-type accuracies exactly, so the
apparatus is verified before anything is concluded from it.

**22.53 <!--@ citation-pending: recomputed 2026-08-22 by replaying the archived run package against the official scorer; reproducible at zero cost but no committed artifact retains it --> QS points are lost on questions where the complete gold evidence
was already in the top 10 <!--@ atm/atm_bench_full_20260821.json # config.retrieval_k -->.** Split by type: `number` 5.92 <!--@ citation-pending: recomputed 2026-08-22 by replaying the archived run package against the official scorer; reproducible at zero cost but no committed artifact retains it -->, `list_recall` 3.58 <!--@ citation-pending: recomputed 2026-08-22 by replaying the archived run package against the official scorer; reproducible at zero cost but no committed artifact retains it -->, `open_end` 13.03 <!--@ citation-pending: recomputed 2026-08-22 by replaying the archived run package against the official scorer; reproducible at zero cost but no committed artifact retains it -->.

The largest single recoverable mechanism is **over-abstention**:

- The model declines to answer **156 <!--@ citation-pending: recomputed 2026-08-22 by replaying the archived run package against the official scorer; reproducible at zero cost but no committed artifact retains it --> of 1,013 <!--@ atm/atm_bench_full_20260821.json # question_count --> questions (15.4% <!--@ citation-pending: recomputed 2026-08-22 by replaying the archived run package against the official scorer; reproducible at zero cost but no committed artifact retains it -->)**, judged by the
  evaluator's own `is_abstention` normalizer. **Only 17 <!--@ citation-pending: recomputed 2026-08-22 by replaying the archived run package against the official scorer; reproducible at zero cost but no committed artifact retains it --> of those are
  correct**, and every one of the 23 <!--@ citation-pending: recomputed 2026-08-22 by replaying the archived run package against the official scorer; reproducible at zero cost but no committed artifact retains it --> gold-abstention questions is `open_end`.
- The other **139 <!--@ citation-pending: recomputed 2026-08-22 by replaying the archived run package against the official scorer; reproducible at zero cost but no committed artifact retains it --> wrong refusals are 13.72 <!--@ citation-pending: recomputed 2026-08-22 by replaying the archived run package against the official scorer; reproducible at zero cost but no committed artifact retains it --> QS of dead
  loss**, and 92 <!--@ citation-pending: recomputed 2026-08-22 by replaying the archived run package against the official scorer; reproducible at zero cost but no committed artifact retains it --> of them had the complete gold evidence retrieved.
- Because no gold abstention exists outside `open_end`, a rescue restricted to `number` and
  `list_recall` (53 <!--@ citation-pending: recomputed 2026-08-22 by replaying the archived run package against the official scorer; reproducible at zero cost but no committed artifact retains it --> and 31 <!--@ citation-pending: recomputed 2026-08-22 by replaying the archived run package against the official scorer; reproducible at zero cost but no committed artifact retains it --> refusals, each scoring exactly
  0.0000 <!--@ citation-pending: recomputed 2026-08-22 by replaying the archived run package against the official scorer; reproducible at zero cost but no committed artifact retains it -->) **cannot lose points**.

⛔ **The calibrated retrieval score does not separate a correct refusal from a wrong one.**
P(correct-abstention `top-1` score < wrong-abstention `top-1` score) = 0.493 <!--@ citation-pending: recomputed 2026-08-22 by replaying the archived run package against the official scorer; reproducible at zero cost but no committed artifact retains it -->,
against 0.5 <!--@ citation-pending: recomputed 2026-08-22 by replaying the archived run package against the official scorer; reproducible at zero cost but no committed artifact retains it --> for a coin.
So "gate abstention on the trust layer" is dead as stated: a threshold on that signal is a rigged
coin, not a judgment. A genuine sufficiency classifier is required, or nothing.

The second mechanism is **selection among retrieved items**: 117 <!--@ citation-pending: recomputed 2026-08-22 by replaying the archived run package against the official scorer; reproducible at zero cost but no committed artifact retains it --> questions,
11.31 <!--@ citation-pending: recomputed 2026-08-22 by replaying the archived run package against the official scorer; reproducible at zero cost but no committed artifact retains it --> QS, were answered wrongly with at least 80% <!--@ citation-pending: recomputed 2026-08-22 by replaying the archived run package against the official scorer; reproducible at zero cost but no committed artifact retains it --> of the
gold answer's content tokens present in the evidence the model actually received. A representative case named "Cambridge, U.S." and was answered with a Cambridge
UK address while both were on screen.

And there is a floor. On roughly 20 <!--@ citation-pending: recomputed 2026-08-22 by replaying the archived run package against the official scorer; reproducible at zero cost but no committed artifact retains it --> questions, 1.97 <!--@ citation-pending: recomputed 2026-08-22 by replaying the archived run package against the official scorer; reproducible at zero cost but no committed artifact retains it --> QS,
the gold item was retrieved but its **text
description does not contain the answer**, which is the modality ceiling of answering from generated
descriptions rather than from the image. That part is not winnable by any prompt or selector.

**Five answer-side arms have already been measured on a 300 <!--@ citation-pending: the answer-selection arm study, whose pre-registrations and appended results live on the branch named below; no artifact in this repository retains it -->-question subset and four had negative point estimates**: answer-format contracts, per-item qualifier marking, a disposition sentence, and
evidence budget reallocation. The best arm was the official prompt with the original greedy packer,
and nothing beat it with a confidence interval excluding zero. Do not repeat that family without a
new mechanism. The pre-registrations and their appended results are on
[`claude/atm-answer-selection-public`](https://github.com/GiulioDER/RE-call/tree/claude/atm-answer-selection-public).

---

## 6. Reproduction, and the gap in it

The offline run package (`manifest.json`, `answers.jsonl`, `retrieval.jsonl` and the four official
evaluator outputs) is archived outside this tree, and its `SHA-256` checksums, the dataset hashes and
the evaluator file hash are recorded inside
[`results/atm/atm_bench_full_20260821.json`](../results/atm/atm_bench_full_20260821.json). The
evaluator is ATM-Bench's own `memqa/utils/evaluator/evaluate_qa.py`, run with `--metrics atm`.

⚠️ **The exact commit that produced this run, `6c0ec26b`, is not on a public branch.** The ATM
harness and its pre-registrations are published on
[`claude/atm-answer-selection-public`](https://github.com/GiulioDER/RE-call/tree/claude/atm-answer-selection-public),
but that branch carries later answer-selection work and its runner differs from the one used here.
So the artifacts are checksummed and the configuration is fully stated, and a byte-exact
re-execution is **not** currently possible from public code. That is a reproduction gap, not a
disclosure choice, and closing it means landing the runner on `master`.

---

## 7. Prior context in this repository

| Document | What it adds |
|---|---|
| [MTRAG_BENCHMARK.md](MTRAG_BENCHMARK.md) | The other external benchmark, where RE-call does **not** top the board, and the abstention result that does hold up |
| [EVIDENCE.md](EVIDENCE.md) | The one-line version of this result beside every other claim and its limit |
| [../results/ARTIFACTS.md](../results/ARTIFACTS.md) | The artifact map, including this run's checksums |
