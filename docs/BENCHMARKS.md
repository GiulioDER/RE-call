# RE-call benchmark results

Published scores, dated positions and who ran each test.

All runs below were operated by the RE-call author. None has an independently verified rerun.

Evidence reviewed 19 October 2026. Leaderboard positions are a dated snapshot.

This page is the maintained copy of the scorecard published at `site/benchmarks.html`. The labels
used below are defined under [Who ran it matters](#who-ran-it-matters).

| Benchmark | Scope | RE-call result | Position / comparison | Run verification |
|---|---|---|---|---|
| [ATM-Bench](#atm-bench) | Personal multimodal memory, full split, 1,013 questions | 90.10% Recall@10, QS 68.43% | 2nd on Recall@10 | Self-tested, organizer-listed |
| [EnterpriseRAG-Bench](#enterpriserag-bench) | Company knowledge retrieval and QA, 500 questions | 63.80% answer correctness | Unranked | Self-tested |
| [MTRAG](#mtrag) | Multi-turn conversational RAG, 842 generation tasks | 0.5527 Task C harmonic score | 4th of 10 locally | Self-tested, official evaluator |
| [LoCoMo](#locomo) | Long conversation memory, 1,540 answerable questions | 48.4% judged answer accuracy | Ahead of Mem0 OSS (44.4%) in a paired test | Self-tested |
| [LongMemEval](#longmemeval) | Session retrieval, historical local-embedder study | 0.970 / 0.366 Hit@5, small / merged haystack | Unranked | Self-tested, historical |
| [BEAM](#beam) | Long-context memory and abstention, historical 1M-token probes | Partial results only | Unranked | Self-tested, historical |
| [BEIR / CQADupStack](#beir--cqadupstack) | Cross-corpus retrieval study, 17 corpora | 16 / 17 corpora improved by cloud embeddings | No system ranking | Self-tested |
| [FinanceBench](#financebench) | Financial-document retrieval, 150 questions | 0.393 → 0.527 Hit@5 as the candidate pool grew | No system ranking | Self-tested, historical |
| [agent-memory-bench](#agent-memory-bench) | Coding-agent task execution, official-003, 317 paired cells | 65.93% task success | Tied 2nd, owner-run board | Self-tested |
| [mem-bench](#mem-bench) | Isolation and temporal axes | Historical runs | Unavailable | Self-tested, historical |

## ATM-Bench

Personal multimodal memory. Run: 21 Aug 2026. Full split, 1,013 questions.

- **Result:** 90.10% Recall@10, QS 68.43%.
- **Position:** 2nd on Recall@10, among listed Memory/RAG configurations, excluding oracles.
  Checked 19 Oct 2026.
- **Verification:** self-tested, organizer-listed. No independent rerun established.

This is RE-call's frozen 21 August 2026 run, accepted on 23 August. The official retrieval
evaluator rescored that same run on 4 October: Recall@10 is 90.0995%. The earlier value labelled
Recall@10 measured whether any gold item was found, a different metric.

Voyage-4-large embeddings, Voyage rerank-2.5 and Qwen image captions; DeepSeek V4 Pro answers. The
official QS judge model and prompt ran through OpenRouter, disclosed and accepted. QS is not
answer-model-matched to the Qwen baselines. Memexa v2 currently leads the listed systems on both
Recall@10 and QS. Oracle rows are excluded from the rank.

The Memexa v2 submission was merged on 4 October 2026. The board compares these submitted
configurations from different dates; it is not a simultaneous head-to-head test or a new test of
current RE-call.

Sources: [official leaderboard](https://atmbench.github.io/leaderboard.html),
[accepted RE-call submission](https://github.com/atmbench/atmbench.github.io/pull/2),
[later Memexa v2 submission](https://github.com/atmbench/atmbench.github.io/pull/3),
[recall artifact](../results/atm/atm_recall_rescore_20261004.json),
[QS and configuration](../results/atm/atm_bench_full_20260821.json).

## EnterpriseRAG-Bench

Company knowledge retrieval and QA. 500 questions.

- **Result:** 63.80% answer correctness. Combined score 46.16, document recall 77.34%.
- **Position:** unranked. No RE-call row on the organizer's board when checked 19 Oct 2026.
- **Verification:** self-tested, public score summary. No independent rerun established.

GPT-4o answers; a homogeneous GPT-5.4 medium judge pass, without correction or citation stripping.
Voyage-4-large, lexical plus SPLADE retrieval, Voyage reranking and a bounded context. Completeness
is 53.23%.

The separate default judge pass reported 65.60% correctness but mixed reasoning settings. This page
uses the homogeneous pass. A locally computed score does not receive a hypothetical place among
accepted submissions.

Sources: [score artifact](../benchmarks/artifacts/enterprise_rag/re_call_voyage_splade_gpt4o.judge_gpt54_medium.no_correction.summary.json),
[protocol and other judge passes](ENTERPRISE_RAG_SUBMISSION.md),
[official leaderboard](https://huggingface.co/spaces/onyx-dot-app/EnterpriseRAG-Bench-Leaderboard).

## MTRAG

Multi-turn conversational RAG. 842 generation tasks.

- **Result:** 0.5527 Task C harmonic score. Correct refusals: 16/55 (29.1%).
- **Position:** 4th of 10 locally, in a recomputed end-to-end comparison. Tied 2nd on correct
  refusals. No official placement.
- **Verification:** self-tested, official evaluator. An external judge model is not an independent
  operator.

GPT-4o generation with the official prompt, scored by the benchmark's GPT-4o-mini judge. All
baselines were recomputed through the same corrected, answerability-conditioned scorer. Their
instance set differs from the paper's, so these positions cannot be quoted as the published
leaderboard.

The recomputed GPT-4o baseline scores 0.5591. Holding the prompt fixed, replacing ELSER contexts
with RE-call contexts changes 0.5516 to 0.5527. Correct refusals are tied with Llama-3.1-70B; that
metric depends on the generator prompt as well as retrieval.

Retrieval Task A, a separate 777-query evaluation, reaches nDCG@5 0.4342 and Recall@100 0.7668
with SPLADE and Voyage reranking. The free hybrid SPLADE arm scores 0.3573 and 0.7377
respectively. Task B supplies gold contexts and is a harness control, not a RE-call retrieval
result.

Sources: [corrected results and reproduction](MTRAG_BENCHMARK.md),
[scoring correction](../results/mtrag_generation/CORRECTION-idk-conditioning-2026-08-09.md),
[organizer benchmark](https://github.com/IBM/mt-rag-benchmark).

## LoCoMo

Long conversation memory. Paired QA, 1,540 answerable questions.

- **Result:** 48.4% judged answer accuracy. Mem0 OSS: 44.4% in the same setup.
- **Position:** ahead in the paired test of RE-call versus Mem0 OSS, with matched reader and judge.
  No official placement.
- **Verification:** self-tested, public compact artifact. A second scoring script is not a
  third-party rerun.

GPT-4o reader and judge, token-matched retrieval budgets. RE-call uses the local bge embedder. The
direction depends on the reader: a post-hoc Sonnet 4.5 subset reverses it. The paired artifact
records the raw-run hashes; raw generated answers are not committed.

Separate retrieval-only tests use 1,536 answerable questions: turn-level hit@5 is 0.671 without
reranking and 0.777 with the shipped cross-encoder. These are not session-level competitor scores or
judged answer accuracy. Default retrieval abstained on none of the 446 adversarial questions; that
failure is retained in the record.

Sources: [paired QA artifact](../results/head_to_head/paired_accuracy.json),
[review and reader controls](../benchmarks/REVIEW.md),
[retrieval artifact](../results/locomo_rerank/rerank_shipped.json),
[full retrieval and abstention record](../results/RESULTS.md).

## LongMemEval

Session retrieval. Historical local-embedder study.

- **Result:** 0.970 / 0.366 Hit@5, small / merged haystack. Retrieval only, not answer accuracy.
- **Position:** unranked. Different haystack scopes; no comparable official row.
- **Verification:** self-tested, historical. Original artifact lost; documented result, limited
  auditability.

The per-question haystacks contain roughly 49 sessions; the merged index contains 19,195. The
intermediate Oracle scope scores 0.719. Knowledge-update retrieval is 36/36 on the comparable arm,
while false abstention is 0.481.

This run predates the sparse-leg fix and is effectively dense-only. The indexes and original run
output were not retained. The published commands provide a reproduction method, not a retained
artifact or independent verification. A separate Bench'd harness run is recorded
[below](#benchd-harness-historical-locomo-and-longmemeval-runs).

Sources: [historical result and retention limits](../results/RESULTS.md),
[analysis](../results/FINDINGS.md#10-longmemeval-the-retrieval-result-and-the-abstention-failure-underneath-it).

## BEAM

Long-context memory and abstention. Historical 1M-token probes.

- **Result:** partial results only. Temporal category: 0.408. Mem0 reference: 0.567.
- **Position:** unranked. No retained full-run aggregate suitable for a leaderboard claim.
- **Verification:** self-tested, historical. Published study notes; incomplete retained evidence.

The harness follows upstream answerer and judge prompts and compares against historical Mem0
Platform answers. The published temporal probe identifies wrong date selection. The source record
does not retain a complete aggregate artifact, so this page does not assign an overall score or
rank.

BEAM includes deliberately difficult unanswerable questions. Follow-up entailment and threshold
probes did not establish a usable general abstention rule. The committed Voyage k-sweep measures
nugget-coverage proxy, not judged answer quality.

Sources: [temporal result](../results/FINDINGS.md),
[temporal diagnostic artifact](../results/beam_9l_temporal.json),
[upstream protocol](../benchmarks/beam/UPSTREAM.md),
[proxy artifact](../results/beam_voyage/ksweep.json).

## BEIR / CQADupStack

Cross-corpus retrieval study. 17 corpora.

- **Result:** 16 / 17 corpora improved by cloud embeddings. Median hit@5 lift: +0.059.
- **Position:** no system ranking. Within-RE-call embedder comparison.
- **Verification:** self-tested, public study summaries. Per-corpus artifact set incomplete.

Hybrid retrieval with bge-small-en-v1.5 versus Voyage-3. This measures the embedder change within
RE-call, not RE-call versus other retrieval systems. The corpus samples and configuration are part
of the study; these figures are not official full-corpus BEIR nDCG scores.

The study and analysis summaries are public; the complete set of per-corpus run artifacts is not
retained in this tree.

Corpora: NFCorpus, SciFact, SCIDOCS, FiQA, ArguAna, and CQADupStack Android, English, Gaming, GIS,
Mathematica, Physics, Programmers, Stats, TeX, Unix, Webmasters and WordPress.

Sources: [study results](../results/gap/FINDINGS-embedder-gap.md),
[corpus inventory](../results/gap/summary.json),
[analysis artifact](../results/gap/analysis.json).

## FinanceBench

Financial-document retrieval. 150 questions.

- **Result:** 0.393 → 0.527 Hit@5 as the candidate pool grew, pool 40 versus 100.
- **Position:** no system ranking. Within-RE-call configuration study.
- **Verification:** self-tested, historical. Published aggregate; no compact run artifact linked in
  the record.

Voyage-finance-2, dense retrieval plus reranking over 72,151 chunks, with the dense search-width cap
corrected. This is a retrieval-pool experiment, not the benchmark's end-to-end financial QA score.

Source: [published measurement and scope](../results/FINDINGS.md).

## agent-memory-bench

Coding-agent task execution. official-003, 317 paired cells.

- **Result:** 65.93% task success. Bare: 65.93%, placebo: 67.19%.
- **Position:** tied 2nd on the owner-run board, among displayed arms by success. Null versus the
  designated baseline.
- **Verification:** self-tested; benchmark by the RE-call author. Execution-graded; no independent
  operator.

Claude Code with DeepSeek V4 Flash over a bulk-ingested corpus. No positive arm's confidence
interval excludes zero against the designated CLAUDE.md baseline. RE-call ties the bare arm;
placebo scores highest. This does not demonstrate an overall coding benefit.

The write lifecycle is not measured. The run disclosed late preregistration, no advance
announcement, and source and condition names visible to agents. The current board joins later
vendor runs with different admitted cell counts, so its descriptive order is not a controlled
pairwise ranking. Earlier pilots and protocol runs remain in the linked history.

Sources: [owner-run leaderboard](https://giulioder.github.io/agent-memory-bench/),
[displayed data](https://github.com/GiulioDER/agent-memory-bench/blob/master/site/data/leaderboard.js),
[all public runs and deviations](https://github.com/GiulioDER/agent-memory-bench/blob/master/docs/STATUS.md).

## mem-bench

Isolation and temporal axes. Earlier author-built benchmark.

- **Result:** historical runs. No current score promoted here.
- **Position:** unavailable. Former public repository returned 404 on 19 Oct 2026.
- **Verification:** self-tested, historical. Adapters retained; declared versions were not
  live-verified.

The retained documentation records earlier isolation and temporal runs. The original board's
artifacts declared a version without confirmation; a declared version and deterministic scoring do
not establish an independently verified run. The adapters are public, but the former board is not
currently accessible.

Source: [retained adapters and version caveat](../benchmarks/membench/README.md).

## Who ran it matters

- **Self-tested:** the RE-call author operated the run. An official evaluator, public artifact or
  signed manifest can make it auditable without making the operator independent.
- **Organizer-listed:** the benchmark maintainer accepted a row onto its leaderboard. Acceptance
  alone does not establish that the organizer reran it.
- **Independently verified:** a separate operator reproduced the run and published evidence. No
  result on this page currently has that status.

Positions apply only to the named metric, track, eligible rows and review date. A local baseline
comparison is labelled local. Missing placements stay unranked.

## Other tests and historical runs

This register covers published benchmark families. Configuration sweeps, diagnostic controls and
earlier runs remain in their linked records. In-progress and unpublished competition results are
outside this scorecard.

### Bench'd harness: historical LoCoMo and LongMemEval runs

The August 2026 self-run release reports nuance scores of 71.6 on LoCoMo and 69.0 on LongMemEval.
These use a different pipeline from the retrieval and paired QA rows above: Voyage embeddings and
reranking, DeepSeek synthesis inside memory, and the harness's locked GPT-4o-mini answerer and
judge. Abstention was suppressed on the answerable-only evaluation.

The manifests are signed by the submitting operator. They were not accepted onto a public board and
were not independently rerun. No rank or independent-verification claim is made from them.

Sources: [signed manifests and run records](https://github.com/GiulioDER/RE-call/releases/tag/benchd-official-2026-08-23),
[harness protocol and disclosures](../benchmarks/benchd/README.md).

### Internal agent A/B, answerability and trust evaluations

- Additive agent A/B: known-hazard trap rate 0.000 with RE-call versus 0.525 with the same static
  memory alone, over 40 memory-only pairs. This author-designed task set is narrow and is not a
  public coding leaderboard. [Artifact and limits](../results/agent_ab/agent_ab_additive_2026-08-21.json).
- SDK replication: a separate in-process tool replication was inside its preregistered equivalence
  bands. It was still run by the author. [Replication artifact](../results/agent_ab/agent_ab_sdk_replication_2026-08-26.json).
- Answerability ladder: diagnostic levels and the corrected H1 verdict are retained, without a
  leaderboard position. [Corrected verdict](../results/ladder/H1_VERDICT_v2.md).
- Trust, supersession and entailment: synthetic controls and near-miss studies test particular
  failure modes. Generated-corpus success is not a production reliability rate.
  [Full result tables](../results/RESULTS.md), [study](ENTAILMENT_SUPERSESSION_STUDY.md).
- Reasoning controls: frozen offline control and ablation suites, distinct from externally graded
  reasoning benchmarks. [Audit and release limits](REASONING_SESSION8_AUDIT.md).

### PEPs, real-corpus, fine-tuning, scale and latency studies

The author also tested PEP retrieval, labelled private-corpus retrieval, embedding fine-tuning,
synthetic scale, reranking, cosine distributions, generation parity and store latency. These measure
individual mechanisms or operating costs. They do not assign a place among competing memory
products.

Only published aggregates are linked here. Private corpus text and raw operational traces are not
part of this page. All of these studies are self-tested.

Sources: [result tables](../results/RESULTS.md), [artifact map](../results/ARTIFACTS.md),
[fine-tuning study](RAG_TRAINING_STUDY.md).

### Corrections and withdrawn figures

Withdrawn measurements are excluded from the scorecard. The linked record preserves the
corrections, including contaminated LoCoMo pool runs, an invalid known-item evaluation, and ATM
metric and token-accounting errors. Historical figures with lost artifacts remain explicitly
labelled above.

Sources: [withdrawal register](../results/WITHDRAWN.json),
[findings and corrections](../results/FINDINGS.md).
