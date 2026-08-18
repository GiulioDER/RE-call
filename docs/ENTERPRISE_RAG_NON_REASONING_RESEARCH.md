# EnterpriseRAG non reasoning retrieval research

**Date:** 2026-08-18
**Status:** global reranker rejected; completeness candidate passed retrieval confirmation, answer
comparison blocked by OpenRouter credits
**Recommendation:** continue the fixed answer comparison when credits are available. Do not
promote the candidate or claim top five status.

## Scope and evaluation boundary

This session studies retrieval, reranking, indexing, chunking, hybrid configuration, and answer
generation without a reasoning expansion arm. The requested runtime policy is:

1. `--no-correction`
2. `--skip-citation-stripping`
3. `--parallelism 1`
4. no gold expected document ids, answer facts, or competitor answers in the runner

The official evaluator may use gold fields only after an answer file has been generated. The
runner now strips those fields while loading questions. Retrieval quality is computed by
`scripts/enterprise_rag_posthoc_metrics.py` or the official evaluator after generation.

The code and experiment plan are in the current isolated worktree on branch
`claude/reasoning-366-next`. The preregistration was committed as `eb9dace` before the new
measurement attempt.

## Reproduction of the five question finding

The existing triage report records this prior paired retrieval result on five fixed
`project_related` development questions:

1. Configuration: Voyage `voyage-4-large`, lexical hybrid retrieval, `k=8`, `candidate_k=200`,
   extractive output, and no reasoning arm.
2. No reranker: document recall `46.67%`, exact document set coverage `20.0%`.
3. Voyage `rerank-2.5`: document recall `66.67%`, exact coverage `40.0%`, mean extra document
   delta `-0.4`, two retrieval gains, and zero retrieval losses.

The answer files were recovered from the isolated VPS2 stage and compared locally after
generation. The five-question result reproduces the earlier finding. The held-out confirmation
is negative for the global reranker. The paired reports are
`results/enterprise_rag/non_reasoning/paired_dev.json` and
`results/enterprise_rag/non_reasoning/paired_confirmation.json`.

The capture recovery used these checks on 2026-08-18:

```powershell
mcp__qwen_mcp__port_check(host="vps2", port=22, timeout_ms=3000)
ssh -o BatchMode=yes -o ConnectTimeout=8 vps2 "printf 'remote-ok\n'"
py -3 -c "import zipfile; p=r'C:/Users/gde00/Documents/recall/.benchdata/enterprise-rag-v1.0.0/all_documents.zip'; z=zipfile.ZipFile(p); n=[x for x in z.namelist() if x.endswith('.txt')]; print(len(n), sum(z.getinfo(x).file_size for x in n))"
```

The VPS2 port and SSH checks later succeeded. The local official release contains `511962` text
documents and `2473634648` uncompressed bytes. I did not rebuild that corpus locally. The
recovered VPS2 answer artifacts remain historical captures because their older manifests include
evaluator-style posthoc retrieval fields. I used those fields only for post-generation analysis,
never as runtime retrieval inputs.

I started a session owned local database on port `5641` for harness verification. I did not reset,
drop, or mutate the shared VPS2 benchmark table.

## Experiment implementation

The following files are reproducible artifacts:

1. `benchmarks/enterprise_rag.py` now removes gold fields from runtime question objects and records
   retrieval stage timings, document capture stability, embedding calls, lexical calls, SPLADE
   calls, reranker calls, and answer model calls in the run manifest. Retrieval cost remains
   `null` when the provider does not expose a measured usage record.
2. `scripts/enterprise_rag_experiment.py` constructs and optionally executes a serial grid with
   isolated table and tenant names. Its default grid is `k` in `5,8,12`, `candidate_k` in
   `100,200,400`, and reranker in `none,voyage:rerank-2.5`. The default sparse arm is lexical.
3. `scripts/enterprise_rag_posthoc_metrics.py` computes document recall, exact coverage, and
   invalid extra documents only after answer rows exist.
4. `scripts/enterprise_rag_retrieval_compare.py` compares paired answer files and can include the
   two run manifests for latency and call deltas. Its output declares the posthoc phase.
5. `scripts/enterprise_rag_score_openrouter.sh` now passes the requested no correction and skip
   citation stripping flags and defaults evaluator parallelism to one.
6. `results/enterprise_rag/reranker_grid_plan/experiment.manifest.json` records the lexical grid
   plan with the official question and document hashes.

The full grid is not executed against an incomplete local index. The recovered VPS2 index was
used only for the paired project reproduction and confirmation. Any changed index configuration
still requires a new isolated table and tenant.

SPLADE is handled through the existing checked-in artifact rather than a new CPU run. The source
is `benchmarks/artifacts/enterprise_rag/re_call_voyage_splade_gpt4o.answers.jsonl` with manifest
`benchmarks/artifacts/enterprise_rag/re_call_voyage_splade_gpt4o.answers.manifest.json`. It has
500 answer rows and records Voyage dense retrieval, lexical plus SPLADE sparse retrieval, Voyage
`rerank-2.5`, `k=8`, `candidate_k=200`, and `openai/gpt-4o` answering. Its posthoc retrieval
summary is 77.34% document recall, 72.13% exact coverage, and 6.94 mean invalid extras. This is
useful full-benchmark evidence, but it is not a paired SPLADE versus lexical retrieval capture,
so it does not establish a causal SPLADE gain. The no-GPU VPS2 availability check remains valid
for new SPLADE backfill work.

## Retrieval arms and current status

The five fixed slices are `project_related`, `completeness`, `semantic`, `basic`, `constrained`,
and `conflicting_info`. The dev and confirmation ids are in
`results/enterprise_rag/top5_slices`.

The requested full paired grid remains pending. The measured project pair and the completeness
development and confirmation pairs are recorded in the JSON artifacts. The planned measurements
are:

1. No reranker and Voyage `rerank-2.5`.
2. Candidate pools `100`, `200`, and `400`.
3. Final `k` values `5`, `8`, and `12`.
4. Lexical retrieval, then lexical plus SPLADE only after the SPLADE model and isolated index are
   available and source coverage is verified.
5. Three repeated captures per selected arm.
6. Document recall, exact coverage, invalid extras, per question gains and losses, latency, stage
   timings, provider call counts, and measured monetary cost.

The Pareto rule is not higher `k` by default. A candidate must improve paired recall or exact
coverage without a material increase in invalid extras or context latency. It must also remain
stable across repeated captures.

## Answer generation

No global reranker answer run is accepted for promotion because the global reranker failed the
held-out project retrieval gate. Existing reader artifacts are retained as historical exploratory
output, not as a promotion result. Retrieval must first improve paired confirmation rows before
the same answer model and prompt are run over a changed document set.

The completeness candidate did pass its paired retrieval confirmation, so I ran the controlled
answer experiment on the same 10 confirmation questions. Both arms used `openai/gpt-4o`, the
baseline answer policy, a 3,500-character context, serial retrieval, and the requested official
evaluator flags. The answer metrics were:

| arm | correctness | completeness | combined score | recall | invalid extras |
|---|---:|---:|---:|---:|---:|
| no reranker | 10.0% | 4.17% | 4.17 | 21.0% | 7.0 |
| Voyage reranker | 50.0% | 21.67% | 21.67 | 37.58% | 6.0 |

The candidate added one reranker call per question and increased mean retrieval latency from
20,568.9 ms to 21,566.0 ms, with p95 latency from 27,391.0 ms to 27,734.2 ms. Retrieval-provider
cost was not exposed by the runtime telemetry. The answer manifests are
`results/enterprise_rag/non_reasoning/completeness_confirmation_none_openrouter.answers.jsonl.manifest.json`
and
`results/enterprise_rag/non_reasoning/completeness_confirmation_reranker_openrouter.answers.jsonl.manifest.json`.
The initial answer experiment was promising but not promotable because it covered only one category
and had no stable repeated-capture evidence or 500-question evaluation.

I then repeated both completeness confirmation arms three times per question with extractive
output. Both arms had capture stability `1.0`, and the paired retrieval result remained the same:
the no-reranker arm had 21.00% recall and the reranker arm had 37.58%, with unchanged 10% exact
coverage, one fewer invalid extra per question, six gains, and one loss. The repeated comparison
is `results/enterprise_rag/non_reasoning/paired_completeness_confirmation_repeat3.json`.
The no-reranker arm averaged 30,314.7 ms retrieval latency with p95 44,903.5 ms. The reranker
arm averaged 31,242.1 ms with p95 72,509.7 ms. Each arm recorded 30 embedding calls and 30
lexical calls; the reranker arm recorded 30 additional reranker calls. Retrieval cost was not
available. The mean latency increase was modest, but the p95 increase is material, so this result
supports continued investigation rather than promotion.

The answer side remains a separate experiment. Candidate variables are answer model, context
length, document order, structured answer format, source labels, abstention wording, and the
official correction and citation settings. They must not be changed together with the first
reranker comparison.

## Index improvement agenda

The current implementation exposes chunk size, overlap, reranker document truncation, candidate
pool size, sparse backend, title and source type inclusion, query embedding caching, table, and
tenant in the manifest. The following items are not yet measured in this session:

1. Chunk size and overlap sweep.
2. Document level ranking compared with chunk level ranking.
3. Source type weighting.
4. Lexical normalization.
5. Duplicate and near duplicate handling beyond the current document id collapse.
6. Tunable dense and sparse fusion weights beyond the current RRF behavior.
7. SPLADE fusion on an isolated index.
8. Reranker truncation values.
9. Query cache correctness on repeated captures.
10. Tenant freshness and source coverage checks.

Each future index arm must use a distinct table and tenant. No shared VPS2 benchmark table may be
reset or mutated.

## Official competitor artifact analysis

I analyzed only the downloaded leaderboard snapshot in `.tmp_erbleaderboard`, using its official
`leaderboard.csv`, `systems.yaml`, and `results_*.json` artifacts. The reproducible output is
`results/enterprise_rag/official_competitor_analysis.json`.

The official top five rows are:

1. Troml: score `76.79`, correctness `83.8`, completeness `81.84`, recall `86.55`, invalid extras
   `12.65`.
2. Skyller: score `71.93`, correctness `77.0`, completeness `79.14`, recall `81.6`, invalid extras
   `8.86`.
3. OpenClaw: score `68.22`, correctness `81.6`, completeness `72.86`, recall `79.02`, invalid
   extras `0.47`.
4. fgroo: score `63.27`, correctness `71.0`, completeness `71.03`, recall `72.5`, invalid extras
   `0.63`.
5. OpenAI File Search: score `61.03`, correctness `69.8`, completeness `67.87`, recall `71.65`,
   invalid extras `15.7`.

The artifacts show outcome associations, not undocumented internal mechanisms. They support three
limited observations:

1. Strong systems are not uniformly low in invalid extras. Troml leads on recall and answer
   metrics while carrying more extras than OpenClaw and fgroo.
2. Low extras alone do not establish a top score. OpenClaw and fgroo combine low extras with
   different recall and answer quality profiles.
3. The strongest measurable public pattern is joint performance on recall, correctness, and
   completeness. The artifacts do not prove whether any system used reranking, source weighting,
   or document set construction internally.

## Held out reranker confirmation

The preregistered confirmation used 23 held out `project_related` questions with the same index,
`candidate_k=200`, `k=8`, lexical hybrid retrieval, and extractive output. The no-reranker and
`voyage:rerank-2.5` arms were paired question by question.

| arm | document recall | exact coverage | invalid extra delta | gains/losses |
|---|---:|---:|---:|---:|
| no reranker | 61.7% | 13.0% | reference | reference |
| Voyage reranker | 60.9% | 26.1% | +0.22 | 6/6 |

The exact-coverage increase is useful evidence that the reranker can improve set precision on
some questions, but the primary recall prediction failed and invalid extras increased. The
simple global reranker is rejected. No reranker answer-quality confirmation was run after this
failure, preserving the preregistered retrieval-first gate.

The next candidates should explain the dev to confirmation reversal through deterministic score
calibration, adaptive reranking, source or parent-document coverage, or chunk selection. Each
candidate needs a new preregistration and a held out confirmation before answer generation.

## Rank blend and adaptive depth follow-ups

The preregistered reciprocal rank blend kept the original hybrid ordering in the candidate score
and added Voyage rank with weight `0.50`. On the 17-question project development split, recall
rose from `53.77%` to `55.57%`, exact coverage stayed at `23.53%`, and invalid extras fell by
`0.06` per question. On the 23-question confirmation, recall moved from `61.71%` to `61.59%`,
exact coverage stayed at `13.04%`, and invalid extras rose by `0.043`. The blend is rejected for
promotion because its confirmation recall was lower than baseline.

The preregistered deterministic depth test changed only `k` from 8 to 12 on the same project
confirmation. It raised recall from `61.71%` to `66.55%` and exact coverage from `13.04%` to
`17.39%`, with six gains and one loss. Invalid extras increased by `3.61` per question, above
the `2.0` guardrail, so raw `k=12` is rejected as a global setting. The strongest next retrieval
hypothesis is selective depth, using a runtime confidence or source coverage signal to ask for
more documents only when the baseline is uncertain, while preserving the smaller submitted set
when it is confident.

## Promotion decision

The candidate is not promoted. Promotion criteria remain positive paired confirmation, stable
repeated captures, no material invalid extra increase, no citation or document id mismatch, no
info not found regression, measured latency and cost, and a homogeneous 500 question official
evaluation using the requested evaluator flags. Only that final evaluation can be compared with
the `61.03` top five threshold, and no top five claim is made here.

## Selective-depth development screen

The development feature screen selected `max_dense_score < 0.75` as the frozen selective-depth
candidate. It expands six of 17 project development questions, raises mean document recall by
3.43 points, leaves exact coverage unchanged, and adds 1.24 invalid documents per question. The
feature priority was exploratory because the parent preregistration did not fix an order between
the two feature candidates; that limitation is disclosed in the confirmation amendment.

The first held out selective-depth capture was invalid because the implementation retrieved depth
12 before deciding whether to expand, so it was discarded. I corrected the arm to retrieve depth 8
first and perform a second depth 12 pass only when the frozen confidence rule fires. The corrected
three-capture confirmation rejected the hypothesis. With `max_dense_score < 0.75`, recall fell from
61.71% to 60.27%, exact coverage fell from 13.04% to 8.70%, and invalid extras rose by 0.83 per
question. The candidate had one gain and two losses. It made 84 embedding and lexical calls across
23 questions, with mean retrieval latency 27.47 seconds and p95 latency 43.28 seconds per question.
Reranker, SPLADE, and answer-model calls were zero. Capture stability was 1.0 and retrieval cost was
unavailable. No answer-quality test is authorized for this arm.

## Candidate pool screen

The completeness confirmation was screened at `k=8` with candidate pools 100, 200, and 400 using
the existing lexical index. Candidate pools 100 without reranking and 400 without reranking
matched the repeated `candidate_k=200` baseline at 21.0% recall, 10% exact coverage, and no
invalid-extra delta. Candidate pool 100 took 17.84 seconds mean latency and 25.93 seconds p95;
candidate pool 400 took 30.75 seconds mean latency and 91.23 seconds p95. The smaller pool is
therefore the lower latency tie, while the larger pool has no retrieval benefit.

The candidate pool 100 Voyage reranker arm reproduced the `candidate_k=200` reranker result:
37.58% recall, 10% exact coverage, and 1.0 fewer invalid extras per question. It took 18.15
seconds mean latency and 26.23 seconds p95, with 10 reranker calls. It does not dominate the
candidate pool 200 reranker on retrieval quality.

The candidate pool 400 Voyage reranker arm with the preregistered 4,000 character truncation
failed before producing an answer file. Voyage rejected a 666,493 token batch against its 600,000
token limit. This provider failure is retained. A separately preregistered 2,000 character
truncation fallback produced 40.92% recall, 10% exact coverage, and 1.2 fewer invalid extras per
question, with five gains and one loss. Three repeated captures were stable at 1.0. The repeated
fallback averaged 19.74 seconds per question with 29.18 seconds p95, and recorded 30 embedding,
30 lexical, and 30 reranker calls. This is a retrieval continuation candidate, not a promotion.

The fixed answer comparison for the fallback was attempted with `openai/gpt-4o`, the baseline
answer policy, 3,500 character context, and no correction or citation stripping. OpenRouter
rejected the first request because the remaining credit allowed only 1,197 tokens while the runner
requested up to 16,384. No complete candidate answer file or answer score exists. I did not change
the model, prompt, or policy to work around the credit limit.
