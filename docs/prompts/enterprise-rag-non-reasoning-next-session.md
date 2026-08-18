# Separate-session prompt: EnterpriseRAG non-reasoning improvement study

You are continuing a research task on the RE-call repository. The objective is to move RE-call into
the top five systems on the official EnterpriseRAG-Bench leaderboard by improving retrieval and
evidence plumbing, with this session focused on approaches that do not depend on answer-time
reasoning, decomposition, or model-generated query expansion.

Do not fabricate benchmark data. Use the official sources and preserve exact commits, hashes,
configuration, and per-question artifacts. Use cheap mini models for exploratory answer tests and
judge calls. Do not spend on a full 500-question answer evaluation until a retrieval candidate has
passed a preregistered held-out confirmation.

## Official sources

* Leaderboard Space: https://huggingface.co/spaces/onyx-dot-app/EnterpriseRAG-Bench-Leaderboard
* Benchmark repository: https://github.com/onyx-dot-app/EnterpriseRAG-Bench
* Dataset: https://huggingface.co/datasets/onyx-dot-app/EnterpriseRAG-Bench
* RE-call repository: https://github.com/GiulioDER/RE-call

The frozen official references are:

* Leaderboard snapshot commit: `9d1fd7ec34cb137ba89f385741500116dbf9600f`
* Benchmark commit: `d36685e273713975ee20299bbf1ab64165575b3c`
* Questions SHA256: `f9524b9157cd43aae36b99333a124738804306ea6d07f332d49faa6d3d147905`
* Official release size: 500 questions

The leaderboard score is the mean of binary correctness multiplied by answer completeness. A
retrieval-only improvement is not a leaderboard claim. The current public comparison is the
homogeneous checked-in RE-call medium, no-correction result.

## Current measured baseline

RE-call baseline:

| metric | value |
|---|---:|
| combined leaderboard score | 46.16 |
| correctness | 63.80% |
| completeness | 53.23% |
| document recall | 77.34% |
| invalid extra documents | 6.94 |
| inserted board rank | 11 of 23 |
| top-five threshold | 61.03 |
| gap to fifth place | 14.87 points |

The mixed default run scored 48.03, but it mixes two GPT-5.4 reasoning settings across the 500
rows and must not replace the homogeneous comparator.

## Official category gaps

Use these official category aggregates to prioritize work. The RE-call score and best public score
are benchmark score values, not retrieval-only values.

| category | questions | RE-call | best public | best system | RE-call retrieval | exact coverage |
|---|---:|---:|---:|---|---:|---:|
| project_related | 40 | 8.55 | 51.53 | Troml | 51.3% | 17.5% |
| completeness | 20 | 18.08 | 49.41 | Troml | 52.3% | 20.0% |
| high_level | 10 | 25.00 | 77.50 | Troml | not applicable | not applicable |
| semantic | 125 | 36.07 | 65.04 | Troml | 75.2% | 75.2% |
| conflicting_info | 20 | 38.48 | 82.40 | RAGFlow | 72.5% | 65.0% |
| constrained | 30 | 49.27 | 88.42 | fgroo | 86.7% | 76.7% |
| basic | 175 | 54.27 | 88.36 | Troml | 81.1% | 81.1% |
| intra_document_reasoning | 40 | 62.50 | 92.50 | Troml | 90.0% | 90.0% |
| miscellaneous | 20 | 68.50 | 85.00 | Skyller | 100.0% | 100.0% |
| info_not_found | 20 | 100.00 | 100.00 | tied | not applicable | not applicable |

The largest retrieval opportunity is project_related. RE-call misses at least one expected
document on 33 of 40 questions and has complete expected-document coverage on only 7 of 40.
Completeness misses at least one expected document on 16 of 20 questions and has exact coverage
on 4 of 20. Conflicting information is mostly a two-document supersession problem. Semantic,
constrained, and intra-document reasoning already have relatively strong document recall, so
retrieval changes there must be justified by evidence-selection or answer-level measurements.

Metadata from the official questions:

| category | mean expected documents | mean answer facts |
|---|---:|---:|
| project_related | 4.22 | 11.72 |
| completeness | 6.50 | 14.20 |
| conflicting_info | 2.00 | 6.90 |
| constrained | 1.43 | 10.57 |

## What has already been tested

### Cheap model query expansion

The true cheap arm used `openai/gpt-5-mini` through OpenRouter, with the Voyage lexical index,
`candidate_k=200`, `k=8`, extractive answers, and paired no-reasoning baselines. Model execution
was verified by manifests and capture stability was 1.0.

| slice | n | baseline recall | cheap recall | recall delta | baseline exact | cheap exact | extra-doc delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| project_related | 5 | 46.7% | 46.7% | 0.0 pp | 20.0% | 20.0% | 0.0 |
| conflicting_info | 5 | 80.0% | 80.0% | 0.0 pp | 60.0% | 60.0% | +0.2 |
| completeness | 5 | 25.8% | 25.8% | 0.0 pp | 0.0% | 0.0% | 0.0 |
| semantic | 5 | 60.0% | 40.0% | -20.0 pp | 60.0% | 40.0% | +0.2 |

Generic cheap expansion is rejected as a promotion candidate. A depth retrieval arm was also
negative on the project dev slice in the v2 measurement: recall fell from 53.77% to 51.65%, with
two gains and four losses and no exact-coverage improvement.

### Voyage reranking

The first reranker used `voyage:rerank-2.5`, `candidate_k=200`, `k=8`, lexical sparse fusion,
Voyage embeddings, and a 4,000 character reranker document limit. It had a strong five-question
project dev result:

* recall 46.7% to 66.7%
* exact coverage 20.0% to 40.0%
* invalid extras decreased by 0.4 per question
* two gains and zero losses

The preregistered 23-question project confirmation did not reproduce the recall gain:

* recall 61.7% to 60.9%
* exact coverage 13.0% to 26.1%
* invalid extras increased by 0.22 per question
* six gains and six losses

The simple global reranker is therefore rejected for promotion. The exact-coverage increase may
justify testing an adaptive reranker or score calibration, but only with a new preregistration.

The follow-up rank blend also failed confirmation. A reciprocal rank blend with Voyage weight
`0.50` improved the 17-question project development recall from 53.77% to 55.57%, but on the
23-question confirmation recall moved from 61.71% to 61.59%, exact coverage stayed at 13.04%,
and invalid extras rose by 0.043 per question. It is rejected.

A deterministic `k=12` test with no reranker improved the same confirmation recall from 61.71% to
66.55% and exact coverage from 13.04% to 17.39%, but increased invalid extras by 3.61 per
question, above the preregistered 2.0 guardrail. Raw `k=12` is rejected globally. The next useful
hypothesis is selective depth based on runtime confidence, dense score gap, source coverage, or
other non-gold signals, with a smaller submitted set when confidence is high.

Selective depth was then tested with the development-selected rule `max_dense_score < 0.75`,
expanding from `k=8` to `k=12` for six of 17 development questions. On the 23-question held out
confirmation, recall fell from 61.71% to 60.75%, exact coverage fell from 13.04% to 8.70%, and
invalid extras rose by 0.96 per question. Mean retrieval latency was 31.6 seconds per question
over three captures. This candidate is rejected and no answer test was run.

### Answer-side reader tests

These are context for interpreting future retrieval work. They are not the focus of this session.
Using `openai/gpt-5-mini` as the fixed answer reader, category-aware conflict synthesis improved
the official 20-question conflicting aggregate from combined 34.4105 to 36.4875, with
completeness 60.53% to 64.04%, and unchanged correctness, retrieval, and invalid extras. It adds
only about 0.08 points to the 500-question overall score. The category-aware completeness policy
was rejected because it reduced the five-question dev combined score from 20.00 to 13.33.

## Environment and locations

The work is on VPS2. Use the isolated stage so the old checkout does not shadow the experiment:

* SSH host: `vps2`
* isolated code: `/home/sentiment/enterprise-rag-run/reasoning-366-next`
* benchmark: `/home/sentiment/enterprise-rag-run/EnterpriseRAG-Bench`
* Python: `/home/sentiment/recall-repos/.venv/bin/python`
* index database: `postgresql://sentiment@localhost:55432/recall_bench`
* index table: `ber_voy_lex_12k_full`
* tenant: `enterprise-rag-voyage-lexical-chunk12k-full`
* remote environment file: `/home/sentiment/enterprise-rag-run/RE-call/.env`

Do not print or copy secret values. The environment has `OPENROUTER_API_KEY`, `VOYAGE_API_KEY`,
and `RECALL_DSN`. When invoking the official evaluator, source the remote environment inside the
remote shell and set `LLM_API_KEY=$OPENROUTER_API_KEY` there. Do not let a local PowerShell shell
expand that variable to blank.

Local evidence is under `results/enterprise_rag/vps2_cheap_mini/`. The official downloaded data is
under `.benchdata/enterprise-rag-v1.0.0/`. The key implementation is
`benchmarks/enterprise_rag.py`; comparison helpers are in `scripts/enterprise_rag_compare.py`,
`scripts/enterprise_rag_make_slices.py`, and `scripts/enterprise_rag_retrieval_compare.py`.

## Rules for this session

1. Keep the answer reader fixed. This session studies retrieval, indexing, ranking, chunking,
   document selection, deterministic query normalization, and source diversity. Do not add model
   reasoning or answer-time decomposition as the independent variable.
2. Before every measurement that can decide whether an arm works, write and commit a dated
   preregistration containing the hypothesis, exact arm, primary metric, guardrails, slice, and
   promotion rule.
3. Start with retrieval-only extractive comparisons on frozen dev ids. Use paired per-question
   deltas and report recall, exact coverage, invalid extras, latency, provider calls, index
   configuration, and stability.
4. A candidate must pass a held-out confirmation slice before any answer-quality test. A dev-only
   result is not sufficient.
5. Use the same question hash, index, embedding model, sparse backend, `candidate_k`, `k`, answer
   context rules, and evaluator flags across paired arms.
6. Run the official evaluator with `--no-correction --skip-citation-stripping` for comparable
   diagnostics unless a preregistration explicitly says otherwise. Confirm judge responses parse.
7. Never claim a leaderboard improvement from a slice. A top-five claim requires a homogeneous
   500-question official evaluation and a comparison against the frozen public board.
8. Preserve all answer JSONL, manifests, metrics, retrieval comparison JSON, logs, prompt digests,
   and commands. Never expose credentials.

## Research questions worth testing

Prioritize approaches that can explain the dev versus confirmation reranker reversal:

* adaptive reranking only for queries with low lexical and dense agreement, with a fixed fallback
  to the baseline ranking;
* rank-score calibration and fusion weight sweeps using a small preregistered grid, not an
  unconstrained search on the confirmation rows;
* parent-document or source-level coverage selection that preserves multiple expected documents
  without allowing duplicate chunks to crowd out other documents;
* deterministic query normalization, identifier and project-name expansion, and source-aware
  lexical aliases, with no LLM-generated reasoning;
* chunk window or neighboring-chunk retrieval around high-scoring hits, evaluated with an invalid
  extra-document guardrail;
* adaptive `k` or candidate cutoff based only on retrieval scores, document diversity, or query
  length;
* reciprocal rank fusion, weighted hybrid variants, and calibrated score thresholds;
* category or source-specific retrieval policies that are selected before confirmation and then
  evaluated unchanged on held-out questions.

For each idea, explain why it should help project_related or completeness, what failure mode it
could create, and how it differs from reasoning. Prefer small, falsifiable experiments over a
large rewrite.

## Required deliverable

Return a research report and implementation changes that include:

* the exact hypothesis and preregistration commit for every measured arm;
* paired dev and held-out confirmation tables;
* per-question gain and loss ids, with category breakdowns;
* retrieval and answer-level metrics, latency, provider calls, and invalid extras;
* a clear accept or reject decision for each candidate;
* the best next experiment if no candidate confirms;
* a reproducible command for any accepted configuration;
* no top-five claim unless the homogeneous 500-question official evaluation is complete.

Begin by inspecting the current branch, existing preregistrations, and the official downloaded data.
Do not repeat the already rejected generic cheap expansion, v2 depth arm, or global Voyage reranker
without changing the hypothesis and preregistering the new test.
