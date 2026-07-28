# BEAM harness provenance

The BEAM arm exists to produce RE-call cells that are comparable with the numbers Mem0 published
for its "token-efficient memory algorithm". Comparability here is a property of the *whole*
(retrieval, answerer, judge, metric) stack, so every part of that stack that is not RE-call is
taken from Mem0's own harness rather than reimplemented.

## What is vendored

| File | Source | Upstream commit | SHA-256 at vendoring |
|------|--------|-----------------|----------------------|
| `prompts.py` | `benchmarks/beam/prompts.py` in [mem0ai/memory-benchmarks](https://github.com/mem0ai/memory-benchmarks) (Apache-2.0) | `4b61c5d31b9c668a12b4f5e78064248a02c82d2b` (2026-05-13) | `0833206ac40cddfa6067e047cde84bb541509c6199ddf177bdad8badfff0d969` |

Vendored 2026-07-26. Only a header docstring was added; the prompt bodies are byte-identical.
Re-copy wholesale if upstream changes them — a prompt edit invalidates every cell produced with
the old one, so it is a re-run, not a patch.

## What Mem0's published BEAM numbers actually are

Read from `results/platform/beam_1m_results.json` and `beam_10m_results.json` in the same
upstream repo, which carry per-question evaluations for the published run.

| Field | Value | Why it matters here |
|-------|-------|---------------------|
| Answerer model | **`gpt-5`** (Azure) | NOT gpt-4o. The BEAM runner's CLI default is `gpt-5`; the README's "default: GPT-4o" describes the generic pipeline, not this benchmark. |
| Judge model | **`gpt-5`** (Azure) | Same. |
| Retrieval cutoff | `top_200` (a `top_50` variant is published alongside) | The headline number is the `top_200` one. |
| Backend | **Mem0 Cloud / platform** | `results/platform/`, not `results/oss/`. Mem0's own blog states the platform "includes proprietary optimizations" absent from the OSS SDK, so the published cells are NOT reproducible from `pip install mem0ai` at any version. There are no published OSS BEAM results. |
| Headline metric | **`avg_score`**, the mean rubric-nugget score | 1M: `avg_score` 0.6409 → the published **64.1**. 10M: 0.486 → **48.6**. The pass-rate (`accuracy`, score ≥ 0.5) is a *different* number: 70.14 % and 50.5 %. Quoting the pass-rate as if it were the headline would overstate both systems by ~6 points. |
| n | 700 (1M, 35 conversations × 20 questions), 200 (10M) | Matches the dataset splits below. |

## Dataset

`Mohammadta/BEAM` on HuggingFace, split `1M` → `data/1M-00000-of-00001.parquet` (66 MB).
35 conversations, 700 probing questions, 10 ability types (20 questions per conversation,
2 per type), ~40 M tokens of dialogue in total (~1.14 M per conversation).

`datasets.load_dataset` imports pandas, which is DLL-blocked on the author's Windows machine, so
`dataset.py` reads the parquet directly with pyarrow. Same bytes, one less dependency.

## Why the Mem0 arm is re-judged rather than re-run

The published artifact stores, per question id, the retrieved-memory count, the **generated answer**
and the per-nugget scores. That makes a genuinely *paired* comparison possible without spending a
cent on Mem0: run RE-call over the same 700 question ids with the same answerer, then score BOTH
systems' answers with the SAME judge instance in the same session.

This removes the confound a naive comparison would carry — their judge run (Azure gpt-5, May 2026)
versus ours (OpenRouter gpt-5, now) — and isolates what the benchmark is supposed to isolate: what
the memory layer retrieved and what the answerer did with it. The cost is that Mem0's *answers*
are historical: their platform may have moved since. That is stated wherever these cells are
published, and it is the honest direction of the trade — an historical answer scored by today's
judge is comparable; today's answer scored by two different judges is not.

The re-judged Mem0 score is reported alongside their published score. If the two diverge materially,
the divergence is a judge-drift measurement in its own right and is reported as such, not silently
resolved in either direction.
