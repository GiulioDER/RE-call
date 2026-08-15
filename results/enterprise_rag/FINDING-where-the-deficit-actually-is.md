# Where the EnterpriseRAG deficit actually is: mostly retrieval, not answer control

Measured 2026-08-15 from the per question judge output recovered from VPS2, against the
`gpt-5.4 medium` judge, no correction pass. Artifacts:
[`judgements.gpt-5.4.medium.json`](judgements.gpt-5.4.medium.json) and
[`judgements.gpt-4o.default.json`](judgements.gpt-4o.default.json), 500 rows each.

**This changes what Track B should build.** The plan that commissioned Track B routed effort to
answer control on the strength of a category level deficit table. The per question judgements were
never copied off the scoring host, so nobody had checked the one thing that decides whether answer
control can help at all: **on a failing row, was the gold document even retrieved?**

## The measurement

Every row carries `document_recall_pct`, the share of that question's gold documents present in
the submitted list. A failing row with recall 0 could not have been answered correctly by any
generator, because the evidence was never in the context.

| category | wrong | recall 0 | partial | recall 100 | no gold docs |
|---|---:|---:|---:|---:|---:|
| basic | 52 | **30** | 0 | 22 | 0 |
| semantic | 52 | **30** | 0 | 22 | 0 |
| project_related | 27 | 2 | 20 | 5 | 0 |
| completeness | 13 | 1 | 9 | 3 | 0 |
| constrained | 11 | 1 | 6 | 4 | 0 |
| intra_document_reasoning | 10 | 4 | 0 | 6 | 0 |
| conflicting_info | 10 | 4 | 1 | 5 | 0 |
| high_level | 6 | 0 | 0 | 0 | 6 |
| **total** | **181** | **72** | **36** | **67** | **6** |

`basic` and `semantic` are **104 of the 181 failures**, 57 percent, which confirms the routing
correction that put them at the top of the list. What is new is the split inside them:

- **60 of 104 (58%) retrieved no gold document at all.** These are retrieval failures. No prompt,
  no citation policy and no synthesis layer can reach them.
- **44 of 104 (42%) had the gold document fully retrieved and still answered wrong.** These are
  the answer control ones.

Neither category has a single partially retrieved failure, which fits: these are single document
lookups, so recall is 0 or 100 and nothing else.

## What this caps

Answer control can address at most the **67 rows** where the gold evidence was fully present and
the answer was still wrong. Each row is 0.2 aggregate correctness points, so the ceiling on pure
answer control is **13.4 aggregate points**, and that assumes perfect answering on every one of
them. The remaining 108 failing rows are 72 outright retrieval misses, 36 partial retrievals, and
6 `high_level` rows that have no gold documents to score recall against.

This is both better and worse than the plan's estimate. The plan put the honest ceiling at 3.20
points, counting only `conflicting_info` and `high_level` as clean answer control failures; the
real figure is four times that, because 22 `basic` and 22 `semantic` rows are answer control
failures nobody had counted. But it also says **the majority of the headroom in the two biggest
categories is retrieval**, which is exactly the direction the plan's Correction 1 established for
`project_related` and `completeness` and then did not test for `basic` and `semantic`.

## What I am not claiming

- **Not that the 67 are winnable.** "Gold document retrieved" is not "answer derivable from the
  retrieved chunks". A document can be in the submitted list while the chunk carrying the fact is
  not in the top k, and this measurement cannot see that. 13.4 points is a ceiling, not a forecast.
- **Not a judge independent result.** The split is computed under one judge. `document_recall_pct`
  is mechanical rather than judged, but `answer_correct` is not, and the second artifact exists so
  the same split can be recomputed under `gpt-4o`.
- **Nothing about the parity arm.** That is a separate, pre-registered experiment
  ([`PREREGISTRATION-library-parity.md`](PREREGISTRATION-library-parity.md)) and it has not run
  against this substrate.

## Provenance, and the convention it fixes

Recovered from `vps2:/home/sentiment/enterprise-rag-run/EnterpriseRAG-Bench/answer_evaluation/`.
The published aggregate recomputes exactly from the 500 rows, 63.80 against a published 63.80, so
the artifact's summary and body agree.

The judge's `correctness_reasoning` field is **dropped**, not published. It paraphrases and
sometimes quotes the benchmark's gold answers, and this repository is public. What is kept is
judgements and metrics only, with a longest per row value of 24 characters.

Both artifacts record the judge model, its reasoning level and whether the correction pass ran
**inside the file**. Two of the three previously committed summaries carry `evaluator_options:
null` and an empty judge block, so their configuration was recoverable only from a filename. That
is the convention this changes.
