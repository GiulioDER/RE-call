# EnterpriseRAG reasoning triage

**Date:** 2026-08-18
**Status:** analysis complete, full reasoning run pending

## Evidence boundary

This triage uses the public Hugging Face leaderboard Space snapshot at commit
`9d1fd7ec34cb137ba89f385741500116dbf9600f` and the official
`onyx-dot-app/EnterpriseRAG-Bench` repository at commit
`d36685e273713975ee20299bbf1ab64165575b3c`.

The Space snapshot contains the base questions, every submitted system's answer JSONL,
per-question evaluator results, and the generated leaderboard table. The official benchmark
repository is the source of the question and document release and the evaluator. RE-call's
comparison row is the checked-in `gpt-5.4` medium reasoning, no-correction summary from the
same benchmark commit.

The leaderboard score is the benchmark's per-question binary correctness multiplied by answer
completeness, averaged over questions. It is not the same as a retrieval-only score. The
leaderboard rows use the public evaluator artifacts, while the RE-call comparison summary is
explicitly no-correction and no citation stripping. The numbers below are therefore a triage
baseline, not a claim that the public board has already accepted RE-call.

## Current RE-call position

RE-call's checked-in medium, no-correction result is:

| score | correctness | completeness | document recall | invalid extra docs |
|---:|---:|---:|---:|---:|
| 46.16 | 63.80 | 53.23 | 77.34 | 6.94 |

If inserted into the downloaded board by this score, RE-call would be 11th of 23 rows. The
mixed default run scores 48.03, but it uses two different GPT-5.4 reasoning settings across the
500 rows and must not replace the homogeneous comparison row.

## Category comparison

The table compares RE-call with the best public competitor in each category. Category score is
the same binary correctness times completeness quantity used by the board. Retrieval recall and
exact coverage are computed directly from the RE-call answer document ids and the official
`expected_doc_ids`.

| category | n | RE-call score | best score | best system | score gap | RE-call retrieval | exact coverage | board rank if inserted |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| project_related | 40 | 8.55 | 51.53 | Troml | 42.98 | 51.3% | 17.5% | 19/23 |
| completeness | 20 | 18.08 | 49.41 | Troml | 31.34 | 52.3% | 20.0% | 12/23 |
| high_level | 10 | 25.00 | 77.50 | Troml | 52.50 | not applicable | not applicable | 21/23 |
| semantic | 125 | 36.07 | 65.04 | Troml | 28.97 | 75.2% | 75.2% | 6/23 |
| conflicting_info | 20 | 38.48 | 82.40 | RAGFlow | 43.92 | 72.5% | 65.0% | 21/23 |
| constrained | 30 | 49.27 | 88.42 | fgroo | 39.15 | 86.7% | 76.7% | 16/23 |
| basic | 175 | 54.27 | 88.36 | Troml | 34.09 | 81.1% | 81.1% | 12/23 |
| intra_document_reasoning | 40 | 62.50 | 92.50 | Troml | 30.00 | 90.0% | 90.0% | 9/23 |
| miscellaneous | 20 | 68.50 | 85.00 | Skyller | 16.50 | 100.0% | 100.0% | 13/23 |
| info_not_found | 20 | 100.00 | 100.00 | tied | 0.00 | not applicable | not applicable | 1/23 |

The strongest retrieval signal is project related work. RE-call misses at least one expected
document on 33 of 40 project questions, and only 7 of 40 retrieve the complete expected set.
On 15 of those missed questions, at least one of Troml, Skyller, OpenClaw, fgroo, OpenAI File
Search, RAGFlow, or BM25 both retrieved the complete expected set and received a correct answer.

Completeness is the second direct retrieval target. RE-call misses at least one expected document
on 16 of 20 questions, with exact coverage on 4 of 20. Five of those misses have a strong public
competitor that both retrieved the complete expected set and answered correctly.

Conflicting information is a different target. Seven of 20 questions miss an expected document,
and four of those misses have a strong competitor with complete retrieval and a correct answer.
The reasoning gain here must include conflict detection and evidence comparison. More retrieval
alone could increase unsupported or stale evidence.

Semantic questions are a generator and evidence-selection target, not an obvious retrieval
recall target. RE-call's document recall is 75.2%, higher than the best public competitor's
71.2%, yet its category score is only 36.07. A reasoning arm that only expands retrieval is not
expected to close this gap. It needs to improve query interpretation, evidence selection, and
answer synthesis while preserving citations.

The same distinction appears in intra-document reasoning and constrained questions. RE-call
already retrieves 90.0% and 86.7% respectively, but answer scores are 62.50 and 49.27. These
are candidate niches for bounded evidence planning and qualifier checking after retrieval.

The official question metadata explains why the first three lanes are structurally different:

| category | mean expected documents | mean answer facts | main source mix |
|---|---:|---:|---|
| project_related | 4.22 | 11.72 | Confluence, Jira, GitHub, Slack, Linear, Gmail, Google Drive |
| completeness | 6.50 | 14.20 | Confluence, Jira, Slack, Gmail, Google Drive, Fireflies, GitHub |
| conflicting_info | 2.00 | 6.90 | Confluence, Google Drive, Jira, Linear, Gmail, Fireflies, GitHub, HubSpot |
| constrained | 1.43 | 10.57 | Confluence, Jira, Slack, Google Drive, Linear |

Project and completeness questions are genuinely multi-document retrieval problems. Conflict
questions are mostly two-document supersession pairs, so they should be tested with a separate
contradiction policy rather than a larger undifferentiated context. The official methodology
also lists people-centric questions and true sequential multi-hop questions as future extensions;
those are good follow-on research directions after the released categories are measured.

## Research order

1. **Project related, 40 questions.** Test depth expansion and closed-loop decomposition. Measure
   whether missing project documents are recovered and whether added evidence improves the answer
   rather than merely increasing context noise.
2. **Conflicting information, 20 questions.** Test contradiction-aware evidence assembly. The
   result must report conflict detection, answer correctness, citation validity, and abstention or
   review outcomes separately.
3. **Completeness, 20 questions.** Test exhaustive multi-document coverage with a bounded stop
   rule. The primary risk is over-retrieval, so invalid extra documents and context size are
   first-class metrics.
4. **Constrained and intra-document reasoning, 70 questions together.** Test a deterministic
   qualifier checklist and bounded multi-hop planning. The retrieval ceiling is already close to
   the board, so this is mainly an answer-quality experiment.
5. **Semantic, 125 questions.** Test query rewriting and evidence selection only after the first
   three lanes show that the expansion mechanism can improve answer quality on a fixed slice.

High-level questions are useful for a later reasoning study, but they have only 10 rows and no
document recall denominator. Info-not-found is already perfect for RE-call in this comparison.

## Next measurement

Run the four #366 arms on the same frozen question ids and index:

* `none`, the current hybrid baseline
* `depth`, deterministic deeper retrieval
* `cheap`, model proposed expansion
* `closed_loop`, depth first, with cheap expansion only when the depth pass still reports a gap

Use three retrieval captures per question. Preserve the question hash, retrieval configuration,
provider identity, prompt digest, expansion cache, document ids before and after expansion, and
all per-question evaluator fields. Report paired deltas by category for document recall, exact
coverage, binary correctness, completeness, combined score, invalid extra documents, capture
stability, latency, and model calls.

The 20-question calibration fixture already reached 1.0000 document recall for depth, cheap, and
closed-loop arms. That result does not establish a full-benchmark gain because the arms shared
the same retrieval ceiling on that fixture. The full benchmark slice is the required test of
whether reasoning helps where the public comparison shows a real gap.

## Cheap-mini dev result

On 2026-08-18, the cheap arm was tested with `openai/gpt-5-mini` through the configured
OpenRouter provider. The test used the frozen `top5_slices/dev.ids` split, the Voyage lexical
index, `k=8`, `candidate_k=200`, no reranker, extractive answers, and paired no-reasoning
baselines. The mini provider actually ran: capture stability was 1.0 in every category, with
two to four expanded rows per five-question slice.

| slice | n | baseline recall | mini recall | recall delta | baseline exact | mini exact | extra-doc delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| project_related | 5 | 46.7% | 46.7% | 0.0 pp | 20.0% | 20.0% | 0.0 |
| conflicting_info | 5 | 80.0% | 80.0% | 0.0 pp | 60.0% | 60.0% | +0.2 |
| completeness | 5 | 25.8% | 25.8% | 0.0 pp | 0.0% | 0.0% | 0.0 |
| semantic | 5 | 60.0% | 40.0% | -20.0 pp | 60.0% | 40.0% | +0.2 |

This is a paired retrieval result, not a leaderboard score. The official evaluator's answer-level
diagnostic on the mini project slice returned zero correctness and completeness with empty judge
reasoning for all five rows, so that judge/provider path is not treated as evidence of answer
quality. The retrieval result is still sufficient to reject generic cheap expansion as a
promotion candidate: it produced no gain in the three primary weak lanes and harmed semantic
retrieval on this fixed sample.

The next experiment should therefore keep the hybrid retrieval set stable and spend reasoning
on evidence assembly: select a bounded representative set, detect contradictory or superseded
claims, and produce a fact checklist for completeness and constrained answers. Any reader arm
must be evaluated with a judge configuration whose responses parse successfully before its
answer score is used for promotion.

## First reader comparison

The first answer-side comparison used the same five project-related questions and the same
retrieved documents, with `openai/gpt-5-mini` generating the answer. The official metrics
evaluator was then run with the same model through OpenRouter, with `--no-correction` and
`--skip-citation-stripping`. The corrected environment was important: an earlier diagnostic had
expanded the API-key variable in the local PowerShell wrapper and silently produced blank judge
reasons.

| reader | correctness | completeness | combined score | document recall | invalid extras |
|---|---:|---:|---:|---:|---:|
| baseline | 0.0% | 24.86% | 0.000 | 46.67% | 7.0 |
| category-aware | 0.0% | 26.86% | 0.000 | 46.67% | 7.0 |

The category-aware reader improved completeness by 2.00 percentage points on this tiny dev
sample, with gains on qst_0341 and qst_0353 and a loss on qst_0346. Because correctness was zero
for all ten paired evaluations, this is a direction signal only, not a promotion result. It
supports a larger confirmation test of reader policies while leaving retrieval unchanged.

## Reader policy by category

The next paired dev tests separated the policies instead of assuming one reader strategy fits all
categories. On five conflicting-information questions, category-aware synthesis improved the
combined score from 35.50 to 37.50 and completeness from 60.83% to 66.83%, with correctness,
document recall, and invalid extras unchanged. On five completeness questions, the same policy
reduced the combined score from 20.00 to 13.33. Preserving all selected chunks instead of one
chunk per document recovered part of that loss (17.68), but remained below baseline.

The conflict-only result held on the 12-question confirmation slice:

| slice | reader | correctness | completeness | combined | recall | invalid extras |
|---|---|---:|---:|---:|---:|---:|
| conflicting dev, n=5 | baseline | 40.0% | 60.83% | 35.50 | 80.0% | 6.0 |
| conflicting dev, n=5 | category-aware | 40.0% | 66.83% | 37.50 | 80.0% | 6.0 |
| conflicting confirmation, n=12 | baseline | 50.0% | 64.11% | 42.56 | 87.5% | 6.25 |
| conflicting confirmation, n=12 | category-aware | 50.0% | 65.07% | 45.19 | 87.5% | 6.25 |

This supports a conflict-only reader promotion candidate for the next full-category run. The
completeness reader remains rejected. The implementation now preserves all submitted chunks for
completeness and other multi-evidence categories, while retaining one representative chunk per
document only for `conflicting_info`; this keeps the citation set aligned without throwing away
facts needed by checklist questions.

The remaining three dev questions completed the full conflict-category screen. Across all 20
official conflicting-information questions, the paired aggregate was:

| reader | correctness | completeness | combined | recall | invalid extras |
|---|---:|---:|---:|---:|---:|
| baseline | 40.0% | 60.53% | 34.4105 | 77.5% | 6.35 |
| category-aware | 40.0% | 64.04% | 36.4875 | 77.5% | 6.35 |

The +2.0775 category-point gain is real on the official paired rows, but it contributes only about
0.08 points to a 500-question overall score. It is therefore a safe localized promotion candidate,
not by itself a path from RE-call's 46.16 baseline to the 61.03 top-five threshold.

## Non-reasoning retrieval experiments

The first non-reasoning retrieval candidate was Voyage reranking. On the five-question project
dev slice, `voyage:rerank-2.5` with `candidate_k=200`, `k=8`, lexical sparse fusion, and a 4,000
character reranker document limit improved document recall from 46.7% to 66.7%, exact coverage
from 20.0% to 40.0%, and reduced invalid extras by 0.4 per question. This result was preregistered
as a retrieval experiment and looked promising enough to require held-out confirmation.

The 23-question project confirmation did not reproduce the recall gain. Exact coverage doubled,
from 13.0% to 26.1%, but mean document recall fell from 61.7% to 60.9%. The paired result had
six recall gains and six losses, while invalid extras increased by 0.22 per question. The simple
reranker configuration therefore fails its confirmation gate and is rejected for promotion. The
exact-coverage increase is a diagnostic signal for a possible adaptive or calibrated reranker,
not evidence for enabling the current global reranker.

| reranker slice | n | baseline recall | reranker recall | baseline exact | reranker exact | extra-doc delta | gains/losses |
|---|---:|---:|---:|---:|---:|---:|---:|
| project dev | 5 | 46.7% | 66.7% | 20.0% | 40.0% | -0.4 | 2/0 |
| project confirmation | 23 | 61.7% | 60.9% | 13.0% | 26.1% | +0.22 | 6/6 |

These results point toward non-reasoning work on rank calibration, adaptive reranking, source
diversity, and chunk or parent-document selection. Any new arm must keep the answer reader fixed
and report retrieval deltas before answer quality is measured. A dev result alone is insufficient;
the next candidate needs a frozen held-out confirmation before an answer-side test.

## Follow-up non-reasoning confirmations

The first follow-up preserved the original hybrid order and blended it with Voyage reranker order
using reciprocal rank fusion with constant 60 and weight 0.50. On the 17-question project
development split, recall improved from 53.77% to 55.57%, exact coverage stayed at 23.53%, and
invalid extras fell by 0.06 per question. On the preregistered 23-question confirmation, recall
was 61.71% for baseline and 61.59% for the blend, exact coverage stayed at 13.04%, and invalid
extras increased by 0.043. The blend is rejected because the held-out recall prediction failed.

The second follow-up increased deterministic retrieval depth from `k=8` to `k=12` with no
reranker. On the same 23-question project confirmation, recall improved from 61.71% to 66.55%,
exact coverage from 13.04% to 17.39%, and there were six gains against one loss. Invalid extras
increased by 3.61 per question, exceeding the preregistered 2.0 guardrail. Raw `k=12` is rejected
as a global setting. The result justifies a selective depth policy based on runtime confidence or
source coverage, not a blanket increase in submitted documents.

The development-only selective-depth screen tested two non-gold dense-score features against the
fixed threshold grid. The selected arm uses `max_dense_score < 0.75`, expands six of 17 questions,
improves mean recall by 3.43 points, leaves exact coverage unchanged, and adds 1.24 invalid
documents per question. Feature selection was exploratory because the parent preregistration did
not establish a priority order. The choice and confirmation gate are recorded in a separate
amendment before held out measurement.

The first held out selective-depth capture was invalid because it retrieved depth 12 before deciding
whether to expand. After correcting the arm to retrieve depth 8 first and perform a second depth 12
pass only when the frozen confidence rule fires, the three-capture confirmation rejected the
hypothesis. Recall fell from 61.71% to 60.27%, exact coverage fell from 13.04% to 8.70%, and
invalid extras rose by 0.83 per question. The candidate had one gain and two losses, capture
stability 1.0, 84 embedding and lexical calls, 27.47 seconds mean latency, and 43.28 seconds p95
latency. No answer-quality comparison is authorized.
