# All 500 rows, segmented: where could reasoning actually help?

Measured 2026-08-15 from [`judgements.gpt-5.4.medium.json`](judgements.gpt-5.4.medium.json) joined
to the question structure in `questions.jsonl`. No model calls. Cross-checked against
[`judgements.gpt-4o.default.json`](judgements.gpt-4o.default.json) wherever a claim depends on the
judge.

**Why this exists.** Every prior pass at this, including my own, scoped "reasoning" to the
`conflicting_info` category, because that is the category whose NAME says conflict. That category
is 20 questions and **1.2 of the 36.2 aggregate points** available. Scoping by category label was
the mistake. This segments all 500 rows by what the question structurally REQUIRES instead.

## The whole deficit, segmented

181 of 500 rows are wrong, worth 36.2 aggregate correctness points. Every row is 0.2 points.

| segment | n | wrong | agg pts | can answer-side reasoning reach it? |
|---|---:|---:|---:|---|
| **single-doc, gold NEVER retrieved** | 68 | 64 | **12.8** | no. The evidence is absent |
| **single-doc, gold fully retrieved** | 309 | 51 | **10.2** | yes: extraction and coverage |
| **multi-doc, gold PARTIALLY retrieved** | 55 | 36 | **7.2** | no, but retrieval-side reasoning can |
| multi-doc, gold fully retrieved | 30 | 16 | 3.2 | yes: synthesis, the original target |
| multi-doc, gold never retrieved | 8 | 8 | 1.6 | no |
| no gold documents (`high_level`) | 30 | 6 | 1.2 | yes, but tiny |

## Finding 1: the multi-document penalty is large and judge-stable

Restricting to rows where the gold was FULLY retrieved, so retrieval is not the explanation:

| | single-doc | multi-doc | gap |
|---|---:|---:|---:|
| correct, `gpt-5.4 medium` | 83.5% | 46.7% | **36.8** |
| correct, `gpt-4o` | 82.8% | 50.0% | **32.8** |

Both judges agree within 4 points. That matters because the completeness metric moved **10.56
points on judge choice alone**, which is what withdrew the earlier "answers are too short" thesis.
**This effect is roughly three times that noise floor and it survives the judge swap.** It is the
most robust signal in the dataset.

Across ALL multi-doc questions regardless of retrieval, correctness is **35.5%** against 69.5% for
single-doc. Multi-document questions are the worst-served segment of this benchmark.

## Finding 2: required-fact count degrades correctness monotonically

Answer-control rows only (gold fully retrieved, n=339):

| required facts | n | correct |
|---:|---:|---:|
| 1 | 51 | 86.3% |
| 2 | 73 | 83.6% |
| 3 | 65 | 83.1% |
| 4 to 5 | 74 | 81.1% |
| **6 or more** | 76 | **69.7%** |

A 17-point drop from one fact to six. **But it is largely the same phenomenon as finding 1**: of
the 23 compound-question failures, **14 overlap** the 16 multi-doc failures. Compound questions
tend to be multi-document questions. They should be treated as one target, not two.

## Finding 3, and it is the one that reframes the work

🔑 **The largest reasoning-shaped pool is on the RETRIEVAL side, not the answer side.**

Group the deficit by what would have to change to fix it:

| what would have to change | agg pts | share |
|---|---:|---:|
| **retrieval finds evidence it currently misses** | **21.6** | **60%** |
| answer uses evidence it already has | 13.4 | 37% |
| neither (no gold documents exist) | 1.2 | 3% |

The 21.6 splits into two shapes, and both are things a *planner* does:

- **12.8 points: single-doc, zero retrieval.** 68 questions, 64 of them wrong, a 94% failure rate.
  The one document needed never came back. The reasoning move is **query reformulation**: notice
  that nothing retrieved is on-topic, and ask differently.
- **7.2 points: multi-doc, partial retrieval.** 55 questions where retrieval returned SOME of the
  needed documents. The reasoning move is **decomposition and a second hop**: notice the question
  needs several sources, notice only some are present, and go back for the rest.

⚠️ Those 55 partial-retrieval rows are `project_related` (20), `completeness` (9), `constrained`
(6) and `conflicting_info` (1). **Not one of them is in a category anyone labelled as reasoning.**

## What this says about the architecture

`recall/reasoning_planner.py` already has a frontier, operations and a budget: it is shaped like
something that decides it needs more evidence and goes to get it. Two things stop it doing that
here:

1. **`ReasoningBudget.max_model_calls` is 0** and the planner never calls a model, so it cannot
   reformulate a query.
2. **The evidence bundle is built BEFORE the planner runs**, and `render_evidence_prompt` takes the
   bundle alone, so nothing the planner concludes can send retrieval back for another pass.

The second is the same structural gap that makes `check_contradiction`'s output unusable. It is one
gap, and closing it unlocks the 21.6-point pool rather than the 3.2-point one.

## What I am not claiming

- **Not that 21.6 points are winnable.** They are the rows where BETTER RETRIEVAL is a
  precondition. Whether a reformulation or a second hop actually finds the missing document is
  exactly the untested question, and on the 68 single-doc misses it may be that the corpus phrasing
  is simply too far from the question.
- **Not that the multi-doc gap is caused by reasoning failure.** It is measured as a correlation
  between structure and correctness. A compound question is also a longer question, and length
  confounds.
- **Not that answer-side work is worthless.** It is 13.4 points, the largest single cell is 10.2,
  and it needs no new retrieval.
- **Nothing about `high_level`.** 30 rows, no gold documents, 1.2 points.

## What changes, concretely

The supersession probe already pre-registered and set up
([`PREREGISTRATION-supersession-annotation.md`](PREREGISTRATION-supersession-annotation.md))
targets the **3.2-point cell**, and within it only the four rows whose conflict is a supersession.
Its apparatus is built, its evidence and anchors are frozen, and it costs pennies to finish, so
finishing it is cheap and it still answers "does annotating evidence change the answer".

But it should stop being described as the main line. **The main line, on this evidence, is
reasoning that acts on retrieval**: reformulate when nothing lands, decompose when the question
needs several sources. That is a bigger change and it needs its own pre-registration.
