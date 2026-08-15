# Pre-registration: what does routing EnterpriseRAG-Bench through the product cost?

Written 2026-08-15, **before the library arm has been run against the benchmark**. Committed ahead
of the result so the prediction cannot be revised to fit it, per `docs/RESEARCH_PROTOCOL.md`.

## Registration

```yaml
registration_commit: PENDING
registration_authored: PENDING
baseline_artifact: benchmarks/artifacts/enterprise_rag/re_call_voyage_splade_gpt4o.judge_gpt54_medium.no_correction.summary.json
baseline_judge: gpt-5.4 medium reasoning, no correction, no citation stripping
```

⚠️ The two `PENDING` fields are filled by the commit that lands this file, in the same commit, and
the runner refuses an artifact generated at or before `registration_authored`. This is the same
mechanism `results/truth_extraction/PREREGISTRATION-prose-extraction.md` uses, for the same reason:
until it existed, "the prediction came first" was a sentence rather than an assertion.

---

## The question

**The submitted leaderboard score has never measured RE-call's generation path.**
`benchmarks/enterprise_rag.py` imports `recall.retriever` and `recall.store` and writes its own
prompt at `generated_answer`. It imports neither `recall.evidence` nor `recall.trust` nor
`recall.reasoning`. So a reader who takes the score as evidence about the product is reading
evidence about a 90 word prompt in a benchmark file.

`--answer-mode library` routes generation through `recall.evidence.generate_from_evidence`, the
product's actual boundary. **This experiment measures the cost of that swap, and nothing else.** No
answer side improvement is in the diff that introduces it, deliberately: this repository has a
documented history of confounding an improvement with a substrate swap, and a parity number is the
only thing that makes a later improvement attributable.

## Baseline, measured, from the committed artifact

Correctness only. It is the ranked metric and the judge stable one; completeness moves 10.56 points
on judge choice alone and a gain under about 10 points there cannot be claimed.

| category | n | correctness |
|---|---:|---:|
| basic | 175 | 70.29 |
| semantic | 125 | 58.40 |
| intra_document_reasoning | 40 | 75.00 |
| project_related | 40 | 32.50 |
| constrained | 30 | 63.33 |
| completeness | 20 | 35.00 |
| conflicting_info | 20 | 50.00 |
| **info_not_found** | 20 | **100.00** |
| miscellaneous | 20 | 100.00 |
| high_level | 10 | 40.00 |
| **aggregate** | **500** | **63.80** |

## What differs between the arms, enumerated before the run

A delta is only attributable if the differences are listed in advance. Four remain after the
apparatus holds everything else constant, and the arm is built so this list is exhaustive.

| # | difference | held constant? |
|---|---|---|
| D1 | **The system prompt.** 46 words, envelope shaped, no format guidance, against a tuned 90 word prompt naming specificity, conflict handling and brevity. | no, this IS the thing measured |
| D2 | **Citations are required.** `validate_answer` refuses an answer with no citation and refuses any citation that does not resolve to a bundle chunk id. The bespoke prompt says verbatim *"Do not include inline citations"*. | no |
| D3 | **Output is a strict JSON envelope**, exactly `{answer, citations, insufficient_evidence}`, against free text. | no |
| D4 | **`question_type` is not passed.** The bespoke arm puts it in the prompt; `render_evidence_prompt` takes the bundle alone and has no channel for it. | no |
| — | **the model** | **yes**: the runner overrides `RECALL_REASONING_MODEL` with `--model`, so both arms answer with the same one |
| — | **the completion ceiling** | **yes**: 16,384 tokens in both arms, and a `finish_reason == "length"` reply is raised rather than scored |
| — | evidence item count | **yes**: `--evidence-max-items` defaults to `--k`, so both arms see 8. The shipped `EvidencePolicy` default of 5 would confound a prompt delta with a context size delta |
| — | evidence text bytes | **yes**: the same per hit character budget, asserted against `generated_answer` itself in `tests/test_enterprise_rag_library_arm.py` |
| — | retrieval and `document_ids` | **yes**: identical code path, unchanged |
| — | the trust layer | **yes, by NOT running it.** Every hit enters as `ok` |

⚠️ **The first two rows were confounds in the first version of this apparatus, and both were
silent.** The library provider defaulted to `openai/gpt-4o-mini` while the baseline used
`openai/gpt-4o`, and `--model` never reached it, so the manifest recorded `"model": null` and the
"prompt delta" would have been a prompt delta plus a model swap. The completion ceiling was 1024
against the bespoke arm's 16,384, on the path that needs MORE budget because the envelope wraps
the prose in JSON; a truncated reply became unterminated JSON, then a refused envelope, then an
empty answer a judge scores as wrong. Both are recorded here rather than quietly fixed, because
the enumerated list above is the thing this experiment's attributability rests on and it was
wrong once already.

⚠️ **D4 and the trust layer are omissions I am choosing, and both could be argued the other way.**
The trust layer is excluded because its dense floor demotes at least 5.6% of questions and reports
`false_confident_rate_on_info_not_found = 1.0` at 0.50; running it here would move `info_not_found`
and make the delta a sum of two changes. Measuring the trust layer is a separate arm with its own
prediction.

## Predictions

| # | quantity | point | interval |
|---|---|---|---|
| B1 | Δ aggregate correctness, library minus bespoke | **−6.0** | −15.0 to +1.0 |
| B2 | `info_not_found` correctness, library arm | 95.0 | **≥ 90.0** |
| B3 | rows failing `validate_answer` | 1.0% | 0% to 5% |
| B4 | rows returning `insufficient_evidence=true` | 6% | 2% to 20% |
| B5 | median answer length, characters | **420** | 250 to 900 |
| B6 | mean citations per answered row | 2.0 | 1.0 to 4.0 |
| B7 | Δ correctness on `basic` | −5.0 | −15.0 to +2.0 |
| B8 | Δ correctness on `semantic` | −5.0 | −15.0 to +2.0 |

**Ordering predictions**, harder to hit by luck than the levels:

- **O1.** The library arm's answers are **longer** than the bespoke arm's 218 character median.
  The bespoke prompt asks for *"the shortest complete answer"* and the library prompt says nothing
  about length. If B5 comes in at or below 218, the length was never coming from that instruction
  and the "answers are too short" thread is dead on different evidence than the judge artifact
  killed it with.
- **O2.** The largest per category correctness drop is in `high_level` or `completeness`, the two
  that need synthesis across evidence, rather than in `basic`. A 46 word prompt with no guidance
  should cost most where the answer is not a lookup.
- **O3.** `info_not_found` does not move at all. Its 100.0 comes from the retrieval returning
  nothing usable, which both arms see identically, and `generate_from_evidence` short circuits an
  empty bundle without invoking the generator.

## Reasoning, heaviest first

1. **I expect a loss, and a loss is the useful outcome.** The bespoke prompt was tuned against this
   benchmark and the library prompt was written as a security boundary for a different purpose. The
   delta is the price of measuring the product instead of a harness, and knowing it is what makes
   every later answer side change attributable. A parity arm that came out ahead would be the
   surprising result and I would check the apparatus first.
2. **The citation requirement is the largest single unknown.** It could help, by forcing the model
   to ground each claim in a retrieved chunk, or hurt, by pushing it toward `insufficient_evidence`
   when it cannot tie a true statement to one. B3 and B4 exist to separate those two, and B4's
   wide interval is honest about how little I can predict it.
3. **`info_not_found` is the one thing I am protecting.** It is at 100.0 and it is the only
   category where the abstention machinery is already doing exactly what it should. B2 is an
   invariant, not a target.
4. **The JSON envelope is a format risk, not a reasoning risk.** A model asked for strict JSON
   sometimes truncates prose to fit; `max_tokens` is set at 1024 rather than omitted, which caps
   that exposure but does not remove it. B5's upper interval is wide for this reason.

## Invariants, and the vacuous version of each

| # | invariant | assert how | vacuous version to avoid |
|---|---|---|---|
| J1 | the two arms retrieved identically | `document_ids` byte identical per `question_id` across the two answer files | comparing aggregate recall, which is equal for many different lists |
| J2 | the prompt did not move | `system_prompt_sha256` in the manifest matches the shipped `SYSTEM_PROMPT` | asserting the constant exists |
| J3 | context size held | `evidence_items == k` on every row's diagnostics | asserting `max_items` was passed, which does not prove it took effect |
| J4 | no corpus byte in the instruction channel | assert on the bytes the PROVIDER received, not on the template | asserting `render_evidence_prompt` returns `SYSTEM_PROMPT` |
| J5 | the arm is not silently broken | `refuse_a_broken_library_arm` raises `SystemExit` above B3's ceiling, after the rows are on disk and BEFORE the manifest, so a refused run keeps its evidence and has no manifest to be mistaken for a result | reporting a mean over the rows that happened to succeed. Until this was written the runner wrote a full file of empty answers, printed a success line and exited 0 |
| J7 | the diagnostics survive the run | the per row library diagnostics are aggregated into `answering.library.tally` in the manifest | leaving them on the in-memory rows, which `write_answers_stream` strips before writing, so B3 and B6 were unmeasurable and J3 had nothing to read |
| J6 | the judge is pinned and recorded | judge model and reasoning level written INSIDE the artifact | a filename, which is what the two committed summaries have, both carrying `evaluator_options: null` |

J6 is a convention change, not a check on this run. Two of the three committed summaries record
their configuration nowhere but in the filename, so their numbers are reproducible only by someone
who already knows how they were produced.

## What falsifies this

- **B2 fails** (`info_not_found` below 90.0): stop. Either the apparatus is wrong or the library
  path abstains differently, and neither is something to trade correctness against.
- **B3 fails high** (over 5% of rows failing validation): **APPARATUS FAILURE.** Publish no number.
  A mean over the surviving rows is a mean over the questions the model happened to format
  correctly, which is a different population.
- **B1 comes in positive beyond +1.0**: check J1 first. A library arm that beats a tuned prompt on
  its own benchmark is more likely a retrieval difference than a prompt win.
- **B1 fails low** (worse than −15.0): the library boundary is not usable for this benchmark as it
  stands, and the finding is that, reported as such rather than patched over in the same session.

## Decision rule, fixed in advance

| outcome | verdict | action |
|---|---|---|
| B3 over 5%, or J1 to J4 fail | **APPARATUS FAILURE** | publish no number. Precedence over every row below |
| B2 below 90.0 | **ABSTENTION REGRESSION** | investigate before any tuning; do not trade this for correctness |
| Δ correctness within ±1.0 | **PARITY** | the bespoke prompt was buying nothing measurable. Re-point the submission at the library and delete the bespoke path |
| Δ correctness −1.0 to −15.0 | **MEASURED COST** | this is the expected outcome. The delta is the budget a later answer control layer has to earn back before it has improved anything |
| Δ correctness worse than −15.0 | **NOT USABLE AS IS** | report it. Do not begin tuning in the same session that discovered it |

**No tuned run may be compared against anything but this parity number.** Comparing a tuned library
arm against the bespoke arm would report the tuning and the substrate swap as one figure, which is
the confound this whole exercise exists to avoid.

## What this does not settle

- **Anything about the trust layer.** Deliberately not run here.
- **Whether the library path is better or worse in production.** This is one benchmark, one judge,
  one model, and the answer envelope exists for a security property this benchmark does not score.
- **The `basic` and `semantic` deficit**, which is 59 percent of the correctness headroom. Parity
  is the substrate for that work, not the work.
- **Cost.** The library arm makes one model call per question, the same as the bespoke arm, but
  token counts differ with the framing and are not predicted here.
