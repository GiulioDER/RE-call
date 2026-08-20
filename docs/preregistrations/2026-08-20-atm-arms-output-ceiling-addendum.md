# ATM answer-arm output ceiling addendum

**Date:** 2026-08-20   **Status:** preregistered before resuming the arms

Addendum to `2026-08-20-atm-evidence-allocation-and-selection.md`. It follows the same practice as
the three addenda of `2026-08-19-atm-full-voyage4-deepseek.md`, which raised the ceiling 1,024 to
2,048 to 4,096 to 8,192 for the same reason.

## What happened

`A-greedy-baseline` completed all 300 questions at a ceiling of 8,192, spending a mean of 466
completion tokens. `B-allocated-baseline` then died at 266 of 300 with `CompletionTruncated`: one
question reached the same 8,192 ceiling. Because the driver runs the arms under `set -e`, arms C
and D never started.

OpenRouter counts reasoning tokens inside the completion ceiling, so a question whose reasoning
runs long consumes the budget the answer needs. The mean is not the quantity that matters here and
never was: A's mean of 466 sits a factor of 17 below the ceiling that killed its sibling.

## Change

1. The ceiling rises from 8,192 to 16,384 for the arms that have not completed. Retrieval,
   evidence packing, prompts, model, reasoning effort, the question subset and the judge
   configuration are unchanged.
2. Each question's `completion_tokens` is now recorded in `diagnostics.jsonl`, so the question
   "did the ceiling distort this arm" is answerable from the artifact instead of by argument.

`B-allocated-baseline` resumes from its 266 checkpointed answers. They were produced at the 8,192
ceiling and are kept: a ceiling only affects a completion that REACHES it, and every one of those
266 finished on its own.

## What I predict

1. No further arm dies of the ceiling at 16,384.
2. **Fewer than 5 of the 900 remaining answers will exceed 8,192 completion tokens.** This is the
   number that decides whether `A-greedy-baseline` is comparable to the rest: if the count is
   near zero, A's lower ceiling never bound and the arms are comparable; if it is large, A was
   silently running a different experiment and must be rerun at 16,384 before any arm is compared
   with it.
3. The ceiling change does not move any quality metric on the questions already answered, because
   it cannot: it changes only which completions are allowed to finish.

## What would falsify this

* Another `CompletionTruncated` at 16,384. Then the tail is heavier than two data points suggested
  and the fix is not a bigger number, it is a reasoning effort or a model that terminates.
* 5 or more completions above 8,192 tokens. Then arm A must be rerun at the same ceiling as the
  others and its published deterministic result is provisional until it is.

## Confound

Arm A ran at 8,192 and the others will run at 16,384. That is a configuration difference between
arms, and prediction 2 exists precisely to measure whether it mattered rather than to assume it did
not. If prediction 2 fails, the honest response is a rerun of A, not a footnote.

## Result

Not yet measured at the time this record was committed.

---

## Judge route addendum and its agreement measurement (2026-08-20)

The OpenAI account holding the official judge key ran out of credit part way through the
comparison: `insufficient_quota` on every call, and the configured fallback to `gpt-4o-mini` bills
the same account, so it failed identically. The judge retried a request that could not succeed for
twelve minutes before it was stopped.

The judge now reaches the same model through OpenRouter, using the transport patch committed at
`scripts/atm_judge_openrouter.patch`. It adds `import os` and replaces one line, the client
construction. With `ATM_JUDGE_BASE_URL` unset it is byte-identical to the original, so nothing
about the prompt, the temperature, the retry policy or any scoring code changes.

### The route was measured rather than assumed

60 `open_end` questions from the completed full run, whose OpenAI verdicts were already on disk,
re-judged through OpenRouter on identical (question, gold, prediction) triples. Selection was
answer-blind, SHA256 order of the question id.

| Quantity | Value |
| --- | ---: |
| Paired verdicts | 59 of 60, one judge call produced no verdict |
| Agreement on the binary verdict | **56/59 = 0.9492** |
| Mean score, OpenAI route | 0.7119 |
| Mean score, OpenRouter route | 0.6949 |
| Verdicts flipped to false / to true | 2 / 1 |

⚠️ **I cannot attribute the disagreeing 5% to the route.** The control that would separate a route
difference from the judge's own run-to-run variance is a second OpenAI pass over the same 59 rows,
and that is exactly what no longer has credit. So 0.9492 is an upper bound on route fidelity and a
lower bound on judge determinism, and this record does not claim to know which.

### What that changes about the comparison, and it is not cosmetic

* **Arm against arm stays valid.** All four arms are judged on one route, over the same 300
  questions, so a paired difference between them carries no route term at all.
* **Arm against the 68.92 figure does NOT stay valid.** That number was produced through OpenAI,
  and this route scored 1.7 points lower on the paired sample, which at `open_end`'s 50.7% share is
  roughly 0.86 QS of bias in the direction that would make every arm look worse.

Rather than carry that bias as a footnote, the OLD answer file will be re-judged through the SAME
route, restricted to the same 300 questions, and that becomes the baseline every arm is compared
against. It costs 120 judge calls and removes the confound instead of describing it. The 68.92
figure is then used only as the provenance of the answer file, never as a term in a difference.

## Result

The ceiling prediction is still open. At 142 answers carrying token counts, 1 exceeded 8,192, with
a maximum of 12,919 observed. The registered threshold was fewer than 5 of the remaining 900.

---

## Result: the ceiling falsifier fired (2026-08-20)

`C-allocated-disposition` died at 299 of 300 with `CompletionTruncated` at **16,384** tokens, and
because the driver runs the arms under `set -e` it took `D-allocated-selection` with it before that
arm had generated a single answer.

That is the registered falsifier, verbatim from above:

> Another `CompletionTruncated` at 16,384. Then the tail is heavier than two data points suggested
> and the fix is not a bigger number, it is a reasoning effort or a model that terminates.

### The distribution, which says why a bigger number is the wrong move

Across 333 completions with recorded token counts:

| statistic | completion tokens |
| --- | ---: |
| mean | 572 |
| p50 | 349 |
| p90 | 1,142 |
| p99 | 3,534 |
| max recorded | 12,919 |
| above 8,192 | 1 of 333 |
| above 16,384 | 0 recorded, plus the one that raised before it could be recorded |

The median is 349 and something wants more than 16,384. A gap of nearly fifty times between the
middle and the tail is a reasoning loop that does not terminate, not a long answer, and there is no
ceiling that catches that. Raising it would be the fourth increase in this project and the record
above forbade exactly that.

### What was done instead

`CompletionTruncated` no longer aborts the run. The question is recorded in `diagnostics.jsonl`
with `truncated: true` and its reason, it is deliberately absent from `answers.jsonl`, and the run
continues. The manifest carries `truncated_questions`, so a reader who counts rows sees the
shortfall rather than a silent 300.

No truncated text ever enters `answers.jsonl`, which is the property `benchmarks/llm.py` documents
for the other harness: a half-written answer that gets scored is a measurement error introduced by
our own configuration. And because every comparison in this study runs on the questions common to
all arms, a question one arm cannot finish drops out of the pairing symmetrically instead of
counting as a zero against it.

### Prediction 2, which decides whether arm A is comparable

Registered: fewer than 5 of the remaining 900 answers exceed 8,192 completion tokens. Measured so
far: **1 of 333**, maximum 12,919. The prediction holds on the evidence available, so arm A's lower
ceiling did not bind and A remains comparable with the rest. This is not final until C and D finish.

### What is still owed

The truncating question is a property of the configuration, not of the arm, and the honest handling
is to report which arms could not finish it rather than to quietly re-run it until it succeeds. It
is one question of 300.
