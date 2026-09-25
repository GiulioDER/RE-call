# Pre-registration: do per-session summaries ahead of C9's items help the reader on BEAM?

**Date:** 2026-09-25   **Status:** predicted, not yet measured

## The question

Three BEAM passes over the official C9 have established:

- the reader, not retrieval, loses summarization and event ordering (quote-verified coverage 0.76
  and 0.71, answer scores 0.33 and 0.23);
- giving it fewer items does not help;
- re-ordering by time helps only event ordering.

The remaining lever is the CONTENT it receives.

This test adds synthesis. gpt-4o-mini, the model C9 already runs at Add, writes one summary of at
most 250 words per BEAM session (90 sessions). Each summary carries no date header and is told
never to invent one, so the arm tests synthesis, not the dates lever another session owns. Arm
`summaries` puts that conversation's summaries, in session order, before the same 100 stored items
in returned order, and re-answers all 400 questions. Answer prompt, reader (Qwen3-14B) and judge
are unchanged. It is offline and sends nothing to C9.

Placing every session summary of the user first models a C9 that returns its session-summary
records at the top of every Search. For 3 to 5 sessions per conversation that is about 1,000 words.

## Baseline (run 1, `returned`)

Summarization 0.304, multi_session_reasoning 0.550, event_ordering 0.225, temporal_reasoning 0.290,
information_extraction 0.831, abstention 0.500, contradiction_resolution 0.063,
instruction_following 0.494, knowledge_update 0.525, preference_following 0.775. The 10-type mean
is 0.456.

## Predictions (`summaries` minus `returned`)

| type | predicted difference |
| --- | --- |
| summarization | +0.05 to +0.25 (point +0.12) |
| event_ordering | +0.02 to +0.15 (point +0.06) |
| multi_session_reasoning | +0.00 to +0.10 (point +0.04) |
| temporal_reasoning | -0.05 to +0.08 (point 0.00) |
| information_extraction | -0.05 to +0.02 (point -0.01) |
| 10-type mean | +0.00 to +0.06 (point +0.03) |

## Decision rule, fixed now

A session-summary record is worth building into C9, which would still need the user's decision,
only if all three hold:

1. summarization improves by 0.08 or more, with a paired bootstrap 95% CI whose lower bound is
   above 0;
2. the 10-type mean improves by 0.01 or more;
3. no type drops by more than 0.05.

Otherwise it is not built. A type-specific gain is reported but not acted on.

## What this cannot show

The summaries always reach the reader here. A real C9 record would have to be retrieved, and the
platform's reader, judge and context format may differ. This is one public split; the four types
of interest have 40 questions each.

Spend caps: summarize $0.60, answer $1.30, judge $0.40.

## Apparatus note, appended before any answer was generated (2026-09-25)

The first summarize pass (`7ce2bafc`, spend $0.41) was broken. The instruction sat above sessions of
tens of thousands of words in one user message, and gpt-4o-mini continued the dialogue instead of
summarizing it. A shape check found 71 of 90 outputs reply-like, 61 over 300 words and 1 empty.
None of them was used: they are archived as `out/summaries-broken-v1.jsonl` on VPS2, and the answer
phase never ran on them.

In v2, the instruction is a system message, the session is fenced as `<recorded_session>` data, and
the task is restated after it; an empty output is retried. A trial on conversations 0 to 2 (9
sessions) passed the same check: 0 reply-like, 0 over 300 words, 0 empty, 174 to 211 words each.
The predictions and the decision rule above are unchanged.

## Result

(appended after the run; nothing above is edited)

### Run 1, 2026-09-25 05:19 to 05:55 UTC, VPS2, offline, v2 summaries

The code is `benchmarks/beam/aml_c9_probe.py` at `3ea75840`. All 90 v2 summaries passed the shape
check: 143 to 212 words each, with 0 empty and 0 reply-like. Answer and judge files each hold 400
unique ids. Spend: summarize $0.41, answer $1.18, judge $0.08.

| type | returned | summaries | difference | 95% CI | n |
| --- | --- | --- | --- | --- | --- |
| summarization | 0.304 | 0.279 | **-0.025** | [-0.125, +0.077] | 40 |
| event_ordering | 0.224 | 0.328 | **+0.103** | [+0.001, +0.210] | 40 |
| multi_session_reasoning | 0.550 | 0.487 | -0.064 | [-0.202, +0.069] | 40 |
| temporal_reasoning | 0.289 | 0.184 | **-0.105** | [-0.184, -0.026] | 38 |
| information_extraction | 0.831 | 0.722 | **-0.109** | [-0.219, -0.014] | 40 |
| other five types | | | -0.013 to +0.037 | all CIs cross 0 | 40 each |
| **10-type mean** | **0.456** | **0.439** | **-0.016** | [-0.052, +0.020] (paired, 398) | |

**Decision rule: FAILED on all three conditions.** Summarization moved -0.025, not +0.08; the
10-type mean moved -0.016, not +0.01; and two types dropped by more than 0.05. **No session-summary
record is built.** Every prediction was falsified except event ordering (+0.103, inside +0.02 to
+0.15) and multi-session (inside only through its CI).

### Apparatus defect found in this run, which affects every BEAM pass

**Some answers are empty, and how many depends on which OpenRouter provider served them.**
OpenRouter routes `qwen/qwen3-14b` to several providers. On one replay, provider NextBit ignored
`reasoning: {enabled: false}` and spent 458 of the 512 answer tokens thinking, while Alibaba
answered in 10 tokens. When the thinking uses the whole budget, the answer comes back empty and
scores 0.

Empty answers per arm: returned 39 of 400, chronological 35 of 200, summaries 55 of 400, top10 64,
top20 63, top40 62. They concentrate in temporal and knowledge_update; summarization had 0 in
returned.

**Re-analysis on the questions where both answers are non-empty changes no decision:**

- top10, top20 and top40 against returned: +0.001, +0.005 and +0.004, all CIs crossing 0;
- chronological: -0.038;
- summaries: -0.023 [-0.061, +0.014], with summarization -0.025, event_ordering +0.120,
  multi-session -0.175 and information extraction -0.091.

The reader-miss finding also stands, because summarization had no empty answers in either arm.

**What the empties do change: absolute scores are understated,** most for temporal and
knowledge_update. Any future run must pin a provider that honours no-thinking, for example
`provider: {"order": ["alibaba"], "allow_fallbacks": false}`, or raise the answer budget.

### A finding outside the memory layer

Under AML's BEAM answer prompt ("Be direct and concise ... Only output the answer"), Qwen3-14B's
median answer is one word for contradiction_resolution (34 of 40 are a bare "Yes." or "No."),
knowledge_update and temporal_reasoning. Contradiction rubrics check about 4 points, so a one-word
answer cannot pass them: that type scores 0.06 whatever memory returns. In summarization, answers
under 60 words score 0.17 and answers of 60 or more score 0.38. **Part of the low scores is set by
the platform's answer format, which the memory service does not control.**
