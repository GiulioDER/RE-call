# Pre-registration: how much can any Code4 or Context4 routing decision gain inside C9, and does a query-only gpt-4o-mini router capture it?

**Date:** 2026-09-24   **Status:** predicted, not yet measured

## The question

On the 1,535 scored LoCoMo questions of the 2026-09-23 C9 route comparison:

1. **Oracle headroom.** What is turn hit@10 when each question is served by whichever of the two
   forced arms (all Code4, all Context4) hit, and how many points is that above all Code4?
2. **LLM router.** Does a query-only `openai/gpt-4o-mini` router, choosing Context4 or Code4 per
   question, beat all Code4 on turn hit@10 by at least +0.5 point with a paired 95% bootstrap
   interval above zero?

Question 1 bounds every per-question choice between the two spaces: a pre-retrieval router, and
any post-retrieval selector that only chooses among the two top-10 lists. Question 2 is the
cheapest version of "let an LLM decide which embedder to use".

## What I predict

| quantity | prediction |
| --- | --- |
| oracle turn hit@10 | 95.0 to 96.5 |
| oracle minus all Code4, turn hit@10 | +1.2 to +2.7 points |
| questions where Context4 hits and Code4 misses at 10 | 18 to 42 |
| share of questions the LLM router sends to Context4 | 30% to 70% |
| LLM router minus all Code4, turn hit@10 | -1.0 to +0.3 points, interval includes zero or is negative |
| LLM router minus the served keyword router, turn hit@10 | -1.5 to +0.3 points |

Reasoning. The 2026-09-23 run measured all Code4 at 93.81 and all Context4 at 91.79, with
Context4 rescuing 24 questions against the router and losing 44. Rescues of that size set the
oracle a point or two above all Code4. The per-question winner looks like noise around a
systematic Context4 deficit, and a router that sees only the query has no signal about which
index happens to hold the evidence higher, so I expect it to capture none of the headroom and to
lose points in proportion to how often it picks Context4. My own record says I over-predict
effects by two to four times ([[i-over-predict-effect-magnitudes]]); the LLM router prediction is
already centred on zero, so that correction mostly narrows the oracle band.

## Decision rule, fixed now

- **Live post-retrieval LLM selection** (retrieve both spaces, let gpt-4o-mini choose the evidence)
  is worth a live arm only if the oracle is at least **+2.0 points** above all Code4 on turn hit@10.
- **A live LLM routing arm** is worth building only if the LLM router replay is at least **+0.5
  point** above all Code4 with a 95% interval above zero.
- Neither outcome changes the official C9 run configuration. That is the user's decision, recorded
  in `official-textual-coding-config-graph-on`.

## What would falsify this

- Oracle headroom below +1.2 or above +2.7 points.
- An LLM router gain of +0.5 point or more with an interval above zero.
- The apparatus check below failing on any row.

## How it will be measured

- **Input artifact:** `docs/results/2026-09-23-aml-c9-locomo-route-comparison.json.gz`, branch
  `claude/locomo-route` (PR 733, commit `ba4ce6be`), decompressed SHA256 prefix `961d50ab`.
  1,535 rows, each carrying turn hit@5, 10, 20, 100 for the `router`, `code` and `context` arms.
- **Questions:** LoCoMo `locomo10.json`, SHA256
  `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`, joined on
  `question_id = <sample_id>:<index into qa>`.
- **Oracle:** per row, `max(code, context)` for each metric.
- **LLM router replay:** per row, the forced arm matching the LLM's choice. Model
  `openai/gpt-4o-mini` through OpenRouter, temperature 0, one call per question, the question text
  only. The prompt, fixed now:

  > You route a memory search query to one of two retrieval indexes built over the same
  > conversation history. A: a contextual embedder that embeds every passage together with its
  > surrounding conversation. It is strongest when the answer sits in a turn whose meaning depends
  > on context: pronouns, follow ups, implicit references, events described across several turns.
  > B: a precise embedder fused with keyword search. It is strongest when the question names
  > specific people, objects, titles, places, numbers or distinctive words likely to appear
  > verbatim in the answer turn. Reply with exactly one letter, A or B.

  A maps to Context4, B to Code4. Any other reply maps to Code4 and is counted.
- **Statistic:** paired difference in points, 10,000 bootstrap resamples over questions, seed 0.
- **Apparatus check, known answer:** replaying the served keyword router's own choices through the
  forced arms must reproduce the router arm's turn hit@10 on every row where the served route is
  `code` or `context`. The replay is valid only if the forced arms are deterministic stand-ins for
  a route choice.
- **Script:** `scripts/aml_c9_route_headroom.py`, committed with the result.

## What I already know

- `c9-context4-route-worse-on-locomo` (2026-09-23): router 93.09, all Context4 91.79, all Code4
  93.81 turn hit@10. The router's 420 Context4 questions score 93.6 on Context4 and 96.2 on Code4.
- `aml-context4-coding-fusion-closed` (2026-09-20): on CAMBench Coding, Code4 already reaches
  34/34 source recall at 10, so no routing policy can add recall there; two protected Context4
  suffix policies failed their frozen gates.
- `voyage-context4-gold-retrieval-result` (2026-09-13): Context4 beat Voyage 4 by +6.25 hit@5 on
  LoCoMo embedding only. That did not survive inside C9.

## Confounds I can name now

- **C9's Context4 index is not the Code4 index re-embedded.** With the graph sidecar on,
  `recall_aml/service.py::HostedService._add_once` persists raw windows only to the primary Code4
  store and the compiled records to the graph store, but passes every chunk, raw and compiled, to
  `persist_specialist`. The Context4 route also skips the stable window order. So "Context4" here
  means Context4 over raw plus compiled records, and the oracle bounds routing between C9's two
  routes as served, not between the two embedders.
- Turn hit@10 counts evidence presence, not answer correctness.
- LoCoMo is not AML Textual, and gpt-4o-mini at temperature 0 through OpenRouter is not
  guaranteed deterministic, so a replay repeated later may differ by a few routes.

## Result (2026-09-24)

**Status:** measured

Apparatus check: 1,507 rows routed `code` or `context` by the served router, 0 mismatches. Dataset
SHA256 matched the pinned value. Output: `docs/results/2026-09-24-c9-route-headroom.json`; the 1,535
gpt-4o-mini replies are in `docs/results/2026-09-24-c9-route-headroom-llm-replies.jsonl`, 0 unparsed.

| quantity | predicted | measured | in band |
| --- | --- | --- | --- |
| oracle turn hit@10 | 95.0 to 96.5 | 95.57 | yes |
| oracle minus all Code4 | +1.2 to +2.7 | +1.76 [+1.11, +2.48] | yes |
| Context4 hits, Code4 misses, at 10 | 18 to 42 | 27 | yes |
| LLM router share to Context4 | 30% to 70% | 45% (690 of 1,535) | yes |
| LLM router minus all Code4 | -1.0 to +0.3, interval includes zero or negative | -0.91 [-1.69, -0.13] | yes |
| LLM router minus keyword router | -1.5 to +0.3 | -0.20 [-1.11, +0.72] | yes |

**Gap:** none outside a band. The LLM router rescued 12 of the 27 reachable questions and lost 26,
agreeing with the keyword router on 615 of 1,535 routes. It behaves like a partial all Context4 arm:
it loses in proportion to how often it picks Context4, which is what a router with no signal about
per-question retrieval quality should do.

**Decision, by the rule fixed above:** oracle headroom +1.76 is below +2.0, so no live post-retrieval
LLM selection arm between the two spaces. The LLM router is below +0.5 with an interval below zero,
so no live LLM routing arm. The official C9 configuration is unchanged.

**Deviation, recorded:** the first replay ran sequentially with a buffered cache, stalled visibly for
40 minutes with nothing flushed, and was killed before any reply was recorded or read. The script
was changed to 8 parallel workers with a cache flushed per line, and rerun from an empty cache. The
prompt, model, temperature and scoring were not changed. No result from the killed run exists.

**Exploratory, not pre-registered:** of the 27 Context4-only questions, 24 are routed to Code4 by the
keyword router. At depth 20, Code4 alone reaches 96.74 and the union of both 97.79, so Code4's own
ranks 11 to 20 add +2.93 over its top 10 while Context4 adds +1.05 beyond that. On the questions
where Context4 loses to Code4, its top 10 holds 1.81 non-raw items against 2.18 over all questions,
so displacement by compiled records does not explain its losses.

## Note (2026-09-24, after the result)

The replies file named above is committed gzipped, as
`docs/results/2026-09-24-c9-route-headroom-llm-replies.jsonl.gz`, so the pull request diff stays
within what the security review accepts. The content is unchanged.
