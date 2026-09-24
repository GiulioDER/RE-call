# C9 Code4 plus Context4: what the pairing does, what it costs, and whether an LLM can make it pay

**Date:** 2026-09-24. **Baseline:** `C9_routed_specialists_grounded_graph_atomic`, served at
`337f2537` (retrieval identical to `936b7bda`). **Pre-registration:**
`docs/preregistrations/2026-09-24-c9-code4-context4-routing-headroom.md`, committed at `6872c12c`
before anything below was measured.

## 1. The short answer

C9 already is the Code4 plus Context4 configuration. Every Add is embedded into both spaces, and a
keyword router sends each Search to exactly one of them. Measured inside C9, Context4 is the weaker
space on every track we can test:

- **Coding:** the router sends all 34 local CAMBench coding prompts to Code4, so Context4 is never
  queried there. Separately, Code4 already reaches 34/34 source recall at 10, so no route or fusion
  policy could add recall.
- **Textual (LoCoMo, 1,535 questions through the real C9 app):** all Code4 scores 93.81 turn
  hit@10, the served router 93.09, all Context4 91.79.

The ceiling on making the two spaces cooperate is small. A perfect per-question choice between
them (an oracle that knows which one found the evidence) reaches 95.57, **+1.76 points
[+1.11, +2.48]** over all Code4, from 27 questions out of 1,535. That ceiling applies to every
pre-retrieval router, LLM or not, and to any post-retrieval selector that chooses among the two
top-10 lists. A query-only gpt-4o-mini router, measured today, captured none of it: it scored 92.90, **-0.91
points [-1.69, -0.13]** against all Code4, rescuing 12 of the 27 reachable questions while losing 26.

**Recommendation:** keep the official run on C9 exactly as decided. Do not add an LLM router. If
there is budget for one more Textual experiment after the official runs, the better candidates
are (a) Code4 everywhere, pre-registered on its own, and (b) LLM evidence selection over Code4's own
top 20, which has more headroom (+2.93) than anything Context4 offers (+1.05 at the same depth).

## 2. How C9 combines the two embedders today

Read from `origin/master` (`recall_aml/specialists.py`, `service.py`, `storage.py`):

| Stage | Code4 (primary) | Context4 (specialist) |
| --- | --- | --- |
| Add: raw windows | embedded, BM25 indexed | embedded as one contextual document per request |
| Add: compiled anchor records (gpt-4o-mini) | go to a separate graph store, never ranked directly | **written into the same index as the raw windows** |
| Add: atomic views | embedded | embedded again |
| Search: when used | default, any code keyword, any ambiguous text | only when the query has a conversational keyword and no code keyword |
| Search: stable window order | on | off |
| Search: graph sidecar | Code4 query embedding over the graph store | also Code4: a second query embedding on this route |
| Fusion across the two spaces | none | none |

So despite the name `routed-rank-fusion-v1`, C9 never fuses Code4 and Context4 results. It picks
one space per query with a regular expression and serves that space alone.

## 3. Measured evidence

### 3.1 Coding

- Code4 replaced Code3 with +8 net paired Task Solve wins (67/100 against 59/100), preregistration
  091.
- Context4 as a direct replacement: source recall at 10 was 30/34 against 34/34, MRR 0.600 against
  0.846, zero rank wins, 16 regressions (preregistration 095).
- Two protected fusion policies, each giving Context4 one or two suffix slots, failed their frozen
  evidence gates before any Task Solve spend (preregistrations 096 and 097).
- Today: `route_query` on the 34 CAMBench prompts returned `code` 34 times.

On Coding, Context4 in C9 has no measured effect on what is served and a certain cost at Add.

### 3.2 Textual, LoCoMo

From the 2026-09-23 route comparison (PR 733), re-analysed today:

| Arm | turn hit@5 | turn hit@10 | turn hit@20 | session hit@10 |
| --- | ---: | ---: | ---: | ---: |
| all Code4 | 88.53 | **93.81** | 96.74 | 95.70 |
| served router | 87.49 | 93.09 | 96.22 | 95.24 |
| gpt-4o-mini router | 87.10 | 92.90 | n/a | 94.92 |
| all Context4 | 85.21 | 91.79 | 95.83 | 94.07 |
| oracle (best of the two, per question) | 90.75 | 95.57 | 97.79 | 97.07 |

Paired against all Code4, turn hit@10, 10,000 bootstrap resamples:

| Comparison | delta, points | 95% interval | wins | losses |
| --- | ---: | --- | ---: | ---: |
| all Context4 | -2.02 | -3.26 to -0.85 | 27 | 58 |
| served router | -0.72 | -1.24 to -0.20 | 3 | 14 |
| oracle | +1.76 | +1.11 to +2.48 | 27 | 0 |
| gpt-4o-mini router (690 of 1,535 to Context4) | -0.91 | -1.69 to -0.13 | 12 | 26 |

Apparatus check: replaying the served router's own choices through the two forced arms reproduced
its turn hit@10 on all 1,507 rows routed to code or context, with no mismatch. So scoring a new route
choice by looking up the forced arm is valid for this data.

Where the 27 Context4-only wins sit (exploratory, not pre-registered): **24 of them are questions
the keyword router sends to Code4**, and among the 420 questions it does send to Context4, Code4
would have won 14 and lost 3. The keyword that triggers the Context4 route is not a signal of where
Context4 helps.

## 4. Pros and cons of the pairing as served

**Pros**

1. **Safe by construction for Coding.** Ambiguous text defaults to Code4, and every local coding
   prompt goes there. The measured Coding baseline is untouched.
2. **Tenant isolation is correct.** The two spaces never share a vector table, never compare raw
   cosines across models, and each carries its own profile, which is the architecture the
   2026-09-22 multi-tenant direction asks for.
3. **The oracle shows real complementarity.** 27 questions (1.8%) are found at 10 only by
   Context4, so the two spaces are not redundant.
4. **Deterministic and observable.** Every Search reports its route and embedding profile, which is
   why a replay like today's is possible at all.

**Cons**

1. **The Context4 route costs points on Textual.** -0.72 [-1.24, -0.20] turn hit@10 against Code4
   everywhere, entirely from the 420 questions it reroutes.
2. **Two full embedding passes on every Add, inside the per-user lock.** Every raw window, compiled
   record and atomic view is embedded twice. On Coding the second pass is never read. This lengthens
   the Add lock hold, which is the resource the 2026-09-23 concurrency ramp found to be the
   ceiling (Add about 0.28/s, breaking at platform concurrency 64).
3. **The two routes do not index the same records.** With the graph on, compiled records go into
   the Context4 index but not the Code4 index, and the Context4 route skips the stable window order.
   So "Context4 is worse" is a statement about C9's Context4 route, not about the embedder. The
   exploratory check does not support compiled records as the loss mechanism (Context4 losses carry
   1.81 non-raw items in the top 10 against 2.18 on average), but the route comparison is still not a
   clean embedder comparison.
4. **The router is English keyword matching.** Words like "test", "fix", "build" or "config", a dot
   followed by letters, or "photo" and "picture" decide the space. A Textual question about a photo
   goes to the multimodal route, which serves from Code4 in a text-only run.
5. **The standalone Context4 result does not transfer.** Its +6.25 hit@5 over Voyage 4 (2026-09-13,
   embedding only) disappears inside C9's BM25 fusion, graph and atomic stages.

## 5. Can an LLM make the two embedders work together?

Every option below either chooses between the two spaces or reorders what they return, so each is
bounded by what is measured above. The table puts the bound next to each.

| Option | Where the LLM acts | Ceiling on LoCoMo turn hit@10 over all Code4 | Status |
| --- | --- | --- | --- |
| A. Query-only router | before retrieval, picks one space | +1.76 (oracle) | **measured today**: -0.91 [-1.69, -0.13], and -0.20 [-1.11, +0.72] against the keyword router |
| B. Sufficiency gate with fallback | reads Code4's top 10; if the evidence looks absent, also queries Context4 and appends | +1.76 at most, and only if the judge is right on the 6.2% of questions Code4 misses | not run; see below |
| C. LLM selection over both top-10 lists | after retrieval, picks 10 of about 20 | +1.76 | not licensed: pre-registered bar was +2.0 |
| D. LLM selection over Code4's own top 20 | after retrieval, no Context4 at all | +2.93 | not run; largest measured headroom |
| E. LLM selection over both top-20 lists | after retrieval | +3.98, of which Context4 contributes +1.05 | not run |
| F. Query rewriting per space (declarative restatement, hypothetical turn) | before retrieval, changes what each space retrieves | not bounded by this data | not run |
| G. Decomposition of multi-hop questions, sub-queries routed per space | before retrieval | not bounded by this data | not run |

Why the options that route by content cannot work here: routing Textual questions by what they are
about (they are all conversational) would send them all to Context4, and that arm is the worst
measured (-2.02). The winner per question is not predictable from the question. It depends on
where each index happened to rank one turn, which is information the router does not have before
retrieval.

Why B, C and E are weak even before running them: the measured ceiling for bringing Context4 in is
+1.76 at 10 and +1.05 beyond Code4's own depth 20, while the only measured reranking step on this
product (Voyage rerank-3, 2026-09-19) lowered MRR, 0.323 to 0.259. An LLM selector has to beat that
prior while running on every Search.

What reasoning in the loop would cost on the official run:

- **Latency.** Search throughput is fixed at about 2.4/s server side (2026-09-23 ramp). One
  gpt-4o-mini call per Search adds its round trip to every request; B only on the fallback share,
  C to E on every request with 10 to 40 passages in the prompt.
- **Contract.** The organisers expect gpt-4o-mini at Add; anything at Search should also be
  gpt-4o-mini, and it is a change to the frozen C9 baseline that needs an explicit user decision.
- **Two Full runs per key and track.** None of these options has a result strong enough to spend
  one.

## 6. Recommendations

1. **Official runs: C9 unchanged.** This report changes nothing about them, and the pre-registered
   decision rule said it would not.
2. **Do not build an LLM router between Code4 and Context4.** Its ceiling is small and a query-only
   router does not reach it (section 5, option A).
3. **After the official runs, if Textual quality is the goal,** pre-register two arms in this order:
   - Code4 on every route for Textual: +0.72 [+0.20, +1.24] over the served router, measured
     2026-09-23 but not pre-registered as an arm. It also removes the second embedding pass on Add
     if Context4 is dropped entirely.
   - Option D, gpt-4o-mini selecting 10 of Code4's top 20, gated on answer accuracy rather than
     hit@10, because the AML score is the judge's verdict, not evidence presence.
4. **If Context4 is kept at all,** index raw windows only in it (the same records as Code4), so that
   any future comparison is an embedder comparison. Then re-measure before drawing conclusions
   about the model.
5. **For Coding, Context4 is pure Add cost.** A Coding-only deployment could drop it and shorten
   the Add lock hold. That is a C9 change, so it is the user's call, and it needs its own smoke test
   before any official run.

## 7. Limits

- LoCoMo is not AML Textual, which also includes LongMemEval, CLBench, PersonaMem, ScriptMem and
  BEAM. Hit@10 measures evidence presence, not whether the judge accepts the answer.
- The 34 CAMBench prompts are local; the platform's Coding queries may be phrased differently and
  could trip the conversational keywords.
- gpt-4o-mini at temperature 0 through OpenRouter is not guaranteed deterministic, so a rerun of
  the replay may move a few routes.

## Artifacts

- Replay script: `scripts/aml_c9_route_headroom.py`.
- Input: `docs/results/2026-09-23-aml-c9-locomo-route-comparison.json.gz` (PR 733), decompressed
  SHA256 `961d50ab…`. Dataset SHA256 `79fa87e9…` (pinned, matched).
- Output: `docs/results/2026-09-24-c9-route-headroom.json` and the cached LLM replies,
  `docs/results/2026-09-24-c9-route-headroom-llm-replies.jsonl.gz`.
