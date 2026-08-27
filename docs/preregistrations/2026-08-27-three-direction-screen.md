# Pre-registration: screening the three surviving directions on the 14 recorded misses

**Date:** 2026-08-27   **Status:** predicted, not yet measured
**Probe:** `scripts/agent_ab_screen_directions.py`, committed with this record.

## The question

The generation-side lane is closed (four registered measurements, no rescue). Three directions
survive. For the 14 recorded miss sessions, which of them can actually reach the governing memo?

This is a screen, not a build decision: it reads one already-certified corpus through the shipped
retriever, costs minutes and no money, and exists to kill at least one direction cheaply.

## What already exists, which narrows all three before a single number

A map of the serving pipeline (`recall/profiles.py`, `recall/retriever.py`, `recall/trust.py`,
`recall/semantic_graph.py`) was made before writing this, and it retires two things I would
otherwise have "discovered":

- ⛔ **Lexical retrieval is ALREADY fused into every served result.** `recall/trust.py` builds
  `HybridRetriever` with the defaults `use_dense=True, use_sparse=True, sparse_backend="lexical"`,
  and `recall/retriever.py` fuses the legs with unweighted Reciprocal Rank Fusion (k=60). So
  "add lexical/hybrid matching" is not a direction: it is on, it was on during the miss run, and
  the 14 misses happened with it on. What does NOT exist is **weighted or score-level fusion**, or
  any tuning of that RRF constant.
- ⛔ **Widening the reranker's pool has already been measured here and lost**: a 547-candidate pool
  lost 0.0513 R@100 against +0.0226 at 200, which is why `FUSED_RERANK_POOL_CAP = 100` exists. So
  direction A cannot be "give the reranker more candidates".
- The served pool is small and the reported number is misleading: each leg takes `candidate_k=20`
  in every profile, so the fused distinct pool is **at most 40**, and the `candidate_pool_size: 20`
  a client sees is the per-leg constant rather than the realised pool.
- A **semantic graph exists with `caused` and `depends_on` relation kinds**, which is exactly the
  shape "task X can trigger hazard Y" needs. Two constraints decide whether it is a direction:
  edges are **authored, never inferred**, and they are reachable only through
  `recall_reasoning_query(graph_expansion="one_hop")`, never through `recall_search`.

So the three directions, restated in the only forms still open:

- **A. A scorer that can follow task-to-hazard causal links.** Live only if the memo is IN the
  candidate pool for a reranker to promote, and a reranker cannot promote what retrieval never
  returned. This screen measures that ceiling and nothing else about A.
- **B. Lexical.** Already fused, so the only open question is diagnostic: is the memo lexically
  reachable AT ALL from a goal query? If yes but it loses in RRF, weighted fusion is a lever. If
  no, lexical is dead for this failure class in every weighting.
- **C. Interactive search behaviour** — searching at a different MOMENT, with different
  information.

## What makes direction C testable on recorded data

The `agent-ab-skill-001` archive records every tool call, including the CONTENT of each
`Write`/`Edit` payload and each `Bash` command. All 18 sessions of the three missed families have
such payloads (median ~3,900 characters). So the agent's own draft is recoverable and C can be
screened with **nothing fabricated**.

⚠️ **Disclosed, because it anchors predictions 3 and 4:** while checking whether drafts were
recorded at all, I read ONE payload, from a `ts-lf-rewrite` session. It contains
`version_file.write_text(content, encoding="utf-8")`. The governing memo
`python-write-text-crlf-churn` contains `path.write_text(text, encoding="utf-8", newline="\n")`.
I have looked at no other payload, no term-overlap statistic, and no retrieval result. That one
observation is why C is in this screen; the predictions below are anchored on it, not on a survey.

## Design

- **Corpus:** `probe2_control`, the certified control of the re-run, which answered identically to
  the first run's control on 78 of 78 queries at the same 1,019 chunks. Nothing is rebuilt or
  written.
- **Population:** the 14 scored miss sessions (`ts-lf-rewrite` 6, `ts-worktree-import` 5,
  `ts-sample-covers-tail` 3), `ts-raise-on-missing` excluded as in both prior records.
- **Retriever: the shipped `HybridRetriever`**, not a reimplementation, in three configurations —
  `dense` (`use_sparse=False`), `lexical` (`use_dense=False`), and `fused` (both, the production
  default) — at `candidate_k=200` so the coverage curve can be read past the served depth.
- **Queries:** each session's recorded GOAL queries (screens A and B) and, separately, each of its
  own recorded draft payloads, one query per payload (screen C). One per payload rather than one
  concatenated blob, because the behaviour this stands for is "search with the operation you are
  about to perform", not "paste your whole diff".
- **Endpoint: the governing memo in the top 5**, the depth a session actually receives, with
  coverage additionally reported at k in {5, 20, 40, 100, 200}. k=40 is called out because it is
  the real fused pool in production.

## What I predict

Per `[[i-over-predict-effect-magnitudes]]`: screens A and B measure an existing system, so no
discount applies. Screen C changes what INFORMATION reaches the retriever, which that memo classes
with decision changes, where I under-predict and should predict at the arithmetic rather than
discounting.

| # | Prediction | Falsified if |
|---|---|---|
| 1 | **Screen A: 4 to 8 of 14 memos are in the FUSED pool by k=40** (the real served pool), and 9 to 13 by k=200 | outside either band |
| 2 | **Screen B: 0 to 2 of 14 at top-5 lexical-only from goal queries.** A goal shares no literal with a hazard note; "bump the version" against a CRLF memo | 3 or more |
| 3 | **Screen C: 6 to 10 of 14 at top-5 for the better retriever**, and **lexical beats dense** on drafts, because a draft shares literal identifiers (`write_text`, `encoding=`) with a memo's fix line while sharing prose style with nothing | outside the band, or dense strictly beats lexical |
| 4 | **Screen C, `ts-lf-rewrite`: at least 4 of 6.** The family the disclosed payload belongs to, and the one that has resisted every previous direction 0 for 6, four times running | 3 or fewer |
| 5 | **Cost: under 20 minutes, 0.00 USD.** No builds, no generation, no network | either bound exceeded |

## Decision rule, per direction, before any number

- **A lives** if >= 8 of 14 are in the fused pool by k=40: a causal scorer then has material, and
  the next registered step is whether an LLM asked "which of these could this task trigger?" can
  pick the memo out of that pool. **A dies** if fewer than 4 of 14 are in by k=200: retrieval never
  finds them and no scorer can. Between those, A is alive only as "widen the pool AND rerank",
  which must be preregistered against the existing negative pool result.
- **B lives** only if lexical-only rescues >= 3 of 14 that fused missed — that is, if the memo is
  lexically reachable from a goal but loses in RRF, which would make **weighted fusion** the lever.
  Otherwise lexical is not a direction on its own.
- **C lives** if it rescues >= 6 of 14, which would be the first thing in this entire line of work
  to move the number. **C becomes the priority even at 4 or 5** if it beats A's k=40 coverage,
  because it needs no new model, no new index and no retrieval surgery: it changes WHEN the agent
  searches and WITH WHAT, and the retrieval it needs already ships.
- **All three below their bars**: these misses are unreachable by any retrieval over this corpus,
  and the recorded-miss benchmark is exhausted as an instrument. That is a publishable finding and
  the end of this lane, not a prompt to invent a fourth direction.

## Confounds I can name now

1. **I have read one payload.** Disclosed above; predictions 3 and 4 rest on it.
2. **A draft query is not free.** If C wins it implies searching once per write, which costs tokens
   and latency the way the skill instruction did (+107k median input tokens). This screen measures
   reachability only; any build proposal must price the behaviour rather than inherit this result.
3. **Post-hoc drafts.** The payloads come from sessions where the memo never arrived. An agent that
   HAD been warned would write different code — which is the correct direction for this question,
   since it is the unwarned draft that must trigger the search.
4. **`ts-sample-covers-tail` has 3 scored misses.** Per-family numbers there are anecdote; the
   record will say so rather than quoting a fraction of three as a rate.
5. **Isolating a leg is not the served fusion.** Screens A and B read one leg to decide a
   direction; neither predicts what the served RRF returns with that leg weighted differently.
6. **`candidate_k=200` is far past production's 20.** The coverage curve is a ceiling, deliberately
   generous to A and B, so a null there is strong and a positive there is not yet deployable.

<!-- frozen_above -->

## Result (2026-08-27)

**Status: measured, positive control passed on all three legs, and direction C wins outright.**

Coverage of the 14 scored misses, governing memo at rank <= k:

| screen | k<=5 | k<=20 | k<=40 | k<=100 | k<=200 |
|---|---|---|---|---|---|
| goal_dense | 0/14 | 6/14 | 6/14 | 8/14 | 9/14 |
| goal_lexical | 0/14 | 1/14 | 3/14 | 11/14 | 12/14 |
| goal_fused (what production serves) | 1/14 | 6/14 | 6/14 | 8/14 | **14/14** |
| draft_dense | 7/14 | 9/14 | 12/14 | 14/14 | 14/14 |
| **draft_lexical** | **14/14** | 14/14 | 14/14 | 14/14 | 14/14 |
| draft_fused | 11/14 | 14/14 | 14/14 | 14/14 | 14/14 |

Artifact: `benchmarks/artifacts/agent_ab/direction-screen.json`.

🔑 **The single most important line: querying with the agent's OWN DRAFT retrieves the governing
memo at top-5 for 14 of 14 misses, lexically, most of them at rank 1.** `ts-lf-rewrite`, the family
that resisted every previous direction 0 for 6 across four registered attempts, is **6 of 6**, at
ranks 1, 1, 1, 1, 1, 3.

🔑 **And the diagnosis this settles: retrieval was never broken. The QUERY was.** The same fused
retriever that returns the memo at rank 1 from a draft ranks it **127 to 142** for the same
session's goal query. `goal_fused` reaches 14/14 by k=200, so every governing memo was always
findable in this corpus, at a depth no consumer ever reads. Four generation-side attempts tried to
move the document toward the query. The document was fine; the query was asked at the wrong moment.

**Predictions against measurements:**

| # | predicted | measured | verdict |
|---|---|---|---|
| 1 | A: 4 to 8 in fused pool by k=40; 9 to 13 by k=200 | 6/14 by k=40; **14/14** by k=200 | first half confirmed, **second half falsified (above band)** |
| 2 | B: 0 to 2 at top-5 lexical-only from goal | **0/14** | confirmed |
| 3 | C: 6 to 10 at top-5, lexical beating dense | **14/14**, lexical 14 vs dense 7 | **falsified on magnitude (above band)**, direction confirmed |
| 4 | C, `ts-lf-rewrite` >= 4 of 6 | **6/6** | confirmed |
| 5 | under 20 minutes, 0.00 USD | ~10 minutes, 0.00 USD | confirmed |

Prediction 3 is the fourth consecutive under-prediction of an intervention that changes what
INFORMATION reaches the system, exactly as `[[i-over-predict-effect-magnitudes]]` says happens for
decision-side changes. I applied that memo's rule ("predict at the arithmetic, do not discount")
and was still low by 40%. The rule needs strengthening again, not the discount.

**Decisions, per the registered partition:**

- **B is DEAD.** Lexical-only from a goal query rescued 0 of 14 at top-5, below the bar of 3. It is
  already fused into production and it does not reach these memos from goal vocabulary in any
  weighting, so weighted fusion is not a lever for this failure class.
- **A is NOT dead but is demoted.** 6 of 14 in the served fused pool at k=40 misses the bar of 8, so
  a causal scorer over today's pool has material for fewer than half the cases. It is alive only in
  the form "widen the pool AND rerank", which must be preregistered against the existing negative
  pool result (547 candidates lost 0.0513 R@100).
- ✅ **C WINS, and by the registered rule it is the priority**: 14 of 14 against A's 6, and it needs
  no new model, no new index and no retrieval surgery. The retrieval it requires already ships.
  What changes is WHEN the agent searches and WITH WHAT: not "before I plan", but "before I save
  this file", using the text it is about to write.

⚠️ **The half this screen did NOT measure, and no build may proceed without it: PRECISION.** Every
number above is recall of a known governing memo. A draft query is ~3,900 characters OR-ing many
identifiers, and nothing here establishes what ELSE comes back, or what a draft returns when there
is no relevant memo at all. A search that returns the right memo at rank 1 and four irrelevant ones
every time, on every write, is a different proposition from this table. The next registered step is
precision and false-trigger rate on drafts whose hazard is not in the corpus, plus the token and
latency cost of searching per write, which the skill instruction measured at +107k median input
tokens for a much cheaper behaviour.

⚠️ **Also unmeasured: whether an agent would act on it.** This screen shows the memo can be
retrieved. `[[agent-ab-task-success-result-2026-08-22]]` measured that reaching the agent and
changing the outcome are different things.

## The apparatus, and why this record spends a paragraph on it

This screen produced a **flawless, entirely fabricated `0/14` across all six columns on its first
run**, and would have done so on the second and third for different reasons: a `PgVectorStore`
pointed at a generation-aware table (every query raised `UndefinedColumn`), a `search()` result
read as a list when it is a wrapper, and `Chunk.source_uri` read with a `getattr` default when the
field is `Chunk.source` (so every comparison was against an empty string). Any one of those alone
yields a perfect null that reads exactly like "none of the three directions works", and the first
version of the script caught retrieval exceptions and scored them as misses.

What caught all three was a **positive control**: query each leg with a memo's own content and
refuse to run if it does not come back at rank 1. It was added because
`[[a-null-is-the-cheapest-result-to-fabricate]]` was written one hour earlier, from an audit of the
previous probe in this same lane — and then the very next script reproduced the same failure three
times over. The guard is now in the script, and a retrieval error kills the run instead of
counting as a miss.

## 🔁 Correction appended 2026-08-27: the 14/14 above counts RETRIEVAL, not APPLICABILITY

**Nothing above is edited.** This record's headline — draft queries retrieve the governing memo at
top-5 for 14 of 14 misses — is correct as a statement about retrieval and **overstates the finding
it was read as**.

A judge control in `2026-08-27-enlarged-negative-set.md` exposed that **"the draft retrieves the
governing memo" is not "the governing memo applies to this draft"**. Three sessions' only
memo-retrieving draft is `ls -la benchmarks/` — eighteen characters, retrieving the memo on a
shared directory name.

Verified by judging all 46 memo-retrieving drafts
(`benchmarks/artifacts/agent_ab/actionable-recall.json`):

| measure | value |
|---|---|
| retrieval-only recall, **as published above** | 14/14 |
| **actionable recall** | **10/14** |
| retrieving drafts where the memo applies | 27/46 (0.587) |

**The comparison this record was written to make survives**: 10 of 14 against 1 of 14 for the
goal-vocabulary queries the same sessions actually issued, and the mechanism is unchanged wherever
a real operation retrieves a memo that really applies. **The number 14/14 does not survive**, and
neither does anything built on it. The lesson is one this project already had in another form
(`[[a-memo-can-be-in-the-corpus-and-unreachable]]`): retrieval is not the endpoint anyone cares
about, and a proxy that is obviously right can be wrong for 4 of 14 cases.


## 🔁 Correction appended 2026-08-27: the query length in this record is wrong, twice

**Nothing above is edited.** Two numbers in this document are wrong and both are the same mistake:
line 52 says the recorded payloads have a "median ~3,900 characters" and the result section says
"a draft query is ~3,900 characters OR-ing many identifiers".

**3,900 is the median TOTAL across all of a session's payloads, not the length of a query.** This
screen issues **one query per payload**, and measured over the 501 recorded payloads the
per-payload length is **median 60 characters, mean 354, maximum 5,125**. Re-measure:

```bash
python -c "import json,pathlib,statistics; ..."   # see scripts/agent_ab_draft_precision.py
```

**The correction strengthens the result rather than weakening it**, which is why it matters
enough to append. The 14/14 was achieved from short, realistic queries — a single edit hunk or a
single shell command — not from pasting a whole file into the search box. Every conclusion above
stands; the described mechanism was more deployable than the description.

It also revealed a real production constraint the record did not anticipate: the server
**refuses** any query over `MAX_QUERY_CHARS = 4096` rather than truncating it, on the stated
ground that searching a prefix answers a question the caller did not ask. That excludes **3 of
501** payloads and leaves no session without a usable one, so it bounds the direction without
threatening it. The precision record measures against that limit explicitly.
