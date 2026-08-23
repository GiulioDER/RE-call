# Pre-registration: can decompose-expansion rescue the queries that missed?

**Date:** 2026-08-23   **Status:** predicted, not yet measured
**Probe:** `scripts/agent_ab_probe_expansion.py`, committed with this record

## The question

For the 15 on-arm sessions of `agent-ab-skill-001` that searched and still missed their governing
memo, does expanding their RECORDED queries through one fixed, generic decomposition prompt
retrieve the memo at the same top-5, over the same corpus and stdio transport?

## Why this probe exists, and what it decides

The skill run just showed that instruction text fully solves search initiation and does not move
query formulation: 15 sessions asked in goal vocabulary and missed, six of them on `ts-lf-rewrite`
where every query was some variant of "version bump script". The candidate fix is server-side
query expansion in `recall_search`: a cheap model decomposes the goal query into the operations it
implies, and the retrieval unions the results. That is core-product surgery, so before building it
this probe asks the only question that matters: **can a model that knows nothing about the corpus
produce the operation vocabulary the agent could not?** The probe costs about two dozen expansion
calls and one stdio session; the feature costs a design, tests, and trust-semantics review.

A session is **rescued** when any expansion of any of its recorded queries retrieves its governing
memo at top-5. Union semantics mean an already-hit session cannot be un-hit, so only the 15 misses
are informative. The per-task denominators, extracted and verified before this record was written:
`ts-lf-rewrite` 6, `ts-worktree-import` 5, `ts-sample-covers-tail` 3, `ts-raise-on-missing` 1;
22 distinct queries.

The expansion prompt is committed inside the probe script and is deliberately generic: it teaches
the decomposition move and names no task, no memo, no file, no tool from the task set. If it needs
task vocabulary to work, the feature does not work.

## What I predict

Per [[i-over-predict-effect-magnitudes]]: mechanism rates are the one class I estimate well;
quarter-to-half of ceiling everywhere else. Ceiling is 15/15, since every governing memo is
qualified reachable at top-5 by the right query.

| # | Prediction | Falsified if |
|---|---|---|
| 1 | **5 to 9 of 15 sessions rescued** | 2 or fewer |
| 2 | **`ts-lf-rewrite`: at least 3 of 6 rescued.** "A python script edits a file" is mechanically derivable from "version bump script"; if decomposition works anywhere it works here | 0 of 6 |
| 3 | `ts-worktree-import` is the hardest rescue (its memo is about WHERE an import resolves, an operation even a decomposition prompt is unlikely to name): **at most 2 of 5** | 4 or more of 5, which would mean my model of WHY queries miss is wrong even when the probe succeeds |

Decision rule, stated in advance: prediction 1 met or exceeded, build the feature and preregister
its run. Prediction 1 falsified, do not build read-side expansion; the remaining lever is indexing
memos under operation vocabulary, which needs its own probe.

## How it will be measured

```bash
python -u scripts/agent_ab_probe_expansion.py \
  --archive ~/.claude/archive/agent-ab-skill-001 \
  --dsn postgresql://recall:recall@127.0.0.1:5407/agent_ab --tenant default
```

Expansion model `anthropic/claude-haiku-4.5` at temperature 0, three queries per input. Corpus,
tenant, generation and top-5 exactly as the run used them. The artifact records every expansion
and every retrieval verbatim, so the vocabulary claim is auditable after the fact.

## What I already know

- The 15 misses and their queries: `~/.claude/archive/agent-ab-skill-001/records.jsonl`, memo
  [[hazard-query-instruction-result-2026-08-23]].
- All four memos retrieve at top-5 for their probe query (`task-qualification.json`), so a miss is
  a vocabulary failure, not an indexing one.
- ⛔ The qualification rule from [[a-memo-can-be-in-the-corpus-and-unreachable]] applies to ME
  here: if the probe misses, the queries are NOT to be reworded until it hits, and the expansion
  prompt is NOT to be iterated against the result. One prompt, one run, one verdict. A second
  prompt is a second preregistration.

## Confounds I can name now

1. **The expansion model is the agent's own model.** A failure may mean haiku cannot decompose,
   not that decomposition cannot work. A success is clean; a failure licenses one follow-up with a
   stronger model, as a new registered probe, not a silent retry.
2. **Recorded queries, future feature.** The probe replays queries the feature would have seen
   this time; live queries will differ. Bounded external validity, accepted for a feasibility
   gate.
3. **Union dilution is not measured.** Expansion adds results the agent must wade through; whether
   that costs more than it buys is a feature-run question, not a probe question, and the decision
   rule above only commits to BUILDING, not to shipping.
