# Result appendix: re-derivation on executed ground truth

**Appended 2026-08-27** to `2026-08-27-checker-ground-truth.md`, kept in a separate file because
that record's own tail is frozen and this is long enough to deserve its own anchor.

**Status: measured. All three gates passed** (population 11/36/1 exactly as registered, retrieval
positive control, no memo-less scored session). Artifact:
`benchmarks/artifacts/agent_ab/checker-truth.json`.

⚠️ **The corpus was rebuilt before this run.** The container holding `probe2_control` was removed
between sessions, taking all four probe databases with it — the hazard `agent_ab_build_corpus.py`
documents as "a removed container is a destroyed corpus; the build is the backup". Rebuilt from the
committed sources and verified identical to the original: **1,019 chunks, 194 sources**.

## The numbers

| endpoint | count | rate |
|---|---:|---:|
| **`recall_exec`** — sessions that NEEDED the memo, where draft search surfaces it | **11/11** | **1.000** |
| **`fire_exec`** — sessions that did NOT need it, where it surfaces anyway | **29/36** | **0.806** |
| `trigger_recall` — the same, with the vocabulary trigger in front | 10/11 | 0.909 |
| `trigger_fire` — negatives the trigger still fires on | 26/36 | 0.722 |

**Registered cell: `recall_exec >= 0.80` × `trigger_fire > 0.60`.**

> **KILL the per-write design.** It works and cannot be aimed; the only remaining shape is a search
> the agent asks for deliberately, not one that fires on every write.

## Predictions: 5 of 5 confirmed

| # | predicted | measured | verdict |
|---|---|---|---|
| 1 | `recall_exec` 8 to 11 of 11 | **11/11** | confirmed, at the top of the band |
| 2 | `fire_exec` 0.55 to 0.90 | **0.806** | confirmed |
| 3 | `trigger_fire` stays above 0.45 | **0.722** | confirmed |
| 4 | `trigger_recall` at least 7 of 11 | **10/11** | confirmed |
| 5 | under 25 minutes, 0.00 USD | ~20 minutes, no model calls | confirmed |

The first clean sweep in this lane, and the class matters: every one of these measures an existing
system on better labels rather than predicting an intervention's effect.
`[[i-over-predict-effect-magnitudes]]` says that is the class I estimate well, and it held.

## What this settles

🔑 **Draft-time search finds the memo in EVERY session that actually needed it. 11 of 11, on
executed ground truth, with no judge anywhere in the measurement.** That is the strongest recall
result in this lane and it retires the 6-to-10 range the judges left behind: on the question that
matters — did the agent fall into the trap, and would the memo have reached it — the answer is all
of them.

⛔ **And it fires on 29 of the 36 sessions that demonstrably did not need it.** The mechanism works
perfectly and cannot be aimed. The vocabulary trigger, the best gate found across four registered
attempts, moves false fires from 0.806 to 0.722 while costing one of the eleven — a 10% reduction
in noise for a 9% reduction in the thing that works.

**The base rate registered before measuring explains why this was always going to happen**: the
memo was needed in 11 of 48 sessions, 23%. A mechanism that fires on every write, in a world where
three quarters of sessions need nothing, is mostly firing when nothing is wrong. No gate tested
here changes that arithmetic.

**The seven quiet negatives are the exception that proves it.** Six of the seven are
`ts-raise-on-missing`, the family excluded from every prior record because its governing memo is
the one reconstruction-approximate source in the corpus — so its memo is hard to retrieve for a
reason that has nothing to do with hazard detection. Strip those and the mechanism is silent on
**1 of 30** sessions that did not need it.

## What survives, and what this closes

**Closed: the per-write design.** Search-before-every-write is not deployable, and the reason is
not retrieval quality, ranking, thresholds, calibration or the gate. It is the base rate.

**Survives: the finding that draft vocabulary is the right query.** When a search happens, using
the code about to be saved rather than the task description is worth 11/11 against 1/14 for goal
vocabulary. That result stands and is now on executed ground truth.

**The remaining shape, unmeasured and unregistered:** a search the agent asks for *deliberately* —
at a moment it chooses, with the draft as the query. That keeps the query formulation that works
and removes the trigger problem entirely by not needing a trigger. It is a different proposal from
anything measured today and needs its own registration, and its endpoint would have to be task
success rather than retrieval, because "the agent chose to search" is not something a replay of
recorded sessions can measure.

## What this cannot claim

1. **The checker is the benchmark's oracle**, not ground truth about the world; it can pass an
   artifact wrong in ways it does not test. It is nonetheless executed and recorded before any
   hypothesis here.
2. **11 positives, and four of eight families contribute none.** Per-family figures are counts.
3. **`N_exec` is weighted toward families that never failed** (24 of 36), so it is not balanced
   across hazard types.
4. **"Passed" means the artifact avoided the trap, not that the agent knew the hazard.** The memo
   was not needed for the outcome, which is the semantics used throughout and stated rather than
   assumed.
