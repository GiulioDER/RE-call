# Pre-registration: re-derive the lane on EXECUTED ground truth

**Date:** 2026-08-27   **Status:** predicted, not yet measured
**Probe:** `scripts/agent_ab_checker_truth.py`, committed with this record.

## The question

Every "was this memo needed" label used today came from a model, and three models disagreed
(implied recalls 10, 9 and 6 of 14, κ 0.33 to 0.51). A human could not settle it, correctly:
"I can't label, I don't know the right answer." An executable oracle was in the archive the whole
time — `metadata.check`, the agent's own artifact run against the task's checker.

**For sessions that DEMONSTRABLY needed the memo, would draft-time search have surfaced it? And
for sessions that demonstrably did NOT, would it have fired anyway?**

## The population, which is the point of this record

Across all 48 on-arm `memory_only` sessions in 8 families, decided by execution rather than opinion:

| family | passed | failed | failure evidence |
|---|---:|---:|---|
| `ts-lf-rewrite` | 0 | 6 | 5 x "rewrote the file with 27 carriage returns against eol=lf"; 1 x "scripts/bump_version.py was not written" |
| `ts-sample-covers-tail` | 2 | 4 | "reached only 6/7 of 51 files over 120 seeds: head bias" |
| `ts-autouse-tmp-path` | 5 | 1 | "a test that enumerates its own tmp_path fails: the fixture w..." |
| `ts-bounded-runner` | 5 | 1 | "took 31.3s for a 3s bound: the grandchild held the pipe" |
| `ts-false-zero-search` | 6 | 0 | — |
| `ts-raise-on-missing` | 6 | 0 | — |
| `ts-separator-canary` | 6 | 0 | — |
| `ts-worktree-import` | 6 | 0 | — |
| **total** | **36** | **12** | |

- **`P_exec` = 11 objective POSITIVES**: the session fell into exactly the trap its memo describes,
  so the memo was needed. (12 failures minus the one indeterminate below.)
- **`N_exec` = 36 objective NEGATIVES**: the session passed, so the memo was not needed for the
  outcome and surfacing it would have been noise. **This is the first real negative population in
  this lane** — everything before was 18 git commands from one task or a retrieval-defined proxy.
- **1 EXCLUDED as indeterminate**: the `ts-lf-rewrite` session whose evidence is
  "scripts/bump_version.py was not written". It failed before reaching the trap, so whether the
  memo would have mattered is unknowable. Replayed and recorded, read from no endpoint.

⚠️ **The base rate is itself a finding and is stated before measuring: the memo was needed in 11
of 48 sessions, 23%.** Even where a hazard exists in the corpus and the task is built around it,
roughly three quarters of sessions do not need the note. Any per-write search proposal is mostly
firing when nothing is wrong, by construction.

## Design

- **Retrieval:** the shipped `HybridRetriever`, `lexical` leg, `candidate_k=200`, top-5, against
  `probe2_control`. The same configuration that reached 14/14 on retrieval.
- **Queries:** each session's recorded `Write`/`Edit`/`Bash` payloads, one query per payload,
  skipping the 3 over the server's 4,096-character limit.
- **Trigger:** the vocabulary test at `df<=2`, the threshold fitted on three families and already
  shown to transfer to four others at 0.933 coverage. Not refitted here.
- **No judge and no human label anywhere in this probe.** That is the whole point.

**Endpoints:**

1. **`recall_exec`** — of the 11 `P_exec` sessions, how many surface the governing memo at top-5
   for at least one draft.
2. **`fire_exec`** — of the 36 `N_exec` sessions, how many surface it anyway.
3. **`trigger_recall`** and **`trigger_fire`** — the same two, restricted to drafts the vocabulary
   trigger fires on, which is the only number that can tell whether a trigger separates sessions
   that needed the memo from sessions that did not.

## What I predict

No discount: this measures an existing system on better labels rather than introducing an
intervention. Rates with counts, never bare counts, per the defect that voided the trigger grid.

| # | Prediction | Falsified if |
|---|---|---|
| 1 | **`recall_exec` is 8 to 11 of 11.** The sessions that fell into a trap are exactly the ones whose drafts contain the hazardous operation, which is the mechanism draft search exploits | 7 or fewer |
| 2 | ⚠️ **`fire_exec` is 0.55 to 0.90 of 36.** A session that avoided the trap still wrote code in the same domain, so the lexical overlap that finds the memo is largely still there | outside the band |
| 3 | **The trigger does NOT separate: `trigger_fire` stays above 0.45**, i.e. it still fires on nearly half the sessions that did not need the memo. Four registered attempts have now failed to find a signal that distinguishes a hazard from a coincidence | at or below 0.45, which would be the first separation found in this lane |
| 4 | **`trigger_recall` is at least 7 of 11**, retaining most of endpoint 1 | 6 or fewer |
| 5 | **Cost: under 25 minutes and 0.00 USD.** ~450 retrievals, no model calls of any kind | either bound exceeded |

## Decision rule, as a cross product over RATES

Bands: `recall_exec` (`<0.60`, `0.60-0.79`, `>=0.80`) crossed with `trigger_fire` on `N_exec`
(`<=0.30`, `0.31-0.60`, `>0.60`). Nine cells, every one assigned:

| recall \ trigger_fire | `<=0.30` | `0.31-0.60` | `>0.60` |
|---|---|---|---|
| **`>=0.80`** | **BUILD**: finds what was needed, suppresses what was not; preregister the live A/B | **GATE**: finds it, gates it poorly; one registered iteration on the trigger | **KILL the per-write design.** It works and cannot be aimed; the only remaining shape is a search the agent asks for deliberately, not one that fires on every write |
| **`0.60-0.79`** | **BUILD WITH A CAVEAT**, recall stated as partial | **GATE** | **KILL the per-write design** |
| **`<0.60`** | **DEMOTE**: gates well but misses most of what actually mattered | **KILL** | **KILL the lane.** The retrieval does not find what was needed and cannot be aimed |

The three `>0.60` cells kill the per-write design regardless of recall, on the standard every
record in this lane has used: a mechanism that fires on most of the cases where nothing is wrong
is not deployable on every write, whatever its recall.

## What I already know

- Retrieval-only recall on the 14 previously-registered misses was 14/14, of which 5 sessions are
  now known to have PASSED their checker — so retrieval fired on 5 sessions that did not need it.
  That is a strong prior for prediction 2 and is disclosed as such.
- The vocabulary trigger covers 9/10 fitted and 14/15 held out, at 0.167 false-trigger against the
  18-draft git-flavoured negative set that this record replaces.

## Confounds I can name now

1. **"Passed" means the artifact avoided the trap, not that the agent knew about the hazard.** It
   may have avoided it by luck or by habit. Either way the memo was not needed FOR THE OUTCOME,
   which is the semantics this record uses and states rather than assumes.
2. **The checker is the benchmark's oracle, not ground truth about the world.** It can pass an
   artifact that is wrong in ways it does not test. It is nonetheless executed, recorded before
   any hypothesis here, and immune to prompt drift.
3. **11 positives is small**, and four of the eight families contribute none. Per-family numbers
   are counts, not rates, and a difference of one session is not a difference.
4. **`N_exec` is heavily weighted toward four families that never failed** (24 of 36 sessions), so
   the negative set is not balanced across hazard types.
5. **This record cannot re-open what the judged numbers closed**: the threshold and fusion
   frontiers were measured on retrieval, not on labels, and are unaffected by any of this.

<!-- frozen_above -->
