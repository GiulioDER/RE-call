# Stage 1 result: the instruction did not change the act

**Appended 2026-08-27** to `2026-08-27-deliberate-draft-search.md`, in its own file because that
record's tail is frozen.

**Status: measured. GATE FAILED at 0.067 against a registered 0.40. Stage 2 is NOT run.**
Artifacts: `benchmarks/artifacts/agent_ab/draft-query-v3-stage1/`,
`benchmarks/artifacts/agent_ab/draft-adoption.json`.

## Deviation from the registered design, stated first

The record fixes stage 1 as "16 sessions (8 families x 2 repeats), treatment arm only". Two things
differed and neither was a choice made after seeing results:

1. **The harness runs PAIRED arms**, so the 16 treatment sessions came with 16 control sessions.
   The endpoint is unaffected — only the treatment arm is scored — but the run cost double.
2. **The harness runs all 10 families**, including the two `ctl-*` controls, so 20 treatment
   sessions were produced. The registered population is the 8 `ts-*` families; the `ctl-*` sessions
   are scored separately and **never pooled**.

## The endpoint

| population | sessions | searched | adopted | rate |
|---|---:|---:|---:|---:|
| **treatment, `ts-*` (registered)** | 16 | 15 | **1** | **0.067** |
| treatment, `ctl-*` (separate) | 4 | 4 | 0 | 0.000 |
| baseline `hazard-query-v2`, `ts-*` | 48 | 46 | 0 | 0.000 |

The single adoption is `ts-lf-rewrite#r2`, whose matching query was
`python scripts/bump_version.py 0.9.8` — a shell command it later ran, not a code draft.

| # | predicted | measured | verdict |
|---|---|---|---|
| 1 | stage-1 mechanism 0.55 to 0.85 | **0.067** | **falsified, far below** |
| 2 | does not reach 1.00 | 0.067 | confirmed, but vacuously |
| 3, 4 | stage-2 endpoints | not run, per the gate | untested |
| 5 | stage 1 under 40 minutes | ~37 minutes for 40 sessions | confirmed |

## Why it failed, which is more useful than the rate

**The agent searched ONCE, at the start, and then wrote 9 to 21 payloads.** Session query counts
are `q=1` almost uniformly, against `w=9` to `w=21` writes. The instruction says "immediately
BEFORE you save a file or run a command"; the agent front-loaded a single search exactly as the
previous instruction produced, and the per-write cadence never appeared.

**And it composed queries rather than pasting drafts.** What it actually issued:

| session | query |
|---|---|
| `ts-lf-rewrite#r1` | `version bump script recall/version.py 0.9.7` |
| `ts-bounded-runner#r1` | `bounded timeout subprocess kill process group` |
| `ts-separator-canary#r1` | `U+2028 line separator normalise test collapse space` |
| `ts-autouse-tmp-path#r2` | `RECALL_INDEX_ROOT uploads memo.md temporary fixture` |

Against what a pasted draft would have been:
`version_file.write_text(content, encoding="utf-8")`.

🔑 **The instruction moved the VOCABULARY partway and did not move the ACT at all.** These queries
carry filenames and identifiers (`recall/version.py`, `RECALL_INDEX_ROOT`, `corpus.jsonl`) that the
goal-vocabulary baseline lacked, so the agent clearly absorbed "be concrete about the artifact". It
did not absorb "paste the literal text", and it did not absorb "before each save". **Told to stop
composing a query, it composed a better query.**

## ⚠️ Unregistered observation, labelled as such

Governing memo reached: **9 of 16** under v3 (2 reps) against **31 of 48** under v2 (6 reps) —
0.563 against 0.646. Nominally lower, on a sixth of the sample, and **this is not a registered
endpoint of stage 1**. It is recorded because it is cheap and because a reader would otherwise
wonder; it is not evidence of anything at this n, and no decision rests on it.

## What this closes and what it does not

**Closed: this instruction.** `draft-query-v3.txt` does not produce draft-time search, and the gate
was registered precisely so a null here would cost 40 minutes instead of a 112-session A/B. It did.

**Not closed: the underlying finding.** Draft text remains the query formulation that reaches the
memo in 11 of 11 sessions that needed it. What is now measured is that **an instruction cannot make
an agent use it** — the agent's habit of one front-loaded, composed query survived an explicit,
specific instruction to do otherwise.

That relocates the problem precisely. The prior instruction moved search RATE from 0.532 to 1.000
and left vocabulary unchanged; this one moved vocabulary partway and left cadence unchanged.
**Both times the instruction changed what the agent SAID it would do and not what it DID.** A
third instruction variant is not licensed by this record, and the prior on prompt-level fixes in
this lane is now four attempts and no adoption.

**The shape that remains untested is mechanical rather than instructional**: a hook or tool wrapper
that performs the search at write time, with the payload as the query, without asking the agent to
remember. That is a different proposal, needs its own registration, and inherits the finding that
killed the per-write design — 29 of 36 sessions do not need the memo — so it would have to justify
the noise on a task-success endpoint rather than a retrieval one.
