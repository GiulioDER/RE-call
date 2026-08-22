# Handoff: the multi-product agent memory benchmark (approved plan, 2026-08-22)

This document records the approved design for `agent-memory-bench`, the public, multi-product
successor to this harness. The full plan lives in the session plan file; this is the durable copy
other sessions read. Read `NEXT-BENCHMARK-TASK-SUCCESS.md` first for the seven paid-for failures
and the two statistical rules; every one of them still applies.

## What it is

A preregistered, execution-graded benchmark of pluggable memory layers measured by **task success
on real coding tasks executed by Claude Code**. Eight arms: `bare`, `claude_md` (designated
baseline), `fs_grep` (the Letta baseline), `recall` (our shipped plugin), `mem0`, `supermemory`,
`zep` (Graphiti MCP + FalkorDB), `cognee`. Letta itself is excluded (it owns the agent loop) and
its critique is pre-empted in the write-up.

Nobody has published this: SWE-ContextBench has the tasks but no products, Supermemory's
MemoryBench has the products but only conversational QA, Bench'd is moving toward agentic but has
no coding. Past fights in this space went viral through methodological write-ups, and our own
product competes, so defensibility is the distribution strategy.

## User decisions, 2026-08-22

1. All five product arms plus the three baselines.
2. A **new neutral task set**. The existing 10 tasks were built from recall's own memory store
   surplus, which a competitor would rightly call rigged.
3. **Pilot first** (bare, claude_md, recall only), then size and preregister the full run from
   measured variance.
4. Separate public harness repo (Apache-2.0) plus a public vendor pre-review invitation with a
   two-week window; responses or documented silence recorded verbatim.

## Design decisions

1. **Official integrations, frozen and vendor-reviewed.** Each product enters through its own
   official Claude Code integration, config hash-pinned. Zep's review checklist explicitly covers
   the three errors they accused mem0 of: single-user model, timestamps appended to text,
   sequential searches.
2. **Additive everywhere.** Every memory arm is the same CLAUDE.md plus that product. Measured
   here already: additive beats substitutional.
3. **Hermeticity via per-session `CLAUDE_CONFIG_DIR`, not `--bare`.** mem0's and Supermemory's
   official integrations ARE lifecycle hooks; each arm gets a generated, isolated config dir
   containing exactly that vendor's integration, digested into the session record.
4. **One neutral experience feed, each product's own write path.** The corpus is verbatim
   recorded Claude Code transcripts (real runs, never hand-edited, structured timestamps). What
   each product stores is part of what is measured.
5. **Executable endpoints only.** No LLM judge in the primary endpoint.
6. **Predict low.** House priors: effects 2 to 4x over-predicted, costs 5x under-predicted.

## What this harness contributes

| here | there | change |
|---|---|---|
| `sandbox.py`, `checkers/_run.py`, `io.py`, salvage | `harness/` | copy verbatim |
| `claude_exec.py` | `harness/claude_exec.py` | add `CLAUDE_CONFIG_DIR` + hook-ledger fields |
| `stats.py` | `harness/stats.py` | add designated-baseline contrasts, Holm, negative transfer |
| `runner.py` | `harness/runner.py` | `run_paired` becomes `run_grid` over N arms per (task, seed) block |
| `gate.py` | `harness/gate.py` | per-arm `AdmissionSignal` (tool prefixes, required hooks, sandbox paths, forbidden prefixes = every other arm's) |
| `arms.py` | `harness/adapters/base.py` | replaced by a `MemoryAdapter` protocol; `ArmSpec` becomes the adapter's output |
| `tasksuccess.py`, `reference.py`, `traps.qualify` | `tasks/`, `scripts/qualify.py` | patterns copied; tasks become data (`task.json`), probes harvested from agent search calls, not authored |

recall enters via `pip install recall-rag==<pin>` like any vendor. `recall_server.py`'s env-block
fixes (APPDATA, SystemRoot, PYTHONPATH) carry into its adapter. Calibration tooling stays here.

## Task and corpus recipe (the long pole)

24 candidate tasks for 15 or more survivors (measured attrition ~35%). Governing facts are
**arbitrary-by-construction** project decisions that cannot be derived from repo, CLAUDE.md, or
world knowledge; that is the contamination control that matters. Per task, 3 to 6 recorded
precursor sessions where the gotcha genuinely manifests, plus 60 to 100 distractors (4:1 or
better). Checkers run against oracles the sandbox never contained; a do-nothing session scores 0;
the naive reference must fail silently; naive-fails and informed-passes asserted in CI. Floor
guard at construction (3 scripted informed runs, 2 of 3 must pass), ceiling guard at pilot
(drop `claude_md` >= 0.7 or `bare` >= 0.5). Qualification v2 probes with queries harvested from
actual agent search calls, which is the fix for the 0.532 search rate diagnosis.

Ingestion isolation: namespace `bench-{arm}-{seed}`; self-hosted stores snapshotted after ingest
and restored per session; SaaS arms get a fresh namespace per seed (disclosed limitation);
ingestion tokens metered through a LiteLLM proxy and published per arm.

## Statistics

Primary, preregistered: recall vs claude_md, per-task cluster bootstrap CI as the headline,
per-cell McNemar labelled as the consistency check that overstates confidence. Secondary: each
other arm vs claude_md under Holm-Bonferroni. Exploratory: head-to-heads and a league table with
CIs and no p-values. Published per arm: success rate, delta with CI, negative-transfer count,
search/hook-fire rate, tokens (median beside mean), wall time, ingestion cost, discarded cells.
Sizing by simulation from pilot variance, 80% power at a true +0.10.

## Phases

0. Harness repo bring-up: port, `run_grid`, `AdmissionSignal`, four baseline+recall adapters,
   smoke on Windows and Linux CI. **In progress.**
1. Corpus and tasks: 24 candidates, ~160 recorded sessions, leakage audit.
2. Pilot: 3 arms x 24 tasks x 3 seeds, preregistered. Decision gate: mechanism healthy but effect
   null means redesign tasks; mechanism unhealthy means fix the skill or placement, not the tasks.
   This is also the first A/B of the shipped plugin path (hooks plus skill).
3. Competitor adapters, one at a time, each with frozen config, versions.lock, metered ingest
   smoke, admission smoke in CI, drafted VENDOR_REVIEW.md.
4. Vendor pre-review window, then freeze config hashes into the full-run preregistration.
5. Full run: Docker on Linux, fsync per session, salvage tested beforehand.
6. Analysis in preregistered order; publish every arm including recall losses (committed in the
   preregistration, not decided after).

## Where the new repo lives

Local scaffold: `C:\Users\gde00\Documents\agent-memory-bench` (GitHub org and name await the
user's confirmation before anything is pushed).
