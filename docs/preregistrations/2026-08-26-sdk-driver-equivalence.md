# Pre-registration: does the Claude Agent SDK driver reproduce agent-ab-skill-001?

**Date:** 2026-08-26   **Status:** measured, see the Result section appended below

## The question

Holding everything but the driver fixed, does `--driver sdk` (`benchmarks/agent_ab/sdk_exec.py`,
over `claude-agent-sdk`) reproduce run `agent-ab-skill-001`'s mechanism and outcome within the
equivalence bands committed in `benchmarks/agent_ab/sdk-replication-bands.json`?

## Why this run exists

The SDK driver replaces `claude_exec.py`'s subprocess-and-parse driver ahead of the multi-product
successor (`benchmarks/agent_ab/NEXT-BENCHMARK-MULTI-PRODUCT.md`). A driver that shifts the
measurement must be caught before it carries a competitor comparison, so the driver's first
measured run is a replication of a known result, not a new question. A driver change conflated
with a treatment change would make both unreadable.

## Baseline (copied from the archived skill-001 analysis, so the comparison is reproducible from
## the repository even though the raw artifacts are archived)

Source: `~/.claude/archive/agent-ab-skill-001/analysis.json` (run 2026-08-23, CLI driver,
`anthropic/claude-haiku-4.5` via OpenRouter, instruction `hazard-query-v2.txt`).

| metric | baseline value |
|---|---|
| admitted pairs | 54 of 56 (2 discarded) |
| search rate (on arm, primary tasks) | 46/46 = 1.000 |
| governing-memo reach among searchers | 31/46 = 0.674 |
| per-task task-success mean delta | +0.208 (8 tasks, cluster CI [-0.021, +0.400]) |
| per-pair task-success delta mean | +0.196 (n=46, McNemar p=0.022) |
| control task success | on 1.000 / off 1.000 (n=8) |
| median input-token overhead (on minus off) | +106,946 |
| median wall-time overhead | +36.5 s (recorded only, see below) |

## What I predict

Every falsifier metric lands inside its committed band. Specifically, and more tightly than the
bands themselves, I predict:

- admitted pairs at least 52 of 56 (the stdio server config and Windows env fixes are carried
  unchanged, so gate discards should stay rare);
- search rate at least 0.95 (the instruction file is byte-identical and travels as the literal
  `--append-system-prompt-file` flag, so the initiation result should transfer);
- governing-memo reach within 0.10 of 0.674;
- per-task mean delta within 0.10 of +0.208, same sign;
- both control means at 1.000;
- median input-token overhead within 25,000 of +106,946;
- median wall-time overhead HIGHER than the baseline's +36.5 s by some seconds per session (SDK
  spawn and transport overhead), which is why wall time is recorded and not a falsifier.

Calibration note: my prediction record says I over-predict magnitudes by two to four times. For an
equivalence run the analogous failure is over-tight bands, so the committed bands are set by
decision relevance (would a difference this size change any conclusion skill-001 drew?) rather
than by my point predictions above.

## What would falsify this

Any `max_abs_diff` metric outside its committed band: search rate differing from 1.000 by more
than 0.15, reach differing from 0.674 by more than 0.15, per-task mean delta differing from +0.208
by more than 0.15, per-pair delta mean differing by more than 0.15, either control mean differing
by more than 0.125 (one flipped control pair), or median input-token overhead differing by more
than 55,000. Falling below 48 admitted pairs VOIDS the run as a wiring result rather than
falsifying equivalence; the run is then fixed and repeated under a new id.

Honesty note on power: 46 primary pairs cannot power a strict TOST at tight bands. These bands are
decision-relevance bands, not a significance ritual, and the record says so rather than implying a
formal equivalence test it cannot deliver.

## How it will be measured

Same configuration as skill-001 in every field but the driver: the 10 committed tasks at their
committed reps (56 pairs, 112 sessions), `benchmarks/agent_ab/task-qualification.json` as
committed, instruction file `benchmarks/agent_ab/instructions/hazard-query-v2.txt`, model
`anthropic/claude-haiku-4.5` through OpenRouter (`openrouter_env`), per-session stdio RE-call
server (`StdioRecallSpec`, production, calibrated; its `check()` output lands in
`environment.json` and must show `calibrated: true` with the generation and calibration ids),
additive arms (`claude_md` off, `claude_md_recall` on), instruction-first prompt ordering,
`pair_concurrency=1`, `timeout_s=1800`.

```bash
python -u scripts/agent_ab_run_tasks.py --run-id agent-ab-sdk-replication-001 --driver sdk \
  --instruction-file benchmarks/agent_ab/instructions/hazard-query-v2.txt \
  --dsn postgresql://recall:recall@127.0.0.1:5407/agent_ab --tenant default \
  > benchmarks/artifacts/agent_ab/sdk-replication-001.log 2>&1
python scripts/agent_ab_analyze_tasks.py --run-id agent-ab-sdk-replication-001
python scripts/agent_ab_compare_drivers.py \
  --run benchmarks/artifacts/agent_ab/agent-ab-sdk-replication-001/analysis.json \
  --baseline ~/.claude/archive/agent-ab-skill-001/analysis.json \
  --bands benchmarks/agent_ab/sdk-replication-bands.json \
  --out results/agent_ab/agent_ab_sdk_replication_2026-08-26.json
```

A `--limit 2 --reps 1` smoke under `agent-ab-sdk-replication-001-smoke` runs first, is excluded
from the result, and exists to catch wiring (the skill-001 pattern).

## Decision rule, stated now

- All falsifiers inside bands: the SDK driver becomes the default for subsequent measured runs.
- Any falsifier outside its band: no further SDK-driven measurement until the difference is
  diagnosed from the raw typed streams; the diagnosis is appended here.
- Admitted-pair floor breached: wiring, not evidence; fix and re-run under a new id.

## What I already know

- skill-001 and tasksuccess-001 (54 admitted pairs each) are the only prior runs on this task set;
  their records are `docs/preregistrations/2026-08-22-hazard-query-instruction.md` and
  `docs/preregistrations/2026-08-21-task-success-executable-endpoint.md`.
- The three stream facts the record builder depends on (tool results pair by id, error results are
  bare strings, input tokens sum fresh plus cache components from the per-model aggregate) are
  reused from `claude_exec.py`, not reimplemented, and `tests/test_agent_ab_sdk_exec.py` pins the
  normalizer to that parser.
- The driver passes `--bare` and the append file as literal flags through `extra_args`
  specifically so the system prompt semantics cannot differ (SDK plain-string `system_prompt`
  REPLACES the base prompt; the baselines APPENDED).

## Confounds I can name now

- OpenRouter serving drift since 2026-08-23: the model alias may resolve to updated weights or
  different providers. Cross-run, unpinnable, and the same confound skill-001 itself carried
  against tasksuccess-001. A drift large enough to move task success would most plausibly show as
  BOTH arms moving together; the paired deltas are the comparison for exactly this reason.
- CLI version: the SDK may bundle or resolve a different Claude Code than 2.1.238. The preflight
  asserts at least 2.1.221 (the stdio-MCP wait) and `environment.json` records both versions; if
  the CLI differs from the baseline's, that is recorded here in the result section before any
  interpretation.
- Latency basis: the SDK's typed messages carry no timestamps, so per-call latencies are stamped
  on arrival. `recall_latency_ms` and wall time are therefore recorded-not-falsifying.
- The SDK's own transport overhead lands inside measured wall time, in the same direction for
  both arms.

## Result (2026-08-27)

**Status:** measured. **Verdict: equivalent.** Every falsifier landed inside its committed band;
the committed summary is `results/agent_ab/agent_ab_sdk_replication_2026-08-26.json` and the
published table is `results/RESULTS.md` §13f. Per the decision rule stated above, the SDK driver
is now the default for subsequent measured runs.

Measured against predicted, including the misses:

| metric | predicted | measured | gap |
|---|---|---|---|
| admitted pairs | at least 52 | 50 of 56 | **missed by 2.** All six discards carry one admission reason (on-arm stdio server never available) and cluster around two host reboots that interrupted the run; the wiring floor of 48 held and the gate discarded visibly. |
| search rate | at least 0.95 | 0.905 (38 of 42) | **missed by 0.045.** Four on-arm sessions never searched; band held. |
| governing-memo reach | within 0.10 of 0.674 | 0.605 (diff 0.069) | hit |
| per-task mean delta | within 0.10 of +0.208, same sign | +0.325 (diff +0.117) | **missed the 0.10 point prediction**, inside the 0.15 band; same sign, and larger rather than smaller |
| control means | both 1.000 | 1.000 / 1.000 | hit |
| median input-token overhead | within 25,000 of +106,946 | +103,329 (diff 3,617) | hit |
| median wall-time overhead | HIGHER than +36.5 s | +12.4 s median, negative mean | **wrong in direction.** Recorded-not-falsifying as preregistered; the resumed segments also ran on a host in different load states, which this record cannot separate from the driver. |

Run integrity notes, stated because they are part of the evidence: the run was interrupted twice
by host reboots and resumed under the same id with `--resume` (34 then 48 pairs carried forward);
resumed segments ran in later OpenRouter serving hours, the cross-hour confound named above, and
the paired design carries it. The corpus database had to be migrated to schema 0016 before the
run could start (the serving refusal named the migrations); chunks, generation and calibration
were re-verified intact afterwards. Spend accounting stays with the gateway; the driver's
`total_cost_usd` remains recorded untrusted.

The prediction-calibration note above proved apt in the usual direction: both effect-adjacent
point predictions that missed (admitted pairs, search rate) were optimistic, and the one
directional prediction about a mechanical quantity (wall time) was simply wrong. The bands, set
by decision relevance rather than by the point predictions, absorbed all of it, which is what
they were for.
