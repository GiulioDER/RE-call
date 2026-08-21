# Does adding RE-call to CLAUDE.md reduce known-hazard mistakes?

**Written 2026-08-21, before any session of this run was executed.** Run identifier
`agent-ab-additive-001`.

## Why this run exists, and what was wrong with the last one

The previous run compared `claude_md` against **RE-call alone**, which measured *replacing* the
hand-written file. That is not a configuration anybody uses, and its control said so: on traps whose
fact lives in `CLAUDE.md`, the arm holding `CLAUDE.md` won by 38 points, purely because the other
arm did not have the file. The 50-point primary result stands, but it answers a question nobody
asked.

This run makes the comparison **additive**:

| arm | system prompt | memory layer |
|---|---|---|
| off (`claude_md`) | `CLAUDE.md` + `MEMORY.md` | none |
| on (`claude_md_recall`) | the **same bytes**, plus one sentence naming the tool | RE-call |

The runner asserts that the on arm's prompt begins with the off arm's byte for byte and aborts
otherwise, so the arms cannot silently differ by more than the memory layer.

## Fixed configuration

| Field | Value |
|---|---|
| Tasks | `benchmarks/agent_ab/tasks/traps-deep.jsonl`, 10 traps |
| Primary | 4 `memory_only` traps x 10 repetitions = **40 pairs** |
| Controls | 6 traps (`both`, `claude_md_only`) x 4 repetitions = **24 pairs** |
| Agent | `anthropic/claude-haiku-4.5` via OpenRouter, `--bare` |
| Judge | `openai/gpt-4.1-mini` via OpenRouter, `max_tokens=8192` |
| Corpus | calibrated generation, threshold 0.731, separability 0.980 [0.952, 1.000] |
| Transport | stdio, `RECALL_ENV=production`, strict trust |
| Database | `recall-agentab-corpus` on port 5407, **named volume** |
| Claude Code | 2.1.238 |

Three defects from the last run are fixed before this one starts, and each cost something:

1. **Records are written incrementally and fsynced.** The last runner died at 71 of 100 sessions
   and lost its index; only the transcripts survived. A dead run should cost the sessions it did
   not do, not the ones it did.
2. **Output goes to a log file, never through `grep`.** Piping the last run through `grep -v`
   swallowed every diagnostic and masked the exit code, so its cause is permanently unknown.
3. **The corpus lives in a container with a named volume.** The session container was removed by
   something outside the session three times, destroying the corpus each time.

## Endpoints and how they will be read

**Primary: hit rate on the 4 `memory_only` traps**, where the governing fact is in the memory store
and not in `CLAUDE.md`. Reported two ways, both preregistered here:

- **Per-task (headline).** One rate per trap, repetitions collapsed. ⚠️ With 4 distinct traps a
  sign test cannot reach p<0.05 **at any effect size**, so this view is reported as descriptive
  (how many traps improved, by how much) with a cluster bootstrap that resamples traps. No p-value
  will be invented for it. This limit is a property of running 4 hazards and is stated wherever the
  result appears.
- **Per-pair (consistency check).** Exact McNemar over all 40 pairs. This will produce the smaller
  p-value and it **overstates confidence**, because repetitions of one trap are correlated. It is
  secondary and labelled as such.

**Controls: the 6 traps whose fact is in `CLAUDE.md`.** Both arms hold the file, so both should
avoid these. A difference here is noise or a side effect of the memory layer, not a benefit, and
either reading weakens the primary claim.

## Predictions

Predicted before measuring, per the standing rule. Last run I **under**-predicted for the first
time; the correction here is not to simply predict bigger, because the treatment has changed: the
on arm now also carries 17.8k characters of `CLAUDE.md`, which gives it an alternative to searching
and may suppress tool use.

1. Off arm `memory_only` hit rate: **0.65**. (Measured 0.688 for the same arm last run.)
2. On arm `memory_only` hit rate: **0.30**. Worse than the 0.188 that RE-call alone achieved,
   because the added static context competes with the tool.
3. Reduction: **35 points**. Falsified if under 12 points, or if the cluster interval includes zero.
4. Per-task: **3 of 4** traps improve.
5. `recall_search` called in **65%** of on-arm `memory_only` sessions, DOWN from 81% when RE-call
   was the only memory available.
6. Governing memo in retrieved contexts in **45%** of those sessions.
7. Controls: difference within **10 points** either way, in both control loci.
8. On-arm input tokens **+8%**. Both arms now carry the static bundle, so the delta is the tool
   definitions plus retrieved text, not the file.
9. On-arm wall time **+10 s** median, of which ~9 s is stdio server startup, reported separately.
10. `recall_latency_ms` median under **400 ms**. (182 ms measured last run.)
11. Ragas `answer_correctness`: **no significant difference**, point estimate within +0.05.

## Exclusions and stopping rules

A pair is discarded, not scored, unless the on arm's `system/init` lists a `mcp__recall*` tool and
the off arm's does not. Missing measurements stay null and are never replaced with zero. No task,
arm, metric or stopping rule changes after any data is seen. Any rerun gets a new identifier and a
new record.

If the run dies again, the salvage path rebuilds records from `records.partial.jsonl` and the
transcripts, and the result says so, including how many pairs were lost.

## Stated limitations, fixed in advance

- **4 distinct hazards.** The claim generalises to the population of hazards these 4 represent, and
  no further. This was a deliberate choice to get data today rather than authoring 8 more traps.
- The corpus is this project's own memory store, so the result describes an agent working in the
  repository whose memory this is.
- The generation was built `--unverified-development`: fit for a benchmark, not for a trust claim.
- `total_cost_usd` from the CLI is not reported; it was measured about 6x wrong through the gateway.

## Result

*Not yet run. Appended below when it is, without editing anything above.*
