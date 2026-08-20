# Claude Code with and without RE-call, on a calibrated corpus

**Written 2026-08-21, before any paired session was run.** Supersedes
`2026-08-20-claude-code-with-and-without-recall.md`, which predicted against an uncalibrated
corpus and a warm HTTP transport; neither is the configuration any more, and that record is kept
unedited with the reason appended to it.

Run identifier: `agent-ab-traps-002`.

## Question

Does an agent doing ordinary work in this repository make fewer known, expensive mistakes when it
has a **calibrated** retrieval memory layer than when it has the hand-written `CLAUDE.md`, and what
does that cost in tokens and wall time?

## Measured before this was written, and therefore not predictions

| Fact | Value |
|---|---|
| Corpus | 188 sources, 981 chunks, from the project memory store |
| Generation | `gen_69eafbee8bcd459fb5e45b6eea8f6c3d`, active, built `--unverified-development` |
| Calibration | `cal_eb7c63dca4424f22b42609aa3f8491c5`, published, certified |
| Threshold | **0.731**, against an uncertified demonstration default of 0.5 |
| Separability | **0.980, 95% CI [0.952, 1.000]**, 50 answerable / 50 unanswerable |
| Score distributions | answerable median 0.789, unanswerable median 0.674; 3 of 50 answerable fall below the threshold |
| Trust state served | `trusted`, `calibrated: true`, under the **strict** policy |
| Transport | stdio, `RECALL_ENV=production`, about 9 s per session |
| Claude Code | 2.1.238 (upgraded from 2.1.220, which does not wait for a pending MCP server) |
| Static bundle | 17,499 chars; memory store 636,191 chars across 185 files, about 36:1 |
| Trap loci | 4 `memory_only`, 4 `both`, 2 `claude_md_only` |

The labelled query set (`benchmarks/agent_ab/calibration/memory-query-set.json`) shares **no query**
with any trap probe, so the threshold was not fitted on the test set.

### The abstention finding, which the design must now account for

At 0.731 the corpus **abstains on two of the four `memory_only` trap queries** (`omp_threads`,
`cairo_render`) even though the governing memo is the top hit. Hits are still returned, so the
agent can use them; abstention is a statement about confidence, not a refusal to answer. This was
invisible before calibrating, because at 0.5 nothing ever abstained. It is kept and reported rather
than tuned away: per-trap abstention is a primary diagnostic below.

## Design

Unchanged from the superseded record except for the configuration above. Paired, one task at a
time, both arms started together; three arms run as two comparisons (`recall` vs `claude_md` is the
headline, `recall` vs `bare` the ceiling); `--bare` in every arm so context arrives only by flag;
`docker` denied in every arm so a shared-container hazard is measured from the denial and never
realised. Task set `benchmarks/agent_ab/tasks/traps.jsonl`, 10 tasks, 5 repetitions.

**Admission.** A pair is discarded, not scored, unless the on arm's `system/init` lists a
`mcp__recall*` tool and the off arm's does not. A tool that was available and never called is
admitted and counted.

## Predictions

Predicted low again, and lower than the superseded record on the primary endpoint, because
abstention now fires on half the winnable traps. The house record is eleven of twelve predictions
falsified, all too high by two to four times.

**Primary: trap hit rate on the 4 `memory_only` traps, headline comparison.**

1. Off arm (`claude_md`) hit rate: **0.70**.
2. On arm (`recall`) hit rate: **0.50**.
3. Absolute reduction: **20 percentage points**. Falsified below 8 points, or if the 95% interval
   on the paired difference includes zero.

**Mechanism, predicted beside the outcome so a miss can be attributed.**

4. `recall_search` is called in at least **70%** of on-arm sessions on `memory_only` tasks.
5. The governing memo appears in the retrieved contexts in at least **50%** of those sessions.
6. On the two traps that abstain, the on-arm hit rate is **worse by 10 to 25 points** than on the
   two that do not. If retrieval fires and the memo comes back but the abstaining traps do not
   improve, the limiting factor is the agent's willingness to act on evidence the layer flagged as
   low confidence, not retrieval.

**Controls, which is why traps RE-call should lose are in the set.**

7. `both` traps: difference within **10 points** either way.
8. `claude_md_only` traps: on arm **worse by 10 to 30 points**.

**Cost of the memory layer.**

9. On-arm input tokens **+40%** against `claude_md`, revised up from the superseded record's 20%
   after a smoke pair measured 26,359 against 18,189, which is +45%. Sixteen MCP tool definitions
   enter the context whether or not a search happens.
10. On-arm wall time **+12 s** per session, median, of which about **9 s is stdio server startup**
    and the rest is the extra turn. Reported with the startup component separated, because a warm
    deployment does not pay it per session and a reader must be able to subtract it.
11. `recall_latency_ms` accounts for under **1 s** of that.

## Exclusions, stopping rules, analysis, artifacts

As in the superseded record, unchanged: no post-hoc task removal or metric change; missing values
stay null; Wilcoxon on tokens and wall time, McNemar on binary outcomes, bootstrap intervals on
paired differences; `total_cost_usd` from the CLI is not reported, because it was measured about
6x wrong through the gateway.

**Stated limitations.** The generation was built `--unverified-development`, so its manifest is not
cryptographically verified: this corpus is fit for a local benchmark, not for a trust claim. The
corpus is the project's own memory store, so the result generalises to "an agent working in the
repository whose memory this is", and no further.

**Publication gate.** Retrieved chunks land verbatim in transcripts and this corpus contains host
inventory and pointers to credential files. No transcript is published unreviewed; the preferred
route is a rerun against a filtered corpus.

## Result

*Not yet run. Appended below when it is, without editing anything above.*
