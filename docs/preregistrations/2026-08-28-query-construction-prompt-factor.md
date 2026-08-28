# Preregistration: query-construction prompt factor

Date: 2026-08-28

## Question

After increasing the original-model output budget, does explicitly asking for the governing
invariant, known failure mode, or decision rule improve query construction and graph recovery?
This is a prompt-factor follow-up to the completed 2,048-token exploratory replay. It must not be
pooled with that replay because the new wording was not deployed during it.

## Frozen inputs and retrieval snapshot

Use the unchanged `agent-ab-skill-001` population: 15 known misses and 31 hit controls, with the
original prompts and initial queries fixed and labels hidden until scoring. The input SHA256 is:

`a87c7eebe0f7ce7daf45a5d689fa9a4212a938b32297b893cd89c8d89572c2ee`

Both prompt arms must use this explicit read-only VPS2 generation pin:

* generation: `gen_168361bd2310433e87beda1fc6f4a5e0` (retired, readable snapshot);
* corpus fingerprint: `0448c9d4a25d00c4561790b75a0d5286e0480d1b894c2aea4c4c74061db558d6`;
* pipeline fingerprint: `77c918cd93f9200b36e505ae874d49a1949f04a8a85ce5b8b72a8c135b472db7`;
* indexed chunks: 9,968.

The current active generation contains later benchmark-memory notes and is not a valid comparison
snapshot. It must not be promoted or used for scoring.

## Paired runs

Run the baseline and both construction arms once against the pinned snapshot for each phase. Keep
the task ordering, workers, graph mode, and all model settings identical.

1. `legacy-prompt-2048`: deployed challenge wording, no challenge marker.
2. `invariant-prompt-2048`: deployed wording requiring the literal marker `governing invariant`.

The baseline is repeated in each phase to verify snapshot stability; it is not a new algorithmic
arm. The two prompt phases are exploratory paired comparisons, not a replacement for the earlier
registered decision rule.

## Fixed apparatus

* Model: `deepseek/deepseek-v4-pro` through OpenRouter.
* Reasoning effort: `medium`.
* Temperature: `0`.
* Maximum output tokens: `2048`.
* Provider timeout: `180` seconds; retries: `3`.
* Tenant: `memory`; embedder: `voyage:voyage-4`; profile: `fast`.
* Graph expansion: `one_hop`; benchmark workers: `2`.
* Maximum two construction rounds, three candidates per round, and one original-model challenge
  per round.

The benchmark runner must pass `--pinned-generation-id` for both phases and
`--challenge-marker "governing invariant"` for the new-prompt phase. Use distinct immutable raw,
summary, and checkpoint paths. A checkpoint may be resumed only with identical settings and input
hash.

## Launch recipe

After the pinned benchmark server version is deployed and the smoke gate passes, use the existing
Python 3.12 benchmark environment and frozen input:

```powershell
$benchPy = 'C:\Users\gde00\AppData\Local\Temp\recall-query-bench-py312-20260826\Scripts\python.exe'
$input = 'C:\Users\gde00\Documents\recall\results\query_construction\agent-ab-skill-001-frozen-population-20260826.json'
$generation = 'gen_168361bd2310433e87beda1fc6f4a5e0'
$common = @('--model', 'deepseek/deepseek-v4-pro', '--reasoning-effort', 'medium', '--max-tokens', '2048', '--timeout', '180', '--retries', '3', '--tenant', 'memory', '--embedder', 'voyage:voyage-4', '--index-root', '/home/sentiment/recall-repos/memory', '--profile', 'fast', '--workers', '2', '--graph-expansion', 'one_hop', '--pinned-generation-id', $generation, '--resume')

# Four-input apparatus smoke; discard from scoring.
& $benchPy scripts/run_query_construction_batch.py $input 'results/query_construction/runs/agent-ab-skill-001-prompt-factor-smoke-legacy.json' @common --limit 4
& $benchPy scripts/run_query_construction_batch.py $input 'results/query_construction/runs/agent-ab-skill-001-prompt-factor-smoke-invariant.json' @common --limit 4 --challenge-marker 'governing invariant'

# Full paired phases, with separate raw/checkpoint artifacts.
& $benchPy scripts/run_query_construction_batch.py $input 'results/query_construction/runs/agent-ab-skill-001-prompt-factor-legacy-2048.json' @common
& $benchPy scripts/run_query_construction_batch.py $input 'results/query_construction/runs/agent-ab-skill-001-prompt-factor-invariant-2048.json' @common --challenge-marker 'governing invariant'
```

Generate each summary with `scripts/summarize_query_construction_batch.py` and record SHA256
digests for both raw and summary files. If a process exits before its raw file is written, rerun
the same command with the same output path and `--resume`; never reuse a checkpoint for the other
prompt phase.

## Recorded diagnostics

Each artifact must preserve the raw MCP calls, challenge prompts, prompt hashes, model frames,
provider finish reasons and token usage, accepted and rejected candidates, trusted evidence IDs,
generation bindings, graph readiness/gate/admission diagnostics, retrieval and model call counts,
latency, cost, fallback/refusal reasons, and the apparatus settings. Every response must report
the pinned generation; a missing or different generation is an apparatus failure, not a retrieval
miss.

## Outcomes

Primary outcomes are governing-memo recovery in top five on the 15 misses and retention on the 31
controls. Secondary outcomes are valid-frame rate, provider truncation, query novelty, accepted
and rejected candidates, new trusted evidence, graph activation and candidates, model/retrieval
calls, latency, token/provider cost, and fallback rate.

Report each phase separately and as paired deltas. Do not claim a prompt effect if deployment,
generation binding, marker receipt, or model-frame return fails.

## Stop/go gate

Before the first live call, require the implementation commit/PR, VPS2 import and tool-registration
smoke checks, a four-input apparatus smoke, and confirmation that every smoke response carries the
explicit pinned generation. The new-prompt phase additionally requires the marker in every issued
challenge. If any gate fails, stop and report apparatus failure with no retrieval conclusion.

The earlier adoption rule remains unchanged for algorithm selection: prefer `pyramid` only if it
rescues at least 5 of 15 misses, retains at least 30 of 31 controls, and stays within two retrieval
rounds; prefer `original_loop` only if it reaches that rescue bar and pyramid does not. This
prompt-factor test can explain a change in recovery, but does not by itself authorize promotion.

## Results appended after measurement

Measured on 2026-08-28 against the frozen 46-input population and pinned generation above. Both
prompt phases completed all 138 rows, with no duplicate task-arm pairs. The raw and summary
artifacts remain in the external benchmark results directory.

| Phase | Arm | Recovery | Misses rescued | Controls retained |
| --- | --- | ---: | ---: | ---: |
| Legacy | baseline | 17/46 | 0/15 | 17/31 |
| Legacy | original_loop | 20/46 | 0/15 | 20/31 |
| Legacy | pyramid | 27/46 | 3/15 | 24/31 |
| Invariant | baseline | 17/46 | 0/15 | 17/31 |
| Invariant | original_loop | 18/46 | 0/15 | 18/31 |
| Invariant | pyramid | 25/46 | 1/15 | 24/31 |

The original loop did not rescue a known miss in either phase. Pyramid rescued three legacy misses
and one invariant miss, below the registered threshold of five, and retained 24 of 31 controls,
below the registered threshold of 30. The adoption rule therefore rejects both construction arms.
The result supports retaining the current retrieval path and investigating index aliasing or a
separate memory representation.

The invariant apparatus passed: all 138 rows carried generation
`gen_168361bd2310433e87beda1fc6f4a5e0`, and all challenge records carried the marker. The original
loop produced 69 model calls, 40 complete rows, and 6 invalid-frame fallbacks. Pyramid produced
67 model calls, 44 complete rows, and 2 invalid-frame fallbacks. These fallbacks are reported
separately from retrieval misses.

| Artifact | SHA256 |
| --- | --- |
| `agent-ab-skill-001-prompt-factor-legacy-2048.json` | `2A99107D1D8B1E1ABCB8B3A1D8913FE84AF0391B8590BECF2C5CB394E07D2D93` |
| `agent-ab-skill-001-prompt-factor-legacy-2048.summary.json` | `93798CC03D83D214FF828097A4AA2B3F6A4A96D940F06A1BE8E5F9BDECB45281` |
| `agent-ab-skill-001-prompt-factor-invariant-2048.json` | `4DAB90A5F6AD22D3108FCEE90563E84024AD993B34F207A4A4B7E3FEC27E10CC` |
| `agent-ab-skill-001-prompt-factor-invariant-2048.summary.json` | `7DB440D78B45EF758A1D08CC455A97C7ED1F81AB3500DAD2FD17BAAAB98B3E81` |
