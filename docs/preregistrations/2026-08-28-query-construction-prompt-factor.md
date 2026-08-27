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
