# Query-construction benchmark restart note

Updated: 2026-08-27

## Current state

- PR: [#516](https://github.com/GiulioDER/RE-call/pull/516)
- Branch: `codex/query-construction-audit-fix`
- Local HEAD: `3b82dc4ffe9d5b48499b542524fd6db0227ba854`
- Remote branch contains the equivalent harness hardening commit (`424065e8cf6f26908bada4679dcd904b6fcb261b`).
- VPS2 source was deployed and import/registration checks passed before the benchmark launch.
- The competing memory-generation job finished; no benchmark or generation process was left running by this task.

## Run history

The four-item smoke completed successfully as apparatus validation:

- `results/query_construction/runs/agent-ab-skill-001-smoke-20260827T000434Z.json`
- `results/query_construction/runs/agent-ab-skill-001-smoke-20260827T000434Z.summary.json`
- 12 rows total (three arms × four inputs).
- Smoke aggregate: `baseline` 0/4, `original_loop` 1/4, `pyramid` 2/4.
- Provider fallback occurred in some smoke rows; the hardened runner records the fallback instead of aborting.
- These were the first four control inputs, so do not use them as the 15-miss decision result.

The full run was started on VPS2-backed retrieval at `2026-08-27T003343Z` and stopped deliberately with Ctrl+C. Baseline reached 46/46; `original_loop` reached 6/46. No final full-run JSON artifact was created.

## Frozen inputs and integrity

- Frozen population: `C:\Users\gde00\Documents\recall\results\query_construction\agent-ab-skill-001-frozen-population-20260826.json`
- Population SHA256: `a87c7eebe0f7ce7daf45a5d689fa9a4212a938b32297b893cd89c8d89572c2ee`
- Population size: 46 (15 misses and 31 controls).
- Runner SHA256: `5e0d8d070c74b25a252b2ebe52ac6938865cea910599ab6a92b260fb91d8d65c`
- Apparatus addendum SHA256: `1bf005f5d0fc13df2877b565b4172e464fdfbb5bfbc3ca2baa164735bf63dfbb`

## Exact benchmark settings

- Model: `deepseek/deepseek-v4-pro` through OpenRouter
- Reasoning effort: `medium`
- Temperature: `0`
- Maximum output tokens: `1024`
- Provider timeout: `180` seconds
- Provider retries: `3`
- Tenant: `memory`
- Embedder: `voyage:voyage-4`
- Index root: `/home/sentiment/recall-repos/memory`
- Retrieval profile: `fast`
- Graph expansion: `one_hop`

Use the Python 3.12 temporary environment with MCP 2.0.0:

```powershell
$benchPy = 'C:\Users\gde00\AppData\Local\Temp\recall-query-bench-py312-20260826\Scripts\python.exe'
$input = 'C:\Users\gde00\Documents\recall\results\query_construction\agent-ab-skill-001-frozen-population-20260826.json'
$stamp = Get-Date -AsUTC -Format 'yyyyMMddTHHmmssZ'
$output = "results/query_construction/runs/agent-ab-skill-001-full-$stamp.json"
& $benchPy scripts/run_query_construction_batch.py $input $output `
  --model deepseek/deepseek-v4-pro `
  --reasoning-effort medium `
  --max-tokens 1024 `
  --timeout 180 `
  --retries 3 `
  --tenant memory `
  --embedder voyage:voyage-4 `
  --index-root /home/sentiment/recall-repos/memory `
  --profile fast `
  --graph-expansion one_hop
```

Before launching, verify on VPS2 that no `index_memory_manifest.sh`, `generation build`, or live holder of `.locks/embed.lock` or `.locks/serving.lock` remains. Do not remove a lock with a live holder. If both locks are stale after the owner check, remove only those two explicit lock files, then verify the deployed imports and MCP registration.

After completion, create the deterministic summary:

```powershell
& $benchPy scripts/summarize_query_construction_batch.py $output "$output.summary.json"
```

Expected completion shape is 138 rows (46 inputs × three arms). Keep the raw JSON and summary immutable; do not overwrite the smoke artifacts.

## Deployment fingerprints recorded on VPS2

The deployed source files matched the local source before the stopped run:

- `recall/query_construction.py`: `1a782a05a91de4874d5520cffd4f593d2dbcc23e434237ced211cba847244e66`
- `recall_mcp/service.py`: `0bbfaa1ca64e0383a05702e8449aca52286e7980be2510bdf1d53d14d2fdbd12`
- `recall_mcp/server.py`: `caa5cf68010f9fc97bdc8d5ed0c6163bb26a0983a8cff11fc06651ba0e9ebad3`

The source backup is under `/home/sentiment/agent-memory-bench/query-construction-backup-20260826T233648Z`.

## Important restart caveat

The benchmark controller and DeepSeek calls run on the local workstation, while VPS2 serves retrieval. Keep this task/terminal open for the run, or relaunch it inside a deliberately detached local process. Do not start a second run while one is active.
