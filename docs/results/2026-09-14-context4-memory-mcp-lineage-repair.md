# Context 4 memory MCP lineage repair

Measured: 2026-09-14.

## Cause

The active `memory` generation and its published calibration are bound to
`voyage-context:voyage-context-4`, but five production launch and verification surfaces still
forced `RECALL_EMBEDDER=voyage:voyage-4`:

* `.mcp.json`
* `scripts/session-mcp.sh`
* `scripts/session_serving_remote.sh`
* `scripts/check_recall_mcp_stdio.py`
* `scripts/launch_codex_ab_smoke.py`

The models have the same vector dimension, so the dimension check could not diagnose the problem.
The strict model lineage check correctly refused before retrieval with `LINEAGE_MISMATCH`.
Recalibration was not appropriate because the active calibration already matched the generation's
pipeline, corpus, and query set.

The machine wide Codex launcher at `C:\Users\gde00\.codex\config.toml` had the same stale override
and was corrected locally. Existing long lived MCP children retain the environment with which they
were created and must reconnect. They were not killed because indistinguishable children can belong
to other active tasks.

## Repair

Every canonical production memory launcher now selects
`voyage-context:voyage-context-4`. Historical benchmark controls that deliberately test Voyage 4
remain unchanged.

The new contract test failed before the launcher repair at the intended assertion:

```powershell
python -m pytest tests/test_context4_memory_launchers.py -q
```

Observed before repair: `1 failed`, reporting that `.mcp.json` did not select Context 4.

After repair, the launcher and lineage checks passed:

```powershell
python -m pytest tests/test_context4_memory_launchers.py tests/test_trust_embedder_lineage.py -q
python -m ruff check tests/test_context4_memory_launchers.py scripts/check_recall_mcp_stdio.py scripts/launch_codex_ab_smoke.py
python -c "import tomllib, pathlib; tomllib.loads(pathlib.Path(r'C:\Users\gde00\.codex\config.toml').read_text(encoding='utf-8')); print('config.toml valid')"
git diff --check
```

Observed: `1 passed, 5 skipped`, Ruff passed, the global TOML parsed, and the diff check passed. The
five skips are database backed lineage tests unavailable in the local test environment; the live
probe below covers the production boundary.

## Fresh process proof

A fresh isolated MCP process was launched through the corrected command and a raw JSON RPC
`recall_search` was executed against tenant `memory`. The reduced result was:

```json
{
  "isError": false,
  "trust_state": "trusted",
  "calibrated": true,
  "failure_code": null,
  "generation_id": "gen_795624b8539e4b71b748405cf4ab886f",
  "calibration_id": "cal_1199bce80573476586f2556baaad11d9"
}
```

The deployed serving verification also passed import, schema migration `0025`, and a 22 tool MCP
handshake with the Context 4 selector:

```powershell
ssh -T vps2 "cd ~/recall-repos/serving && RECALL_SERVING_EMBEDDER=voyage-context:voyage-context-4 bash scripts/session_serving_remote.sh verify"
```

No generation, calibration, active route, or serving checkout was changed by this repair.
