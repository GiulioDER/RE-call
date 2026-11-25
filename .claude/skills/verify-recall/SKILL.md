---
name: verify-recall
description: "Drive RE-call the way a user does (the `recall` CLI and the MCP server over stdio) on a throwaway lite store built from this checkout, and capture proof. Use to verify a change to indexing, search and the trust gate, supersession, lite refresh, forget or the MCP tools works, before calling it done."
---

# Verify RE-call

Prove a change by driving the real user path against a disposable store, never by reading the code
back. Every command below runs THIS checkout's code (`python -m` from the checkout root), prints one
JSON object, and saves its argv, stdout, stderr and exit code under the run's `evidence/`.

```bash
python .claude/skills/verify-recall/control_recall.py --help
```

Generated from pstack's `/create-verification-skill` (github.com/cursor/plugins, MIT, Lauren Tan)
and adapted to this repository.

## Upkeep

- **Nightly, by itself:** the `verify-recall walk` workflow runs `walk.py` on master, which drives
  every recipe below and checks each stated end state. A failure opens (or comments on) the issue
  "verify-recall nightly walk failed", with the failed checks in the run summary and the evidence
  as an artifact. Fix the feature file when the map drifted, the code when RE-call regressed.
- **With every recipe change:** change its check in `walk.py` in the same commit; the workflow
  walks a pull request that touches this directory.
- **Structure:** `tests/test_verify_recall.py` refuses a feature file whose name is not a feature
  id in `docs/feature_map.toml`, one without the four sections, or one the index does not link.
- **Weekly, by an agent:** the maintainer's cloud routine "verify-recall weekly maintenance"
  (Mondays 05:00 UTC) does the judgment the walk cannot: reads each feature file against its
  modules and the week's commits, looks for new commands and tools with no file, runs the walk,
  and opens at most one pull request, on a `claude/verify-recall-maintain-<date>` branch, editing
  only this directory. It never merges, and opens nothing while such a pull request is open.

## Launch

```bash
python .claude/skills/verify-recall/control_recall.py launch
```

Creates `<temp>/recall-verify/<run>/`, copies the 22-memo seed corpus `recall/eval/corpus` into
`scratch/notes/`, and indexes it into `scratch/memory.db` (`sqlite:///`, no server, no Docker).
Ready means the JSON says `"index_exit": 0` and names a `run`; pass that `run` to every other
command. Teardown is `cleanup`.

The default embedder is `hashing`: no model, nothing downloaded, deterministic. **It never
certifies** (separability about 0.80 against a 0.90 floor), so strict search refuses with
`CALIBRATION_UNCERTIFIED` or, after a refresh, `CALIBRATION_STALE`. That is correct behavior and is
itself a proof target. Ranking and supersession are checked with `--development`, which sets
`RECALL_TRUST_MODE=development` and marks every result DEGRADED. A certified run needs a real
embedder (`launch --embedder fastembed`). Launch one only on a machine where your own rules allow
loading an embedding model; the maintainer's do not allow it on their workstation.

## Doctor

```bash
python .claude/skills/verify-recall/control_recall.py doctor --run <run>
```

Read-only. `"healthy": true` requires `recall doctor --json` to have no `fail`, the `recall` package
to load from this checkout, and the run's store to exist. Run it before the first drive and after any
drive that surprised you. `calibration` always warns (hashing never certifies); `claude code`
warns where a Claude Code configuration exists and is skipped where none does (CI), because the run
is registered with no client on purpose.

## Drive

| Command | What it drives |
|---|---|
| `search --run R [--development] "question"` | `recall search`; returns `code` (refusal or degradation) and parsed `rows` (`state`, `conf`, `cos`, `source`, `use`) |
| `refresh --run R [--check]` | `recall lite refresh --json`; `report.current`, `indexed_files`, `chunks_written` |
| `write-memo --run R name.md "text"` | creates or replaces one memo in the run's folder |
| `forget --run R name.md [--yes]` | `recall forget`; previews unless `--yes` |
| `mcp --run R [--development] tool --args '{json}'` | starts `recall_mcp.server` over stdio on the run's store, lists tools, calls one; `structured` is the decoded answer |
| `list` | every run under the verify root, which are this checkout's |

Pick the feature file under [`references/features/`](references/features/README.md) for the change,
and drive every entry point it lists.

## Evidence

Each drive writes `evidence/NNN-<command>.json` inside the run, and the JSON on stdout names it. A
proof is the action and the resulting state together: the drive that changed something, then a
second, read-only view of the result (a search, a `refresh --check`, an MCP call). Name the run id
and the evidence files in the PR or the report.

- Drive the real path: the CLI and the MCP server, not a Python import of the function you changed.
- A refusal is a result. Assert its code, not that "it errored".
- A development-mode answer proves ranking and supersession only; it never proves trust.

## Cleanup

```bash
python .claude/skills/verify-recall/control_recall.py cleanup --run <run>        # --dry-run to preview
```

Removes the run's `scratch/` (store and memos) and keeps `evidence/`; the JSON lists what was kept.
Commands refuse a run another checkout launched, and cleanup removes nothing outside the run. Old
evidence folders are left for a person to delete.
