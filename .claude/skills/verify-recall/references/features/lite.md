# Lite refresh

`recall lite refresh` keeps a one-file store current with its memo folder: new and changed memos
are indexed, removed ones pruned, and calibration is re-verified or marked due for a refit.

## Sub-features

- `refresh-check`: reports what a refresh would find; changes nothing.
- `refresh-apply`: indexes what changed and records the run.

## How to get to it (user POV)

Run `recall lite refresh --store <store> --memory <folder> [--check] [--json]`.

## Driving it with control_recall

- **Baseline.** On a fresh run, `refresh --run <run>` then `refresh --run <run> --check`:
  `report.current` is `true`.
- **Add a memo.** Run `write-memo --run <run> zephyr_rollout.md "# Zephyr rollout ... the zephyr
  flag ..."`. Then `refresh --check` reports `current` `false` and `files_seen` 23.
- **Apply.** Run `refresh --run <run>`: `report.indexed_files` 1, `chunks_written` 1,
  `chunks_total` 23.
- **Second view.** Run `search --run <run> --development "zephyr rollout waves"`: the first row's
  `source` is `zephyr_rollout.md`.

## Gotchas

- `--check` exits 1 for "not current" with a full JSON report; that exit is the answer, not a
  failure. `control_recall` decides on the report.
- After a refresh that changes the corpus, calibration moves from `uncertified` to `stale`, and
  strict search's code changes with it.
