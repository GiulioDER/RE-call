# Forget

`recall forget` removes a source's rows (and what it told the fact ledger) from the store. It
previews by default and deletes only with `--yes`.

## Sub-features

- `forget-preview`: `DRY RUN: would forget N source(s): ...`, nothing deleted.
- `forget-apply`: `forgot N chunk(s) from N source(s).`
- `forget-then-refresh`: what a later refresh does with the file still in the folder.

## How to get to it (user POV)

Run `recall forget <source> [--yes]`, where `<source>` is the path the store recorded.

## Driving it with control_recall

- **Preview.** After the lite recipe added `zephyr_rollout.md`, run `forget --run <run>
  zephyr_rollout.md`. `applied` is false and `stdout` says `would forget 1 source(s)`.
- **Apply.** Add `--yes`. `stdout` says `forgot 1 chunk(s) from 1 source(s).`
- **Second view.** `search --run <run> --development "zephyr rollout waves"` no longer lists
  `zephyr_rollout.md`, and `refresh --check` reports `current` `false` (23 files, 22 chunks).

## Gotchas

- The store names a source by its native absolute path (backslashes on Windows). A mixed-slash
  spelling is reported `not found`; `control_recall` turns a bare memo name into the stored path.
- **Forget does not delete the file, and the next `refresh` indexes it again** (observed
  2026-11-25: `indexed_files` 1, and search returns the memo). The owner's rule that forget keeps
  retired what the memo SUPERSEDED is a different guarantee. `docs/LITE.md` does not say this; to
  keep a memo forgotten, delete its file too.
