# Forget

`recall forget` removes a source's rows (and what it told the fact ledger) from the store. It
previews by default and deletes only with `--yes`. A memo you forgot stays forgotten while its file
is still in the folder: refreshes and folder indexes leave it out until the file is indexed by name.

## Sub-features

- `forget-preview`: `DRY RUN: would forget N source(s): ...`, nothing deleted.
- `forget-apply`: `forgot N chunk(s) from N source(s).`
- `forget-stays-forgotten`: a refresh leaves the file out and counts it as `forgotten`.
- `forget-bring-back`: `recall index <file>` re-adds it and clears the mark.

## How to get to it (user POV)

Run `recall forget <source> [--yes]`, where `<source>` is the path the store recorded. To bring a
forgotten memo back, run `recall index <folder>/<memo>.md`.

## Driving it with control_recall

- **Preview.** After the lite recipe added `zephyr_rollout.md`, run `forget --run <run>
  zephyr_rollout.md`. `applied` is false and `stdout` says `would forget 1 source(s)`.
- **Apply.** Add `--yes`. `stdout` says `forgot 1 chunk(s) from 1 source(s).`
- **Second view.** `search --run <run> --development "zephyr rollout waves"` no longer lists
  `zephyr_rollout.md`, and `refresh --check` reports `current` `false` (23 files, 22 chunks).
- **Stays forgotten.** Run `refresh --run <run>`: `report.indexed_files` 0, `report.forgotten` 1,
  `chunks_total` 22; a following `refresh --check` reports `current` `true`, and search still does
  not list the memo.
- **Bring back.** Run `index --run <run> zephyr_rollout.md`: exit 0. Search lists
  `zephyr_rollout.md` first again, and the next `refresh --run <run>` reports `forgotten` 0 and
  `chunks_total` 23.

## Gotchas

- The store names a source by its native absolute path (backslashes on Windows). A mixed-slash
  spelling is reported `not found`; `control_recall` turns a bare memo name into the stored path.
- Forget does not delete the file. Editing it does not bring it back either (owner decision
  2026-11-25, so an edit that keeps erased text cannot re-leak it); only indexing it by name does.
- A file deleted from the folder is pruned by the next refresh and indexed again when it returns;
  only a person's forget or withdraw leaves the mark.
- PostgreSQL stores do not record the mark; this recipe covers lite stores only.
