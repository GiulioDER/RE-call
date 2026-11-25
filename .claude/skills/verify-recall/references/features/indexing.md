# Indexing

Indexing turns a folder of memos into chunks and vectors in a store, then attempts calibration.

## Sub-features

- `index-folder`: one chunk per short memo, one row per source.
- `index-calibrate`: calibration is attempted after the write and reported on the last lines.

## How to get to it (user POV)

Run `recall --serving-dsn sqlite:///<absolute path>/memory.db --embedder hashing index <folder>`.
`launch` does exactly this for the seed corpus.

## Driving it with control_recall

- **Index.** Run `launch`. The JSON has `"index_exit": 0`, and its evidence file's stdout reads
  `indexed 22 chunks from 22 files`.
- **Calibration result.** The same JSON has `"calibration": "uncertified"` with `hashing`.
- **Second view.** Run `refresh --run <run> --check`: `report.chunks` is 22 and `report.files_seen` 22.

## Gotchas

- A lite DSN must be absolute: on Windows `sqlite:///C:/...`, elsewhere `sqlite:////home/...`
  (four slashes). The CLI's error says which.
- Six memos are too few to calibrate at all (`CALIBRATION_MISSING`: 20 distinct chunks needed);
  that is why the seed is the 22-memo corpus, not the repository's root `corpus/`.
- A store built by `index` has no recorded refresh, so the first `refresh --check` reports
  `"current": false` with exit 1 even though nothing changed.
