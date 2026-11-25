# Doctor

`recall doctor` says which part of an install is wrong, and writes nothing.

## Sub-features

- `doctor-json`: one entry per check with `name`, `status` (`ok`, `warn`, `fail`, `skip`), `detail`.

## How to get to it (user POV)

Run `recall --serving-dsn <dsn> doctor --json --project-root <project>`.

## Driving it with control_recall

- **Health.** Run `doctor --run <run>`. `healthy` is true, `checks.database` and `checks.corpus` are
  `ok`, `checks.docker` is `skip` (a lite store needs none).
- **Which code ran.** The evidence file's `recall package` detail names this checkout's `recall`
  directory; `control_recall doctor` fails if it names anything else.

## Gotchas

- `calibration` always warns (hashing never certifies). `claude code` warns where a Claude Code
  configuration exists and is skipped where none does, such as CI; the run is registered with no
  client on purpose.
- Doctor checks the embedder it is given, else the default `fastembed`, which CI does not install;
  `control_recall doctor` passes the run's own (observed on CI 2026-11-25: `embedder: fail` before).
- `--project-root` is the run's scratch folder, so doctor never reads the checkout's own client
  configuration as if it were the run's.
