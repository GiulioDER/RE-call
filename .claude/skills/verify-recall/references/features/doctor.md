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

- Two `warn`s are expected on every run: `calibration` (hashing never certifies) and `claude code`
  (the run is registered with no client, on purpose).
- `--project-root` is the run's scratch folder, so doctor never reads the checkout's own client
  configuration as if it were the run's.
