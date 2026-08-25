# Pre-registration: a lighter Docker daemon probe for the first command a new user runs

**Date:** 2026-08-25   **Status:** predicted, not yet measured

## The question

Does replacing `docker info --format {{.ServerVersion}}` with `docker version --format
{{.Server.Version}}` in `recall.quickstart.docker_unavailable_reason` reduce the wall clock of that
function by more than 5x on this machine, while still distinguishing all three states it exists to
distinguish (docker absent, daemon not answering, healthy)?

## What I predict

- **Current `docker info` probe: 30 to 40 seconds.** One observation of **34.77s** is already in
  hand, taken while writing `recall doctor`; that observation is what prompted this record, and
  the range above is my prediction for the median of a fresh set of runs rather than a restatement
  of it.
- **`docker version`: 1 to 3 seconds**, so a speedup of roughly **12x to 35x**.
- **Both probes agree on all three states.** Specifically: identical verdict on a healthy daemon,
  and identical verdict (non-zero exit) with `DOCKER_HOST` pointed at an address nothing serves.

I am predicting a large multiple deliberately rather than hedging, because the mechanism is not
subtle: `docker info` gathers the full system inventory (containers by state, images, plugins,
storage driver, registry config) and this machine currently holds 37 containers and a dozen images,
while `docker version` is one round trip to the daemon's `/version` endpoint and returns a fixed
handful of fields. If the speedup is small, the mechanism I believe in is wrong.

⚠️ Against my own record: [[i-over-predict-effect-magnitudes]] says eleven of twelve predictions
here were too high by two to four times. I am predicting a large effect anyway, and noting that if
the measured speedup lands near 3x rather than 12x, that memo predicted this better than I did.

## What would falsify this

Any of:

- `docker version` median under 5x faster than `docker info` median.
- The two disagree on any of the three states, in either direction. A faster probe that reports a
  broken daemon as healthy is strictly worse than a slow correct one, and this failure is the one
  that matters: the quickstart writes a compose file and starts a container on the strength of this
  answer.
- `docker version` succeeds while the daemon cannot actually run a container (for example, a
  reachable CLI proxy with no engine behind it). This would mean the probe answers a weaker
  question than the one the caller asks, which is the "answering a different question well" trap.

## How it will be measured

Five runs of each probe, alternating, on this Windows workstation with Docker Desktop running and
37 containers present. Metric: **wall-clock seconds per probe invocation**, median of five, `n=5`
per arm. Alternating rather than five-then-five because Docker Desktop's own caching would
otherwise be confounded with the arm order.

```bash
python scripts/bench_docker_probe.py
```

State agreement is checked separately and is a yes/no, not a timing: each probe is run three times,
once as-is (healthy), once with `DOCKER_HOST=tcp://127.0.0.1:1` (daemon not answering), and once
with `PATH` stripped of docker (absent). n=1 per state per probe, because the states are
deterministic rather than noisy.

## What I already know

- One observation of the current probe at **34.77s**, timed today while building `recall doctor`.
  It is the single largest fixed cost in that command and in `recall quickstart`.
- The existing 60 second `timeout=` in `docker_unavailable_reason` is therefore not a generous
  backstop on this machine; it is roughly 1.7x the observed cost, which means a slightly busier
  daemon would trip it and report a healthy Docker as "did not respond".
- Nothing in this project's memory store records a docker probe latency. The three docker memos
  ([[docker-population-is-desktop-stacks-not-session-dbs]],
  [[docker-ps-filter-cannot-see-compose-containers]], [[session-db-orphans-false-clean]]) are about
  what `docker ps` can and cannot SEE, not about how long anything takes.

## Confounds I can name now

- **Docker Desktop warm-up.** The first invocation after an idle period is slower than the rest.
  Alternating the arms distributes this rather than removing it, and the median rather than the mean
  is reported for the same reason.
- **This machine is not idle**, and is not a typical user's. 37 containers is an artefact of a repo
  that starts one per checkout, and `docker info`'s cost is expected to scale with exactly that.
  **So the speedup measured here is an upper bound on what a new user sees**, and the honest claim
  from this record is about the mechanism and its direction, not about a number a stranger would
  reproduce. A machine with two containers may show no meaningful difference, and that would not
  falsify the mechanism.
- **`docker version` succeeds against a CLI with no engine.** Named above as a falsifier rather than
  a confound, but it is worth stating twice: the two commands do not ask exactly the same question,
  and cheaper is only better if the answer is still the one the caller needs.
