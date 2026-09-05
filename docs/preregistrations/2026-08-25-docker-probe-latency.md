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

## Result (2026-08-25)

**Status:** measured

```
info     median   14.11s  (n=5, all: 20.24, 14.45, 7.46, 4.01, 14.11)
version  median    0.48s  (n=5, all: 0.48, 0.47, 0.35, 3.53, 0.63)
speedup: 29.1x

healthy       info: rc=0 (56.36s)  | version: rc=0 (0.53s)
daemon dead   info: rc=1 (6.53s)   | version: rc=1 (0.21s)
docker absent shutil.which -> None (shared by both probes; not an arm comparison)
```

**The speedup prediction held. The baseline prediction did not, and that is the interesting half.**

| | Predicted | Measured | Gap |
|---|---|---|---|
| `docker info` median | 30 to 40s | **14.11s** | over-predicted by 2.1x to 2.8x |
| `docker version` median | 1 to 3s | **0.48s** | faster than the floor I predicted |
| speedup | 12x to 35x | **29.1x** | inside the band |
| agreement on healthy and dead | identical | identical (rc=0 / rc=1) | none |

Nothing here is falsified: the falsifiers were "under 5x" and "the two disagree on any state", and
neither occurred. Both probes also emit an equally actionable error on a dead daemon
(`error during connect: ... connectex: No connection could be made ...`), so the
`Docker said: {detail}` line in `docker_unavailable_reason` keeps its content.

**Two things I got wrong, recorded because they are worth more than the number.**

1. **I anchored the baseline on a single observation and predicted a range around it.** The 34.77s
   I had measured that morning is real, and it is near the TOP of this distribution rather than in
   the middle: the five timed runs span **4.01s to 20.24s**, and the untimed healthy run in the
   agreement pass hit **56.36s**. So the full observed spread of one command on one idle-ish machine
   in one hour is **4s to 56s, a factor of 14**. A prediction built on n=1 was a prediction about
   the sample, not the population, and this is the third time
   [[i-over-predict-effect-magnitudes]] has described my error correctly in advance.

2. **The variance, not the median, is the reason to make this change.** I justified the swap on
   mean cost and the justification is weaker than the real one. `docker_unavailable_reason` carries
   `timeout=60`; against a probe whose observed maximum is 56.36s, that backstop is **1.07x the
   worst case seen today**, so a marginally busier daemon makes the quickstart report a perfectly
   healthy Docker as "installed but did not respond" and send the user to fix something that is not
   broken. That failure was reachable before this measurement and I had not noticed it. `docker
   version` at 0.48s puts roughly two orders of magnitude between the probe and its timeout.

**Confound that held as predicted:** this machine had 37 containers when measured, which is an
artefact of a repository that starts one per checkout, and `docker info`'s cost scales with exactly
that inventory. The 29.1x is an upper bound on what a new user with two containers would see. The
claim carried forward is therefore the mechanism and its direction, plus the timeout-margin
argument above, which does not depend on the multiple.

Re-measure:

```bash
python scripts/bench_docker_probe.py
```

## Correction, appended 2026-08-26 (nothing above this line is edited)

A CCA audit of the commit that carried this record found three defects in it. All are corrected by
appending, because a pre-registration is evidence of what was believed when it was written and a
record that gets silently corrected cannot show that.

**1. The Result claims two falsifiers where the prediction lists three.** Line 111 says "the
falsifiers were 'under 5x' and 'the two disagree on any state', and neither occurred". The
prediction lists a third: *`docker version` succeeds while the daemon cannot actually run a
container (for example, a reachable CLI proxy with no engine behind it)*, restated in the confounds
section as "worth stating twice". **It was never measured.** `scripts/bench_docker_probe.py`
exercises healthy and dead-daemon only, and resolves the docker-absent state through `shutil.which`
above the branch either probe would reach — sound reasoning that never made it into this record.

So the conclusion "Nothing here is falsified" is unestablished for one third of its own criteria.
**Untested is not un-occurred.** The third falsifier remains OPEN: there is no apparatus here for a
CLI proxy with no engine behind it, and a probe that answers a weaker question than the caller asks
would not have been caught by anything measured on 2026-08-25.

**2. `1.07x` is a rounding slip, and the CHANGELOG inverted it.** 60 / 56.36 = 1.0646, so the bound
was 1.06x the worst case. The prose in this record has the direction right ("that backstop is 1.07x
the worst case"); `CHANGELOG.md` had it backwards ("that worst case was 1.07x the bound", which
asserts the probe already always timed out). The CHANGELOG is not frozen and has been corrected;
the number above is left exactly as committed.

**3. The margin is stated against the median where the argument is about the tail.** This record's
own conclusion is that "the variance, not the median, is the reason to make this change", and then
measures the new probe's safety at 20 / 0.48 = 41.67x, which is 1.62 orders of magnitude rather
than the "roughly two" claimed in `recall/quickstart.py`. Against the slowest `docker version`
sample in this record's own five-run set (3.53s) the real margin is **5.67x**. That is still a large
improvement on the old probe's 1.06x, and it is the honest number. Measuring the replacement by its
median while faulting the original by its maximum re-imports the mistake the change was made to fix.
