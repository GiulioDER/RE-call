# Pre-registration: does batching `docker inspect` make `session-db.sh orphans` faster?

**Date:** 2026-08-20   **Status:** predicted, not yet measured

## The question

`cmd_orphans` used to run one `docker inspect` per container. It now runs one call per 100
containers. Does that show up as wall-clock time on this machine, at the population this machine
actually holds, and by how much?

The measurement is not academic: `session-close.sh` runs this command on every close, and a
`--dry-run` close taken during this session ran past 120 seconds.

## Setup

17 containers on the daemon, of which 13 running (12 after `prereg-successor` removed its own
session database mid-session). The old implementation is at `a3d69c89`, the new one in the working
tree. Both are timed in the same shell, alternating, three runs each, with the same daemon and no
test suite running alongside.

```bash
git show a3d69c89:scripts/session-db.sh > /c/mut/old-session-db.sh
for i in 1 2 3; do
  /usr/bin/time -f 'old %e' bash /c/mut/old-session-db.sh orphans > /dev/null
  /usr/bin/time -f 'new %e' bash scripts/session-db.sh orphans > /dev/null
done
```

## What I predict

**Primary: the new version is faster by a factor between 3 and 8.** Confidence roughly 80%.

The arithmetic behind that: a bare `docker ps -q` on this machine took **2.455 s** when timed
during the session, which is a Docker Desktop round trip on Windows rather than anything about the
work. The old path is 2 `ps` calls plus 17 `inspect` calls, so 19 round trips; the new path is 3
`ps` calls (one is the daemon probe added earlier today) plus 1 `inspect`, so 4. If every call
costs about the same, that is 19/4, near 4.75x.

**Secondary: the new version comes in under 15 seconds.** Confidence roughly 70%.

**What would falsify the reasoning rather than the number:** if the old version also finishes in a
few seconds, then per-call cost is not constant and the 2.455 s figure was contention from the
mutation suites running at the time, not a floor. In that case the batch is still correct but its
justification was wrong, and I should say so rather than keep the speedup claim.

**What I am not predicting:** any change in output. The two versions must print the identical
report, and if they do not, the timing is irrelevant and the change is a bug.

## What would make me abandon the change

A wrong report. Specifically, a container named in one version's output and absent from the other's,
or any line whose verdict differs. Speed is worth nothing here: this command exists to stop a false
"no orphaned containers", and a faster wrong answer is the failure mode it was written against.

---

## Result, measured 2026-08-20

**Status:** measured. **The primary prediction is falsified.** Predicted 3x to 8x; measured
**1.57x** on the means, 1.73x on the best pair. Appended, not edited: the prediction above stands
as written.

```
containers total: 13   running: 8      (the machine changed under the measurement, see below)

old  24.161s   26.720s   22.149s       mean 24.34s
new  19.281s   14.409s   12.828s       mean 15.51s
```

**Secondary, "under 15 seconds": marginal.** Two of three runs came in under it (14.4s, 12.8s) and
the first did not (19.3s). Calling that a pass would be picking the runs that agree.

**Output identical on all three pairs**, which was the condition that mattered more than the
timing: both versions printed `session-db: no orphaned containers`.

### Why the arithmetic was wrong

The prediction assumed a constant per-call cost and counted calls: 19 down to 4. Two things broke
that.

**The population shrank while I worked.** The pre-registration says 17 containers; by measurement
time another session had removed five of its own, leaving 13. So the old path made 15 calls rather
than 19, and the predicted ratio was really 15/4, or 3.75x. Still well above what happened.

**A batched `inspect` over 13 ids is not the price of one.** Old, at 15 calls in 24.34s, averages
1.62s per call. New, at 4 calls in 15.51s, averages 3.88s. The fixed CLI spawn dominates a small
call and does not dominate a large one, so collapsing 13 inspects into one collapses the spawns but
not the work. That is the part the arithmetic could not see from the outside, and it is the honest
correction: **the speedup is real but roughly half of what call-counting predicts.**

One quarter of the new command's four calls is the daemon probe added in `924bd7f0`, so about 3.9s
of the 15.5s buys the refusal-instead-of-false-clean behaviour rather than any answer. Worth it,
and worth stating rather than folding into the speedup.

### What I am NOT claiming

That this makes `session-close.sh` quick. 15s is not 24s, and neither is fast. The remaining cost
is four Windows docker CLI round trips, and no amount of batching removes them.

Re-measure:

```bash
bash <scratch>/timeit.sh   # git show a3d69c89:scripts/session-db.sh vs the working tree, 3 pairs
```
