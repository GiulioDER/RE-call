# Pre registration: does reusing one LibreOffice user profile across extractions pay, and is it safe?

**Date:** 2026-08-18   **Status:** measured 2026-08-18. Q1 confirmed inside its band, Q4 clean,
Q2 confirmed in substance and wrong in shape, **Q3 falsified outright**. The design changed as a
result, from the single lock this record pre-committed to a small pool, which is what the Q3
falsifier said to do. Predictions and falsifiers below are unedited; the result is appended below
the horizontal rule.

## The question

`_extract_with_libreoffice` (`recall/extraction.py:573`) builds a fresh `tempfile.TemporaryDirectory`
and a brand new `-env:UserInstallation` profile inside it on **every** call, so every `.doc`, `.odt`,
`.ods`, `.ppt` and `.odp` extraction pays a cold LibreOffice profile bootstrap.

**Q1 (the prize).** If one profile directory is reused across calls within a process, by how much
does the wall clock cost of five consecutive `extract_document` calls (one per LibreOffice format)
fall?

**Q2 (the blocker).** What happens when two `soffice --convert-to` invocations that share one
`-env:UserInstallation` run concurrently: do both succeed, does one fail loudly, or does one exit 0
having converted nothing?

**Q3 (the cost of the fix).** If concurrency is answered with a per process lock, is throughput at
four way concurrency worse than today's unlocked but always cold behaviour?

**Q4 (apparatus).** Does the shared profile arm extract the **same text** as the per call arm for all
five formats?

## What I predict

**Q1. Five extractions fall to roughly 55 to 70 percent of the baseline I measure: a 1.3x to 1.9x
speedup, not the ~3x the fixture series suggests.** In absolute terms, if the baseline lands near
56s I expect the shared arm at **30 to 45s, point estimate 38s**.

This is deliberately more pessimistic than the framing that motivated the task, and the reason is in
the existing numbers rather than in caution. The commit message on `96e5aadb` reports two interleaved
series from one run, fixture conversion against extraction: 21.52/17.06, 10.70/12.84, 4.18/9.70,
7.57/8.62, 4.21/7.78. The fixture series shares one profile and declines 21.52 to 4.21. The
extraction series shares **nothing**, every call builds a fresh profile, and it still declines 17.06
to 7.78. A series that declines by better than half with no profile reuse at all is direct evidence
that most of that decline is the operating system page cache warming `soffice.exe` and its DLLs,
not profile bootstrap. So the profile reuse prize is only the residual gap between the two
asymptotes, about 4 to 7s warm against about 7.8 to 9.7s warm, call it **3s per call over four warm
calls, near 12s**, plus whatever the first call cannot avoid.

**Q2. The two concurrent conversions do not both succeed. I predict the second exits 0 and writes no
output file**, because LibreOffice keys a per profile named pipe off the user installation path and
a second invocation hands its request to the already running instance rather than doing the work
itself. Confidence: about 85 percent on "does not cleanly both succeed", about 60 percent on that
specific exit 0 with no output shape rather than an explicit lock error. Under today's code that
shape surfaces as `DocumentExtractionError("LibreOffice produced no text for ...")`, so it is a
spurious failure rather than silent corruption, which is the good half of a bad outcome.

**Q3. Serialising is not a throughput regression.** Four concurrent extractions under a lock with a
warm shared profile beat four concurrent extractions each paying a cold profile, because the cold
arm four bootstraps contend for the same cores. Point estimate: **lock plus warm near 28s against
concurrent plus cold near 45s.** I also predict this question is close to academic for this
repository, because both in tree call sites are serial loops (`recall/index.py:796` inside a `for`,
`recall/manifest.py:365` inside a generator expression), so there is no concurrency to lose today.

**Q4. Byte identical text in both arms, 5 of 5 formats.** The profile changes where LibreOffice keeps
its configuration, not what its import and export filters produce. This is the apparatus check, not
a finding: if it fails, the Q1 number is meaningless regardless of how good it looks.

## What would falsify this

- **Q1** is falsified downward by a shared arm total at or above **90 percent of baseline**, which
  would mean profile bootstrap is not the cost and this whole idea should be abandoned rather than
  tuned. It is falsified upward by a shared arm at or below **45 percent of baseline**, which would
  mean my page cache explanation of the declining series is wrong and profile bootstrap really is
  the dominant term.
- **Q2** is falsified by both concurrent conversions succeeding with correct output. That would be
  the best outcome available: no lock, no pool, the change collapses to a cached directory and
  nothing else.
- **Q3** is falsified by the locked arm being slower than the cold concurrent arm at four way
  concurrency. That sends the design to a pool of K profiles rather than a single lock.
- **Q4** is falsified by any format differing between arms.

## How it will be measured

`scripts/bench_libreoffice_profile.py`, written for this record, on this machine.

- **n = 5 extractions per repetition**, one per LibreOffice format, built from `docx`, `xlsx` and
  `pptx` fixtures converted once up front with a throwaway profile so the timed section measures
  extraction only.
- **3 repetitions per arm, arms alternated** (baseline, shared, baseline, shared, baseline, shared),
  each repetition in a **fresh subprocess** so no in process state carries over. Reported statistic
  is the **median of the three per arm totals**, plus the full per call series for each repetition,
  because the series shape is what distinguishes profile warmth from page cache warmth.
- **Metric names.** `total_extraction_seconds` is the sum of the five `extract_document` calls,
  excluding fixture setup. `per_call_seconds` is the same five, in order, unsummed. The rate that
  matters is **seconds per LibreOffice backed extraction**, denominator five.
- **Q2** is measured directly, outside pytest, by launching two `soffice --convert-to` processes
  against one shared `-env:UserInstallation` at the same moment and recording both exit codes, both
  stderr strings, and whether each output file exists.
- **Q3** is measured with four threads each calling `extract_document`, timed wall clock, once
  against the locked shared profile build and once against the current per call build.
- **Q4** compares the extracted text of both arms with `==` per format.

Baseline is **re measured in this same harness**, never compared against the 56.00s quoted in
`96e5aadb`. See the confounds.

## What I already know

- `96e5aadb` ("Give the LibreOffice test the timeout margin its ten soffice startups need", on the
  unmerged branch `claude/distracted-gould-c57ce9`, **not** on master) records the two series quoted
  above and explicitly parks this change: "a shared profile there needs an answer for concurrent
  extractions locking it, so it is not a change to make from a flaky test". This record is that
  answer.
- STOP. **The per call profile is not an accident, it is a fix.**
  `docs/preregistrations/2026-08-17-ingestion-verification.md:83` records that "a unique temporary
  LibreOffice profile is now used for every conversion, **preventing profile-lock and stale-process
  interference**". So the change proposed here partially reverts a deliberate remedy, and Q2 is the
  question of whether that remedy is still needed. If Q2 confirms the collision, the lock is not
  optional overhead, it is the replacement for the mechanism being removed.
- `docs/preregistrations/2026-08-18-extraction-attestation.md` measured extraction byte identical
  across processes for 17 of 17 formats, including all five LibreOffice ones, which is why Q4 is
  cheap to check and expected to pass.
- Memory: `extraction-depends-on-an-unpinnable-binary.md` (no lockfile pins this binary, and
  `soffice --version` prints nothing on Windows, so the version must be read from the file
  `VersionInfo`).

## The design this is testing

Pre committed now so the result cannot be fitted to whatever the numbers turn out to like:

1. A module level profile directory, created lazily with `tempfile.mkdtemp` and removed at
   interpreter exit. **Per process, not a fixed path under the temp directory**, so two concurrent
   `recall index` processes cannot collide by construction; the price is one cold bootstrap per
   process rather than one per machine.
2. A module level `threading.Lock` around the `soffice` invocation, guarding the shared profile
   within the process.
3. The per call `TemporaryDirectory` is **kept** for the source file and the output directory, since
   two calls must not race on the converted output name. Only the profile becomes shared.
4. On any conversion failure the shared profile is discarded so the next call rebuilds it, which is
   the recovery path for a stale lock left by a killed `soffice`.

## Confounds I can name now

- **Machine load dominates this test.** `96e5aadb` measured the same test at 53.86, 53.93, 57.26,
  75.64 and 116.17 seconds on one machine on one day, a 2.15x spread. Alternating arms and taking
  medians is the mitigation; it is not a cure, and a difference smaller than about 15 percent should
  not be believed from three repetitions.
- **Page cache ordering is the confound that would manufacture the predicted result.** Whichever arm
  runs second inherits a warm `soffice.exe`. Alternating the arms is the mitigation, and the per call
  series is reported so a reader can see whether the shared arm advantage sits in call 1 (page cache)
  or in calls 2 to 5 (profile reuse), which are different claims.
- **The 56.00s figure is not a usable baseline and is not being used as one.** It was measured
  inside the test, where every extraction is immediately preceded by a fixture conversion that warms
  the binary. A standalone benchmark has no such warming, so my measured baseline may legitimately
  come out **higher** than 56.00s without anything being wrong.
- **Version discrepancy, recorded before measuring.** The task brief and `96e5aadb` both say
  LibreOffice **25.8**. This machine reports **26.2.5.2**, and
  `docs/preregistrations/2026-08-18-extraction-attestation.md:113` also says 26.2.5.2 on the same
  day. One of those labels is wrong. I do not know which, I am not editing either, and I am
  recording the version I measure.
- **Windows only.** Profile locking, the named pipe behaviour behind Q2, and process startup cost are
  all platform specific. Nothing here transfers to Linux CI without re measurement.
- **Q3 four way concurrency is synthetic.** No caller in this repository extracts concurrently, so
  a Q3 win or loss is about a future caller, not a current one.

---

## Result (2026-08-18)

**Status:** measured

Windows 11, LibreOffice **26.2.5.2** (the version question in the confounds is settled below), Python
3.13, 12 logical cores, idle machine with no other `soffice.exe` running at the start. Harness:
`scripts/bench_libreoffice_profile.py`. Fixtures built once up front, so no timed section includes a
fixture conversion.

### Q1: confirmed, inside the predicted band

Five serial extractions, three repetitions per arm, arms alternated, each repetition a fresh process.
Medians of the three `total_extraction_seconds`:

| Arm | repetitions | median | share of baseline |
|---|---|---|---|
| per call profile (today) | 30.21, 29.33, 29.90 | **29.90s** | 1.00 |
| pooled profile (landed) | 17.40, 17.52, 17.17 | **17.40s** | **0.582** |

Predicted: 55 to 70 percent of baseline, 1.3x to 1.9x. Measured: **58.2 percent, 1.72x.** Inside the
band, nearer its good end. The earlier single lock build measured on the same day gave 17.94s against
a 29.76s baseline, 60.3 percent, so the pool is if anything marginally better than the lock it
replaced, not a compromise against it.

**Gap: none worth reporting on the ratio. The gap that matters is on the absolute numbers, and it is
large.** I predicted "if the baseline lands near 56s". It did not. It landed at **29.90s**, about
53 percent of the 56.00s quoted in `96e5aadb`. The confound note predicted this figure would be
unusable and it was right about that; it guessed the standalone baseline would come out **higher**
than 56.00s for lack of warming, and the truth was the opposite. The 56.00s was inflated by the
contention of the interleaved fixture conversions around it, not deflated by their warmth.

### The mechanism, which is the part worth keeping

`per_call_seconds`, representative repetition:

```
per call profile   5.45  6.98  5.04  5.87  6.56     <- flat
pooled profile     5.42  3.31  2.33  2.55  3.56     <- first call pays, the rest do not
```

**The baseline series is flat.** That is the direct confirmation of the reasoning the prediction was
built on. `96e5aadb` recorded a no-sharing extraction series declining 17.06 to 7.78 and this record
argued that decline could not be profile reuse, because there was no profile reuse in it. On an idle
machine the same no-sharing arm does not decline at all, so that earlier decline was machine warm up
and load. Profile bootstrap is worth about **2.9s per warm call** (about 6.0s down to about 3.1s),
which is close to the "call it 3s per call" written before the run.

### Q2: confirmed in substance, wrong in shape

Two `soffice --convert-to` processes started at the same moment against one shared profile.

| Profile | process A | process B | both succeeded |
|---|---|---|---|
| cold | **rc=1**, no output, **stderr empty** | rc=0, output written | no |
| warm (6.75s bootstrap first) | rc=0, output written | **rc=1**, no output, **stderr empty** | no |

Predicted "does not cleanly both succeed" at about 85 percent confidence: **correct.** Predicted the
loser would **exit 0** having converted nothing, at about 60 percent: **wrong, it exits 1.** Which
way that error runs matters, and it runs the safe way. `subprocess.run(check=True)` turns a non-zero
exit into `CalledProcessError`, so the collision surfaces as `DocumentExtractionError`, never as a
silently empty document. Had the exit code been 0 as predicted, the code would have fallen through to
the "produced no text" branch instead, which is still an error but a less direct one.

The empty stderr is the genuinely nasty part and was not predicted at all: there is **no diagnostic
whatsoever**, so a deployment meeting this would see an extraction fail with nothing to explain it.

### Q3: falsified

Four concurrent `extract_document` calls, wall clock.

| Arm | repetitions | median |
|---|---|---|
| per call profile (today, 4 truly parallel) | 9.25, 7.10 | **8.18s** |
| one shared profile behind a lock | 12.61, 13.16 | **12.89s** |

Predicted: the lock wins, about 28s against about 45s. Measured: **the lock loses by about 1.6x**,
and both arms are three to five times faster in absolute terms than I guessed. Zero errors in either
arm.

**Why the prediction was wrong.** I reasoned that four cold bootstraps would contend for the same
cores and cost more than serialising. On a 12 core machine they do not contend meaningfully: four
`soffice` processes genuinely run in parallel, so the cold arm pays about one bootstrap of wall clock
while the locked arm pays four conversions end to end. The bootstrap I was trying to save is roughly
3s; the serialisation I would have bought costs roughly 3 conversions, about 9s. I compared the cost
of the fix against the wrong thing.

The falsifier said this "sends the design to a pool of K profiles rather than a single lock", so it
did. Re-measured after the pool landed:

| Arm | repetitions | median |
|---|---|---|
| per call profile | 8.68, 19.56, 10.92 | **10.92s** |
| pool of 4 | 8.70, 9.29, 8.68 | **8.70s** |

The pool is no worse concurrently and much steadier (the 19.56s baseline outlier is four cold
bootstraps landing badly), while keeping the 1.72x serial win. Both wins, which the lock could not do.

⚠️ **What this concurrency measurement does not show.** Each of the four workers takes a pool slot for
the first time, so all four pay a bootstrap and the pool's actual advantage, reuse, is never
exercised. It establishes that the pool does not *lose* concurrently, not that it wins there. A
repeated concurrent workload would be the test for that and was not run.

### Q4: confirmed

**30 of 30** format and run pairs identical, five formats across all six serial repetitions in both
arms. The profile decides where LibreOffice keeps its configuration and nothing about its filters.

### Version, settled

`(Get-Item 'C:\Program Files\LibreOffice\program\soffice.exe').VersionInfo.ProductVersion` reports
**26.2.5.2**, agreeing with `2026-08-18-extraction-attestation.md:113` and disagreeing with both the
task brief and `96e5aadb`, which say 25.8. Neither of those has been edited. The likeliest reading is
that "25.8" was carried over from an older note rather than measured, but that is a guess and is
labelled as one.

### Scoreboard

| Question | Predicted | Measured | Verdict |
|---|---|---|---|
| Q1 serial cost | 55 to 70 percent of baseline | 58.2 percent, 1.72x | **confirmed** |
| Q1 absolute baseline | near 56s, possibly higher | 29.90s | **wrong, and in the direction not guessed** |
| Q2 collision | does not both succeed (85 percent) | does not both succeed | **confirmed** |
| Q2 failure shape | loser exits 0 (60 percent) | loser exits 1, stderr empty | **wrong, and safer than predicted** |
| Q3 lock throughput | lock wins, 28s against 45s | lock loses, 12.89s against 8.18s | **falsified, design changed** |
| Q4 text identity | 5 of 5 identical | 30 of 30 pairs identical | **confirmed** |

### What I would tell the next person

**The interesting result is Q3, and it is the one I would have skipped.** Q1 was the question the
task asked and the prediction was fine; the pre-registration earned its keep on the question that
was only in it because a falsifier had to be written for it. Had I measured Q1 alone, the single
lock would have landed on a 1.72x serial win and quietly made concurrent extraction 1.6x slower for
any future caller, with a measurement in the commit message proving the change was good.

Second: **a declining series is not evidence of the thing you are about to fix.** Both series in
`96e5aadb` declined, one of them shared a profile, and it was tempting to attribute both declines to
that. The flat baseline here shows the attribution was available for the cost of one idle machine.
