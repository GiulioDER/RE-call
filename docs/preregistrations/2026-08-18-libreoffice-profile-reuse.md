# Pre registration: does reusing one LibreOffice user profile across extractions pay, and is it safe?

**Date:** 2026-08-18   **Status:** predicted, not yet measured

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
