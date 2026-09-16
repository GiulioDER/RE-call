# Dense-preserving atomic rescue is promising on the blind exploratory set

Measured 2026-09-16 under
[`2026-09-16-atomic-fact-dense-preserving-rescue.md`](../preregistrations/2026-09-16-atomic-fact-dense-preserving-rescue.md).

## Verdict

`PROMISING_EXPLORATORY_ATOMIC_RESCUE`.

At the same six-parent candidate budget, preserving dense top five and appending the first distinct
atomic parent reached 28 of 31 exact gold parents and 28 of 31 gold sources. Dense top six reached
24 of 31 on both measures. The paired result is four gains, zero losses, and 27 ties for exact
parent and gold source reach.

This is the first source-disjoint evidence that the atomic representation can improve current
production retrieval. The useful deployment shape is a rescue candidate after dense top five,
not pure replacement and not score fusion.

## Quality result

| Arm | Exact @1 | Exact @3 | Exact @5 | Exact @6 | Gold @1 | Gold @3 | Gold @5 | Gold @6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense | 18 | 22 | 24 | 24 | 19 | 22 | 24 | 24 |
| Pure atomic | 26 | 28 | 29 | not measured | 26 | 28 | 29 | not measured |
| Dense top five plus atomic one | inherited | inherited | 24 | 28 | inherited | inherited | 24 | 28 |

Dense rank six recovered none of the seven exact or source misses left after dense top five. The
single atomic slot recovered four of those seven. Atomic top 20 contained five of the seven, so one
additional rescue exists beyond the first atomic slot, but selecting or allocating another slot is
untested and must not be tuned on these rows.

The appended atomic candidate differed from dense rank six on 28 of 31 rows. This is therefore a
genuinely complementary representation, not a duplicate depth control.

## Integrity and execution

The run used production generation `gen_55487101e0d2421594b31f5c9519070f`, certified calibration
`cal_d86150126d3f46c8aefa910c3a0529a6`, and pool SHA256
`97f77c71c1feb278b9d5297511e8b4fb913002208eaae2b3bbd2dc65d578fd18`.

It built 6,313 atomic views over 1,413 sources and 11,370 ordinary chunks. All 31 pool rows passed
source, answer span, and gold parent validation. Candidate scores were finite, vector counts
matched, and the active production generation remained unchanged. The shared embedding lock was
released after the run, and the production manifest check still reported the frozen generation as
active with an unchanged corpus.

View construction took 7.783 seconds, document embedding 28.252 seconds, query embedding 7.469
seconds, and combined dense plus atomic retrieval 9.483 seconds for 31 rows. These are whole-stage
diagnostics, not a serving latency claim. The prior exact-matrix p95 failure still stands.

## Evidence boundary and next step

The result is exploratory because the blind census accepted 31 questions, one below its
preregistered construction gate, and the questions were independently model-written rather than
observed user traffic. It cannot authorize production serving.

The next highest-return step is an off-by-default atomic rescue shadow with an exact or approximate
top-candidate index that clears the existing latency gates. It should preserve dense top five and
record only whether the atomic candidate would rescue a miss. In parallel, accumulate a larger
future source or real-query confirmation set. Do not tune a trigger, second slot, fusion weight, or
threshold on these 31 rows.

The machine-readable aggregate is
[`2026-09-16-atomic-fact-dense-preserving-rescue.json`](2026-09-16-atomic-fact-dense-preserving-rescue.json).
