# Pre-registration: anchor-based verification of `path:line` citations in `docs/`

**Date:** 2026-08-18   **Status:** measured 2026-08-18, see the result at the bottom.
The predictions below are unchanged from the committed version and must stay that way.

## The question

Does checking a **distinctive anchor token** near a cited line, rather than checking that the line
exists, detect stale `path:line` citations in `docs/**/*.md` at a false-positive rate low enough to
gate CI on?

Answerable by four numbers, all stated below: the STALE count, the share of STALE findings that
survive manual inspection, the share of currently-OK citations that a planted displacement turns
STALE, and the verdict on one citation already known to be stale.

## What I predict

The design under test: a citation is OK when an anchor token drawn from the backticked prose
around it appears **within the innermost enclosing Python definition or top-level statement of the
cited line**, or within +/-3 lines. An anchor token must occur on at most 6 lines of the whole
cited file, so a common identifier cannot manufacture a match.

1. **The known-stale citation is caught.** `recall/embedding_registry.py:223` in
   `docs/preregistrations/2026-08-18-uncalibrated-first-run.md:213` is reported STALE. #375 moved
   `bge-small-symmetric-v1` from 223 to 228; line 223 is now a closing `),`. This is the one case
   whose answer I already know, and it is here to test the apparatus rather than the idea.
   **Predicted: STALE.**

2. **STALE count on master falls from 42 to roughly 12-25.** The 42 measured under a naive
   +/-3-line window included many citations that point into a function body while the anchor sits
   at the `def`. Predicted range is wide because I have inspected 8 of the 42, not all of them.

3. **At least 70% of the reported STALE findings survive manual inspection** as citations that
   genuinely no longer point at the code the sentence claims. I will inspect **every** one, so the
   denominator is the full STALE set, not a sample.

4. **Planted displacement is detected in at least 90% of cases.** For each citation currently
   verdicted OK, add 40 to its line number and re-verify. Predicted: >=90% become STALE. The
   denominator is the OK set (predicted 60-85 citations).

5. **An absent anchor reports UNVERIFIABLE, not OK.** A citation whose prose names an identifier
   that does not occur anywhere in the cited file must not pass. **Predicted: UNVERIFIABLE.**

6. **UNVERIFIABLE lands between 35 and 55**, out of ~149 citations. These are real: a citation with
   no quoted identifier beside it cannot be checked by any method that does not read the prose.

## What would falsify this

- Prediction 1 wrong (the known-stale citation reports OK): the apparatus does not measure what it
  claims and nothing else in this record means anything.
- Prediction 3 below 70%: the check cries wolf, and a gate that fails on correct citations gets
  deleted, which loses the coverage entirely. That is the recorded failure mode of a guard that
  blocks real work.
- Prediction 4 below 90%: the check has no power. Passing at a 40-line displacement means the
  anchor rule is accepting almost anything.
- STALE above ~35 with most of them false: same conclusion as 3.

## How it will be measured

```bash
python scripts/verify_citations.py --unverifiable-ceiling 9999          # counts, verdicts
python scripts/verify_citations.py --json                               # per-citation detail
python -m pytest tests/test_verify_citations.py -q                      # apparatus, incl. mutation
```

- **n = every citation in `docs/**/*.md`** matching
  `^(recall|recall_mcp|recall_interop|recall_consistency|tests|benchmarks|scripts)/...:line`.
  Measured at 149 on `cd697668`.
- **STALE rate** is over all citations. **True-positive rate** is over the STALE set only, by
  reading each cited line and deciding whether the sentence's claim is true of it.
- **Detection rate** is over the OK set, under a +40-line displacement of each citation.

## What I already know

- The existence-only version of this check reported "41 citations, 0 broken" while three of those
  41 pointed at a `try:`, a docstring's closing quotes, and the wrong error branch. **An existence
  check is not a correctness check**, and its 0 was the reason nobody looked.
- The naive +/-3-line window measured 42 STALE / 55 OK / 51 UNVERIFIABLE / 1 MISSING_FILE over 149.
  Of 8 inspected by hand: 3 true (`frontmatter.py:12` is an `import` line, `store.py:686` is a
  comment about `close()`, `embedding_registry.py:223` is a closing paren) and 4 false (the anchor
  was at the enclosing `def`), 1 false from a word-boundary rule that refused to match
  `generation_mode` against `self._generation_mode`.
- `tests/test_findings_crossrefs.py` is the sibling gate: it pins lettered section
  cross-references to a substring of the section title they were written against, because checking
  that the section *exists* passes while it points at the wrong text. Same failure shape, same
  remedy. (This sentence originally spelled a label out; see the disclosure in the result.)
- Memory: [guards-that-cannot-fail], [test-a-guards-allow-path], [mutate-the-fix-not-a-nearby-line].
  A guard is not shown to work by passing; it is shown to work by going red on a planted defect,
  and the mutation must hit the guard rather than an ornament beside it.

## Confounds I can name now

- **Tuning until green.** Three thresholds (window, occurrence cap, minimum token length) are all
  adjustable, and any STALE count can be reached by moving them. The window is therefore fixed by
  a stated principle (the enclosing definition, because a citation into a body is a claim about
  that function) and not by which value clears the most failures. Predictions 3 and 4 are the ones
  that constrain the choice, because they move in opposite directions.
- **A false OK from a common token.** The occurrence cap is the only thing preventing it. If it is
  too loose, prediction 4 fails, which is why 4 is the paired check for 3.
- **The archive document.** `docs/archive/ENTERPRISE_PROGRAM_STATUS.md` holds 30 of the citations
  and is by construction historical, so its citations are stale on purpose. If the true-positive
  rate is carried by that one file, the check is measuring archival status rather than drift. I
  will report the STALE breakdown with and without `docs/archive/`.
- **Fenced blocks.** Citations inside code fences are skipped, which reduces n. If that exclusion
  is doing the work, the counts are over a corpus chosen to look good. n with and without the
  exclusion will be reported.

---

## Result (2026-08-18)

**Status:** measured. **Four of six predictions held; prediction 4 was falsified badly, and it is
the one worth reading.**

| # | Prediction | Measured | |
|---|---|---|---|
| 1 | known-stale `embedding_registry.py` line 223 reported STALE | STALE, naming line 228 | held |
| 2 | STALE count 12-25 | **21** | held |
| 3 | >=70% of STALE findings true | **19 of 21, 90.5%** | held |
| 4 | >=90% of +40-line displacements detected | **40%** | **falsified** |
| 5 | absent anchor reports UNVERIFIABLE | 80 of 80 | held |
| 6 | UNVERIFIABLE between 35 and 55 | **52** | held |

Final state of the corpus, after fixing the 19 true positives (n rose from 150 to 152 because
this record's own result section cites code):
`152 citations: OK=87 STALE=0 UNVERIFIABLE=53 FROZEN=10 EXTERNAL=2 MISSING_FILE=0`.

### Prediction 4, which is the whole result

I predicted a 40-line displacement would be detected >=90% of the time. It is detected **40%** of
the time. The cause is not a bug: the check accepts an anchor anywhere in the cited line's
enclosing definition, and many definitions here are longer than 40 lines, so the displaced
citation is still inside the function whose name the prose quotes. The anchor really is in scope.
The check is answering its own question correctly; **my prediction was about a different question
than the one the design asks.**

Detection under a displacement that leaves the enclosing scope, which is what the design actually
claims to catch: **61 of 86, 70.9%**. An existence-only check scores **0%** on the same mutation.

The full sensitivity curve, over the 80 OK citations at the time of measurement:

| displacement | +1 | +2 | +3 | +5 | +8 | +13 | +20 | +40 | +80 | +200 |
|---|---|---|---|---|---|---|---|---|---|---|
| detected | 0% | 1% | 1% | 19% | 23% | 23% | 30% | 40% | 56% | 79% |

**What I did about it: nothing to the thresholds, deliberately.** A scope cap was the obvious
lever and it was measured rather than adopted: capping at 25 lines lifts +40 detection from 40% to
52% and costs 5 of 80 hand-verified correct citations. Cap 10 reaches 62% and costs 7. Every point
on that curve buys detection with false accusations of correct citations, and a gate that fails on
correct work gets deleted, which loses the coverage entirely. So the thresholds stayed where the
stated principle put them, and the CLAIM was corrected instead: this check finds **structural**
staleness -- a cited line that has become a closing paren, an `import`, a blank line, or part of a
different function -- and does not detect drift within one function. That limitation is now in the
tool's own docstring and in the test that would otherwise have asserted 100%.

### The apparatus found four defects in itself, which prediction 1 alone would not have

1. **An enclosing-class header counted as an anchor.** "Line 1776 is inside class `PgVectorStore`"
   is true, checkable, and nearly vacuous for a 2000-line class. It rescued 3 of 88 citations into
   OK, and one of those, line 686 of `recall/store.py`, was a citation to a comment about `close()` in a
   document claiming it showed `delete_sources()`. The rule meant to reduce false alarms was
   concealing a real defect. Removed; the defect is fixed at `recall/store.py:1806`.
2. **Foreign repositories were verified against ours.** `docs/their-harness-parity.md` cites
   line 417 of their `benchmarks/locomo/run.py` and line 690 of their
   `benchmarks/beam/run.py`, in **mem0's** tree. The first
   does not exist here and was loud. The second exists here as a completely unrelated file, so the
   check was quietly issuing verdicts about the wrong repository. Now needs an explicit
   `<!-- citations: external -->`.
3. **A too-common token could drive a false accusation.** `recall/control_plane.py:802` is
   literally `def cutover(...)` in a sentence about `cutover()`, but the name is common enough in
   that file to fail the distinctiveness cap, so the STALE verdict rested on an unrelated token
   from the same sentence. Fixed by letting a common token on the **cited line itself** withhold
   an accusation without being able to certify one. The first attempt scoped that to the whole
   enclosing scope and silently downgraded the motivating defect
   (`embedding_registry.py` line 223) from STALE to UNVERIFIABLE; the cited-line version keeps both
   right.
4. **A document could silently exempt itself by describing the opt-out.** The `external` marker
   was matched anywhere in the file. This record then described the marker in a sentence, and all
   8 of its citations were exempted in one edit -- including the deliberately stale example it
   exists to document. Nothing errored; the count simply moved from UNVERIFIABLE to EXTERNAL. A
   silent exemption is the same defect class as a stale citation, reached from the other side.
   The marker must now be alone on its own line and outside any code fence.

The pattern in all four is worth naming: **every one was a rule added to suppress a false alarm,
and three of the four suppressed a true one as well.** Nothing announced that. Each was found by
mutating the checker or by reading a verdict that had quietly become the wrong kind of right, and
none would have been found by prediction 1, which the tool passed throughout.

### One edit to the pre-measurement text, disclosed

The sentence above about the sibling gate originally spelled out a lettered section label. That
gate scans `docs/` for exactly such labels and requires every one to be registered against a
substring of its section title, so this record failed it -- by *naming* the format, not by citing
anything. It is the same self-reference hazard as defect 4, arriving from the neighbouring tool,
and the two were found within an hour of each other. The sentence was reworded. No prediction,
number or claim changed, and nothing below was written before the measurement.

### Confounds, as named beforehand

- **Tuning until green: real, and the record above is the defence.** The thresholds were chosen by
  the stated principle and the measured cost of moving them is published rather than the movement.
- **The archive.** `docs/archive/` holds 28 of 150 citations and 10 stale ones. The true-positive
  rate is **not** carried by it: excluding it entirely, 8 of 10 STALE findings were true, 80%.
  Archived documents are now reported as FROZEN and do not fail, because the remedy would be
  editing an archive until it agreed with the present.
- **Fenced blocks: null.** Excluding them drops **0** of 150 citations. The exclusion is not doing
  any work on this corpus, so no count above depends on it.

### Apparatus verification

Predicting an outcome does not reveal a broken harness. Eight mutations were applied to
`scripts/verify_citations.py` -- always return OK, remove the distinctiveness cap, widen the
window, widen the scope, match no documents, downgrade STALE, disable the ratchet, drop the word
boundary -- and **all eight turn `tests/test_verify_citations.py` red**. Three stale citations
planted in real documents each drive the CLI to exit 1.

One planted mutation deliberately did **not** fail, and that is recorded rather than removed:
shifting the `recall/store.py` citation from line 1806 to line 1810 stays inside `delete_sources` and reports OK. That is
prediction 4's gap in a single case, and it is what the corrected claim above describes.
