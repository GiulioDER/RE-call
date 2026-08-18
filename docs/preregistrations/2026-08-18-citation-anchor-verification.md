# Pre-registration: anchor-based verification of `path:line` citations in `docs/`

**Date:** 2026-08-18   **Status:** predicted, not yet measured

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
- `tests/test_findings_crossrefs.py` is the sibling gate: it pins `§9f`-style cross-references to a
  substring of the section title, because checking that §9f *exists* passes while it points at the
  wrong text. Same failure shape, same remedy.
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
