# Pre-registration: moving the off-topic subject pool out of Python source

**Date:** 2026-08-18   **Status:** predicted, not yet measured

## The question

Does storing the off-topic subject pool as data rather than as Python string literals restore a
usable gap class for a code corpus that contains recall's own tree, and does the `code` tenant then
certify?

Two numbers, because they can come apart:

1. **Survivors.** How many of the 25 off-topic subjects are disjoint from `recall/**/*.py` after the
   move, and is `survivors × 5` at or above the default `per_class` of 40?
2. **Certification.** Does a `code` generation over `recall/**/*.py` with the `fastembed` embedder
   produce a calibration whose AUC lower bound clears the 0.9 bar, i.e. `certified is True`?

## What I predict

**Survivors: 12 of 25, capacity 60.** Exactly the arm-B figure below, because the only file
contributing the subject vocabulary is the one being changed. I am confident to roughly ±1 subject;
anything below 8 survivors (capacity 40) would leave the default `per_class` unreachable and is the
outcome that matters.

**Certification: yes, and comfortably.** I predict an AUC lower bound of **0.95 to 1.00**, higher
than the 0.9496 the offline generator scored on `docs/`. The reason is that a code corpus and a pool
about penguins, sourdough and gamelan tuning share almost no vocabulary, and separability is
precisely what that disjointness buys. I expect the gap class to be *easier* here than on prose.

**The part I am least sure of is the answerable half, not the gap half.** `_subject_of` builds a
question from a markdown heading sharpened by distinctive terms, and Python chunks have no markdown
headings, so subjects will fall back to bare term lists. If those retrieve their own chunk poorly,
the answerable class degrades and AUC falls even though the gap class is clean. I put this at maybe
30% likely, and if it happens I expect it to show as a *low* AUC with a healthy subject count, not
as a refusal.

## What would falsify this

- Survivors below 8 after the move, i.e. capacity under 40. That would mean the self-reference was
  not the whole cause and something else in `recall/` overlaps the pool.
- `certified is False` for the code tenant with `fastembed`, on a corpus large enough to clear the
  floor. Specifically an AUC lower bound below 0.9.
- `generate_offline` refusing on the answerable side (too few distinct subjects from code chunks)
  rather than on the gap side. That falsifies my claim that the pool was the binding constraint for
  this corpus, and would move the real fix to `_subject_of`.

## How it will be measured

Metric names, stated because a rate is named by its denominator:

- **Survivor count**, over the 25 shipped subjects: `len(offtopic_subjects_absent_from(chunks))`
  where `chunks = chunks_from_directory(recall/, "**/*.py", chunk_code)`.
- **Capacity**, `survivors × len(_OFFTOPIC_TEMPLATES)`, against `per_class` (default 40, floor 20).
- **AUC lower bound** and the boolean `certified`, read off the `CalibrationArtifact` returned by
  `CalibrationRepository.calibrate`, over a real generation built from `recall/**/*.py` with
  `fastembed` against a session container. n is the generated query set: `per_class` answerable and
  `per_class` unanswerable, reported explicitly.

Both arms run on this branch off `f51cec0a`.

## What I already know

- `docs/preregistrations/2026-08-16-generated-calibration-query-sets.md` measured **13 of 25**
  subjects disjoint from `docs/`, giving 65 gap questions, and records that **the pool's size, not
  the corpus, bounds how large a generated set can be**. It names widening the pool as the obvious
  lever. I am deliberately not widening it: the diagnosis below says the pool is not too small.
- The offline generator scored AUC lower bound **0.9496** on `docs/`, the LLM generator 1.0000. Both
  certified. So a generated set can certify at all, on prose.
- Measured today, before predicting, and the reason for this record (three arms, same filter):

  | corpus | chunks | survivors | capacity |
  |---|---|---|---|
  | `recall/**/*.py` as the wizard globs it | 1154 | **0/25** | 0 |
  | `recall/**/*.py` minus `eval/synthetic.py` | 1149 | 12/25 | 60 |
  | `pip/**/*.py`, a stand-in for a user's repo | 2071 | 11/25 | 55 |

  The only file in `recall/` containing `waggle`, `sourdough` or `harpsichord` is
  `recall/eval/synthetic.py`, which defines `_OFFTOPIC_SUBJECTS`. So the pool disqualifies itself by
  being source code inside the corpus under test.
- ⚠️ **A claim of mine this replaces.** I told the user the `code` tenant "cannot certify, ever" and
  that "on any real repository `recall wizard` degrades `code` every run". The `pip` arm shows that
  is false: a user's repository is unaffected. The failure is specific to corpora containing
  recall's own source, which is the dogfood case.

## Confounds I can name now

- **The `pip` arm is a stand-in, not a sample.** One third-party package does not establish that
  user repositories generally survive. It establishes that at least one does, which is enough to
  falsify "any real repository", and no more.
- **`recall/` will still contain the *filenames* and identifiers of the eval module.** Moving the
  literals to a data file removes the subject words, but if any docstring or test still quotes a
  subject, survivors will land below prediction. That is a real possible cause of a miss and I
  should look for it in the result rather than assume the move was complete.
- **A data file inside the package is still readable by a corpus globbing `**/*`.** The wizard globs
  `**/*.py` for code and `**/*.md` for docs, so this fix holds for the wizard's own corpora and does
  not hold for an arbitrary glob. The prediction is about the wizard's globs only.
- **Certification depends on the embedder.** `hashing` scored 0.757 on `docs/` and degraded; the
  prediction above is for `fastembed`. Measuring with `hashing` and reporting it as the answer would
  be answering a different question well.
- **Chunk count is not question count.** 1154 code chunks does not mean 40 usable answerable
  subjects, because `generate_offline` needs *distinct* subjects. If the corpus yields fewer, the
  refusal is on the answerable side and my prediction is wrong for a reason unrelated to the pool.
