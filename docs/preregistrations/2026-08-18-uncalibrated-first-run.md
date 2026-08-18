# Pre registration: serving a corpus before it has a calibration

**Date:** 2026-08-18   **Status:** predicted, not yet measured

> ⚠️ **Private infrastructure identifiers were replaced with neutral descriptions before
> this file was first published, because this repository is public.** No prediction, no falsifier,
> no threshold and no number is affected. Where the original named a specific machine, the text
> says "a remote read only Postgres instance", which is what the measurement actually depends on.

## The question

Two questions, both raised by the redesign in `docs/UNCALIBRATED_FIRST_RUN_DESIGN.md`.

**Q1 (adoption fidelity).** Can a corpus already in the legacy `chunks` table be adopted into a
generation without re embedding, such that the adopted generation's vectors are provably the
vectors that pipeline produced from those bytes?

**Q2 (provisional threshold quality).** Is a threshold fitted on a machine generated query set
(`recall/wizard/queryset.py`) better than the uncertified 0.5 demonstration constant that
`RECALL_TRUST_MODE=development` uses today?

## What I predict

**Q1.** At least 95 percent of sources in the `memory` tenant of a remote read only Postgres corpus
will adopt with a verified binding, meaning `metadata->>'content_hash'` equals a fresh sha256 of the
file on disk AND `metadata->>'embedding_profile'` equals the profile id of the configured embedder.
The remaining failures will be sources edited since indexing, not sources missing the metadata keys.
A re embedding attestation sample of 20 chunks will reproduce the stored vector to within cosine
0.9999. Adoption of 8,671 chunks will complete in under 5 minutes, against the ~10 hours a full
`generation build` costs on the same corpus.

**Q2.** The provisional threshold will beat 0.5 on false confidence rate by at least 0.10 absolute,
measured on a held out human labelled set. It will NOT reach the separability bar that
`Calibration.certified` requires, which is why the design gives it a distinct status rather than
calling it certified.

## What would falsify this

**Q1** is falsified if under 95 percent of sources verify, if the failures are dominated by absent
metadata rather than by edited files, if any attestation sample chunk reproduces below cosine
0.9999, or if adoption takes over 5 minutes. Any of those means the legacy table cannot establish
the binding and the adoption path must be dropped rather than weakened.

**Q2** is falsified if the provisional threshold's false confidence rate is within 0.10 of 0.5's, or
worse than it. That would mean a generated query set buys nothing over the constant, and the design
should keep the constant and be honest about it instead of adding a status for it.

⚠️ Q2 is the one that could talk me into shipping something dishonest, so the falsifier is stated
against a **human labelled** held out set specifically. Scoring a generated set against itself would
measure only that the fitting worked.

## How it will be measured

**Q1.** Against the `memory` tenant of a remote read only Postgres instance, 1,080 files and
8,671 chunks. For every distinct `source` in `chunks`: read the file, sha256 it, compare to
`metadata->>'content_hash'`. Record the four way split (verified / file changed / file missing /
metadata absent). Then draw 20 chunks uniformly at random, re embed their `text` with the configured
embedder, and record cosine against the stored `embedding`. Sample drawn with a fixed seed passed in
rather than generated, and the full population enumerated before sampling, per the sorted sample
plus early stop lesson. Time the adoption transaction with the DB clock, not wall clock in Python.

**Q2.** Build one generation over the `memory` tenant. Generate a query set with
`recall/wizard/queryset.py` at `DEFAULT_PER_CLASS`. Fit a threshold on it. Separately, hold out the
existing human labelled set. Score three arms on the held out set: the 0.5 constant, the provisional
threshold, and (as a ceiling) a threshold fitted on the held out set itself. Primary metric is false
confidence rate, matching `recall/cli.py`'s `fcr_at_050` reporting. Report the Hanley McNeil
interval, not just the point.

## What I already know

- Calibration binds to a `ready` generation, not only an `active` one
  (`recall/calibration_v2.py:349`), and `pin_generation` accepts the same three states
  (`recall/generation_store.py:146`). So neither calibration nor serving actually requires promotion.
- The legacy `chunks` table has no `source_sha256` column (`recall/migrations/sql/0001_v08_baseline.sql`),
  but chunk metadata carries `content_hash`, `index_fingerprint` and `embedding_profile`
  (`recall/index.py:707`), all written at embed time.
- `recall/eval/synthetic.py:69-75` already measured that perturbation derived unanswerable queries
  are not separable. `queryset.py` builds them off topic instead, which is why Q2 is worth asking at
  all rather than being obviously false.

## Confounds I can name now

The `memory` tenant is small and is my own writing, so its vocabulary is unusually consistent and Q2
may flatter the generated set. `content_hash` proves which bytes were read, not which chunker ran,
which is why the attestation sample exists as a separate check rather than being folded into the
verification count. Voyage query embeddings are not deterministic, so Q2 must use fastembed. The
attestation cosine bar of 0.9999 assumes ONNX determinism on one host; a different host is a
different measurement and would need its own record.

