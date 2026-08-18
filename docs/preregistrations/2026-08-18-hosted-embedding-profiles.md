# Pre-registration: serving a corpus built with a hosted (API) embedding model

**Date:** 2026-08-18   **Status:** predicted, not yet measured

## The question

Can a corpus embedded by a hosted model (Voyage, OpenAI, Gemini) be served through
`recall.embedding_registry` the way a local fastembed corpus now is, and do the identity gates
give the right yes/no on each of five specific paths? Answerable as five booleans.

## What I predict

Against the implementation described below, before running the suite:

1. **P1 — identity survives `build()`.** `registered_profile('voyage-code-3-v1').build(api_key=...)`
   yields an embedder whose `.profile.profile_id` is `voyage-code-3-v1`, not `voyage:voyage-code-3`,
   and whose `artifact_digest` is not `legacy-unverified`. Currently False on both counts.
2. **P2 — the enterprise gate REFUSES a hosted profile.** `check_enterprise_readiness(...).ready is
   False` with a failure naming the unpinned artifact, driven end to end from a built hosted
   embedder rather than from a synthetic identity. A hosted provider can change weights behind a
   model name, so "ready" would be a claim recall cannot back.
3. **P3 — a declared/actual width disagreement raises at construction.** A profile declaring 1024
   against a client returning 512 raises `ValueError` from the constructor. Currently it builds
   cleanly and `check_enterprise_readiness` passes the dimension check, because the legacy
   fallback profile sets `dimension` FROM `embedder.dim` and the comparison is vacuous.
4. **P4 — the index fingerprint separates two widths of one hosted model.** `_index_fingerprint`
   for the same file differs between voyage-code-3 at 1024 and at 2048. Currently identical:
   measured today, both mint `profile_id='voyage:voyage-code-3'`, which is the exact defect #370
   fixed for fastembed and left live on the hosted path.
5. **P5 — a hosted profile stays SUBJECT to the `Indexer` context check.** Constructing an
   `Indexer` with a hosted profile whose `context_version` disagrees with the context policy
   raises. The reverted attempt made this exempt via a shared `is_unverified_artifact()`
   predicate; that exemption is wrong, because a registered hosted profile's context IS known.

I predict 5 of 5 hold, and that P4 is the one most likely to surprise, because it is the only
prediction whose current-state measurement I made myself rather than inheriting from the brief.

## What would falsify this

Any of the five measuring the opposite. Specifically damaging:

- P2 returning `ready=True` — that is the reverted attempt's decorative-feature failure recurring.
- P3 building cleanly — the declared width would then be documentation, not a checked claim.
- P5 raising nothing — a hosted corpus could then be indexed under a context mode its profile
  does not name, which is the aliasing the registry exists to prevent.
- A test that passes with the hosted branch of `build()` DELETED. Each assertion must be driven
  from a constructed embedder, never from a hand-made `EmbeddingProfile`.

## How it will be measured

```bash
python -m pytest tests/test_hosted_embedding_profiles.py -q          # P1-P5, stubbed clients
python -m pytest tests/test_enterprise_readiness.py tests/test_profile_identity_gates.py \
  tests/test_fallback_profile_id_distinctness.py tests/test_embedding_cache_identity.py -q
python -m ruff check .
```

n = 5 predictions, each a single asserted boolean. No API key is used: both hosted clients are
stubbed, since CI installs only the `dev` extra. The apparatus check required by rule 2 is the
deletion test above — each new test must fail with the feature removed, and I will verify that by
mutating `build()` rather than by asserting it.

## What I already know

- Measured today against master (42bbe818), not inherited: neither `VoyageEmbedder` nor
  `OpenAICompatEmbedder` accepts `identity=` or exposes `.profile`; `EmbeddingProfile` and
  `recall.lineage.EmbedderIdentity` are DIFFERENT classes, the former unvalidated and already
  carrying a `legacy-unverified` sentinel, the latter refusing any non-64-hex digest.
- Measured today via OpenRouter `/v1/embeddings`, n=1 request per row:
  `openai/text-embedding-3-small` 1536, `openai/text-embedding-3-large` 3072,
  `google/gemini-embedding-2` 3072, `google/gemini-embedding-001` 3072, all unit-norm. The BARE
  `text-embedding-3-small` also answers at 1536, which refutes the brief's claim that unprefixed
  ids would 404. `gemini-embedding-001` is unit-norm ONLY at 3072 (0.694 at 1536, 0.582 at 768),
  so truncated widths must not be registered while profiles declare `normalization='l2'`.
  Re-measure: `scratchpad/emb2.py`, reproduced in the registry docstring.
- Voyage widths are NOT verified here: no `VOYAGE_API_KEY` on this machine. They are declared from
  the provider's documented defaults and are protected by P3 rather than by a measurement.
- `_index_fingerprint` hashes `embedding_profile_id(embedder)` alone, no dimension term
  (recall/index.py:441).

## Confounds I can name now

- **A stub that is more cooperative than the real client.** A stub returning a fixed width proves
  the check fires, not that the real provider's response shape reaches it. P3 is only as good as
  the stub's fidelity to `voyageai.Client.embed(...).embeddings` and
  `OpenAI().embeddings.create(...).data[n].embedding`.
- **P4 could pass for the wrong reason** if the two profiles differ in anything besides width.
  They must differ ONLY in dimension, or the test proves nothing about the width term.
- **P2 could pass because the store or control-plane stub failed first.** The assertion must name
  the artifact failure specifically, not merely `ready is False`.
- Voyage entries could be wrong in width and every test still pass, because no test can reach the
  real API. That is a known, stated gap, not a covered one.

## Result (2026-08-18)

**Status:** measured

**5 of 5 predictions held.** Measured with `tests/test_hosted_embedding_profiles.py` (45 tests
in that file plus the registry inventory), full suite `5506 passed, 32 skipped` with a database.

| | Prediction | Measured |
|---|---|---|
| P1 | identity survives `build()` | held: `profile_id` is `voyage-code-3-v1`, digest `hosted-unverifiable` |
| P2 | enterprise gate refuses hosted | held: `ready is False`, failure names attestation, not width |
| P3 | width disagreement raises at construction | held: 512 against a declared 1024 raises `ValueError` |
| P4 | index fingerprint separates two widths | held: differed once both carried registry ids |
| P5 | hosted stays subject to the context check | held: `raw-v1` under a `section` policy raises |

**Gap: none in direction, one in confidence.** I predicted P4 was likeliest to surprise because it
was the only current-state number I measured myself. It did not surprise. What surprised instead
was the apparatus, twice, and both would have produced a confident wrong answer:

1. **The re-measure script imported a different checkout.** Running a file under `scripts/` puts
   the script's directory on `sys.path[0]`, not the working directory, so `import recall` resolved
   through the editable install, which this project shares across ~18 worktrees. It reported the
   MAIN checkout's registry while appearing to work. Fixed by pinning the repo root. This is the
   `guards that cannot fail` shape applied to a measurement tool rather than to a guard.
2. **Two of the brief's premises were false, and one was false in the direction that stops you
   trying.** Bare `text-embedding-3-small` does NOT 404 on OpenRouter (measured, 1536 wide), and
   OpenRouter's `/v1/models` lists ZERO embedding models while `/v1/embeddings` serves them, so
   the catalogue endpoint cannot validate an id and its silence is not evidence. Had I trusted the
   catalogue result alone I would have concluded no Gemini embedding model existed there and
   dropped the profile.

**Confound check.** The named confounds were addressed rather than assumed away: the stubs
reproduce both SDKs' real response shapes; P4's two profiles differ ONLY in `dimension`
(asserted in the test); P2 asserts the specific attestation failure and has a positive control
proving the harness can produce a passing gate. The apparatus requirement (rule 2) was met by
mutation rather than by assertion: each of the five guards was removed in turn, compiled, and its
test confirmed RED.

**Still not covered, and stated rather than hidden:** Voyage's two declared widths. No
`VOYAGE_API_KEY` on this machine, so no test can reach the real API and the 1024 declarations rest
on provider documentation. The construction-time width check converts a wrong declaration into a
loud startup failure instead of a corpus of mislabelled vectors, which bounds the damage without
verifying the claim.
