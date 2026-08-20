# Immutable index generations

RE-call v1 identifies every index by the complete retrieval pipeline and builds replacements as
immutable generations. It never treats equal vector dimensions as proof that embeddings are
compatible.

## Identity boundary

`PipelineIdentity` canonically hashes all inputs that can change retrieval semantics:

* embedder provider, model, immutable revision or artifact digest, and dimension;
* chunker algorithm, schema version, and complete canonical configuration;
* PostgreSQL text-search configuration;
* pipeline schema version.

Production rejects embedders without an immutable revision or artifact digest. Development can
use one only with `--unverified-development`; the stored generation remains visibly unverified.
A source can reuse chunks only when its SHA-256 and the complete pipeline fingerprint both match.

## Corpus manifests

Production corpus inputs are canonical JSON manifests containing one exact S3 object version per
URI. Every entry records tenant, corpus version, URI, object version ID, media type, byte size, and
SHA-256. Before indexing, RE-call verifies the returned object version, declared length, actual
length, and digest. A missing or changed object fails the generation.

Configure deployment-owned access with:

```text
RECALL_S3_ALLOWLIST=corpus-bucket/tenant-prefix/,archive-bucket/released/
RECALL_S3_ENDPOINT_URL=https://s3.example.internal
```

The endpoint is optional. Credentials use the normal SDK credential chain and are never accepted
from a manifest or request. A URI outside the configured bucket and prefix allowlist is refused.

Create and verify a manifest:

```bash
recall --tenant acme manifest create --corpus-version 2026-08-03 \
  --objects inventory.json --output manifest.json
recall --tenant acme manifest verify manifest.json
```

`inventory.json` is an array of objects with `uri`, `version_id`, `media_type`, `size`, and
`sha256`. A manifest stored in S3 must itself be supplied with its exact version ID, size, and
SHA-256.

## Lifecycle

The enforced lifecycle is:

```text
building -> validating -> ready -> active -> retired
                  \-> failed
```

Build, validate, and inspect with:

```bash
recall --tenant acme generation build manifest.json \
  --embedder-provider local --embedder-revision <immutable-revision>
recall --tenant acme generation validate <generation-id>
recall --tenant acme generation list
```

Promotion updates active and previous pointers in one transaction. Searches pin the active pointer
for the whole retrieval operation, so a concurrent promotion or rollback cannot mix generations.
Rollback atomically restores the previous ready generation.

Generation-bound calibration artifacts are documented in [CALIBRATION.md](CALIBRATION.md).

**In production**, promotion requires the generation's calibration to resolve CERTIFIED, and
`--unsafe-development-promotion` is refused there rather than honoured:

```bash
recall --tenant acme calibration calibrate --generation <generation-id> --queries labels.json --publish
recall --tenant acme generation promote <generation-id>
```

**In development**, there is no calibration requirement and the conspicuous flag is mandatory,
so the unchecked path always names itself:

```bash
recall --tenant acme generation promote <generation-id> --unsafe-development-promotion
```

**Rollback is ungated in both**, because it is what an operator reaches for when the active
generation is the problem. It records the target's calibration status and an optional reason
instead of refusing (`docs/UNCALIBRATED_FIRST_RUN_DESIGN.md`, section 6):

```bash
recall --tenant acme generation rollback
```

What is still deferred is strict refusal at **read** time: an active generation whose calibration is
absent or stale still answers, marked uncalibrated.

## Retention and erasure

Garbage collection retains two previous generations for at least seven days by default. Capacity
planning must allow **2.2 times** steady-state index storage during a blue-green rebuild. Change
retention only through an explicit administrative command:

```bash
recall --tenant acme generation gc --retention-days 7 --retain-previous 2
```

Forget acquires a source-specific transaction lock, deletes the source from active, previous,
ready, building, validating, retired, and failed v1 generations, then commits an audit event and a
persistent tombstone. A concurrent build observes the tombstone and cannot reintroduce the source.
Rollback and garbage collection do not remove tombstones. The tombstone also changes each affected
generation's effective corpus fingerprint, so a calibration measured before erasure becomes stale
and a replacement generation cannot recover the old binding.

Rows from a v0.8 table remain readable through the legacy API and are registered as
`legacy_unverified` evidence. They are never copied into `recall_chunks_v1`, selected as an active
generation, or exposed through strict v1 generation search.

## Generations and the strict trust gate

A generation without a published, exactly-bound calibration cannot serve under the strict policy.
Two consequences worth planning around:

- **Promotion is not the last step.** A newly promoted generation has a fresh corpus fingerprint,
  so any calibration bound to its predecessor is `CALIBRATION_STALE`. Calibrate and publish
  against the new generation before promoting it, or the promotion will refuse every search.
- **Privacy erasure stales calibration.** Erasure changes the effective corpus fingerprint by
  design, which is what makes the old measurement no longer applicable. Expect `CALIBRATION_STALE`
  after an erasure and recalibrate.

A tenant in either state refuses only that tenant: `process_readiness` reports it under
`tenants_unready` but stays ready, so the pod continues serving everyone else.
