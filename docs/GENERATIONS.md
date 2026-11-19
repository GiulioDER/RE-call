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
SHA-256. A local manifest file may be pinned with `--sha256` and `--size` (for `generation build`,
`--manifest-sha256` and `--manifest-size`), and each pin given is compared with the file. A local
file has no object version, so `--version-id` and `--manifest-version-id` are refused for one.

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

The registered multimodal profile is a separate physical specialist. Its generation and calibration
must use tenant `re-call-multimodal` and profile `voyage-multimodal-3.5-v1`; matching dimensions are
not sufficient for reuse.

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

The one exception is an active atomic rescue: rollback then builds the target's artifact first and
refuses when it cannot, since the target would refuse every unscoped search without one
([below](#atomic-rescue)).

Strict refusal at **read** time has landed too: under the default strict policy, a search against
an active generation whose calibration is absent, stale or uncertified raises `TrustRefusal`
before any corpus text is fetched. Only development mode still answers, marked `degraded` with
the reason in `failure_code` ([CALIBRATION.md](CALIBRATION.md#the-seven-failure-codes)).

## Atomic rescue

When `RECALL_ATOMIC_RESCUE_MODE` resolves to `active` for a tenant (`RECALL_ATOMIC_RESCUE_MODE_BY_TENANT`
overrides it per tenant), an unscoped search loads
`<RECALL_ATOMIC_RESCUE_ARTIFACT_ROOT>/<generation_id>.<calibration_id>/manifest.json` (or, for an
artifact built before 2026-11-17, `<generation_id>/manifest.json`) and refuses when it is missing or
was built for another calibration, unless `RECALL_ATOMIC_RESCUE_WHEN_MISSING=skip` (a desktop install,
[WIZARD.md](WIZARD.md#atomic-rescue)) lets it answer without the rescue. A certified generation
with five or fewer chunks has no sixth candidate and serves ordinary trusted results without an
artifact. With the rescue active, `recall generation promote` and
`recall generation rollback` build the target generation's artifact first, and refuse, saying how
to turn the rescue off, when they cannot: the generation needs a published certified calibration,
and the embedder must be the one that built it. A certified upload through an MCP server does the
same and fails instead of activating; a development-mode upload promotes without building one. Publishing a calibration (`recall calibration calibrate --publish`,
`carry-forward --publish`, `auto` unless `--no-publish`, the MCP publish tool) rebuilds the artifact for it,
since the artifact names its calibration; a rebuild that fails is reported with the command that
retries it, and the publication stands. The mode is read from the environment of the process doing
this, so the CLI must see the same settings as the server (the wizard writes them into the
project's `.env`). `recall wizard` does it between publish and promote ([WIZARD.md](WIZARD.md#atomic-rescue)).

```bash
recall --tenant acme generation rescue                  # the active generation
recall --tenant acme generation rescue <generation-id>  # any generation
```

`rescue` builds whatever the mode, with only the root set, so an existing deployment can build the
artifacts of its active and previous generations before it turns the rescue on, the order in which
no search refuses. Run it when a publish reported that its rebuild failed: until the artifact is rebuilt
unscoped searches that need the artifact refuse under the default missing-artifact policy.
Views already embedded are reused from `<root>/micro-views.sqlite`, so a rebuild for a new
calibration embeds nothing. An
artifact already in force is reported and kept.

`gc` removes the artifacts of the generations it collects, and a forget removes every artifact of
the tenant's generations that names the forgotten source and erases its views, whatever the mode.
It does so under `RECALL_ATOMIC_RESCUE_ARTIFACT_ROOT` when it is set, and under every root a build
of the tenant recorded in the audit trail (event `atomic_rescue_built`), so a `recall forget` run
outside the project directory, which does not load its `.env`, still erases them. A recorded root
this process cannot reach (another host, a container, a folder it may not read) is named in the
forget's result, since a re-run finds no rows to erase by. A recorded root that no longer exists
anywhere is named on every forget, because nothing here can tell it from one on another host:
create an empty folder at that path to retire it. An empty folder there is skipped.

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
