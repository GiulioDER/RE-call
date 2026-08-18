# Serving a corpus before it has a calibration

**Status:** design, 2026-08-18. **Nothing here is implemented.** It is a proposal with measured
support for two of its claims and a list of open questions it does not answer. The measurements are
pre registered in `docs/preregistrations/2026-08-18-uncalibrated-first-run.md`. All source citations
are against the tree at commit `bd582316`.

The goal is unchanged: someone must be able to index and search before they have labelled queries.
What follows argues that `RECALL_ENV=development` is the wrong mechanism for that, not because it is
badly implemented, but because it answers a question about a **process** when the question is about
a **tenant**.

## 1. What I confirmed in the code

| Claim | Site | Verdict |
|---|---|---|
| Server builds `GenerationStore` only in production | `recall_mcp/server.py:627` | confirmed |
| Missing `generation_id` degrades to `"legacy"` | `recall_mcp/service.py:906` | confirmed. A second site uses the same default but maps it to `None` immediately after, so the two do not behave identically |
| `promote()` refuses in production, needs a flag otherwise | `recall/generations.py:796` | confirmed |
| No generation means `INDEX_NOT_READY` **at the readiness endpoint** | `recall/readiness.py:110` | confirmed, but this is **not** the search path. See Q2 |
| `calibration = None` is deliberate, and names an open design question | `recall/cli.py:2026-2037` | confirmed |
| Legacy `chunks` has no `source_sha256` **column** | `recall/store.py:271` vs `recall_chunks_v1` | confirmed as stated, and **narrower than "nothing to reuse"**: the metadata carries `content_hash`, which is what F3 is about |

### Four findings that change the available answers

**F1. Promotion is not required, for either calibration or serving.**
`CalibrationRepository._generation` accepts states `{"ready", "active", "retired"}`
(`recall/calibration_v2.py:349`). `GenerationStore.pin_generation` accepts the same three
(`recall/generation_store.py:146`). And `SERVABLE_ACTIVE_STATES = frozenset({"ready", "active"})`
(`recall/control_plane.py:34`), so the enterprise control plane **already treats `ready` as
servable**. What `promote()` adds over calibration and serving is that it sets
`recall_tenant_state.active_generation_id`, the *default selection* rather than the permission, and
even that is not exclusive to it (see F2).

**F2. The promotion gate does not hold the invariant it claims to hold.**
`rollback()` (`recall/generations.py:843`) writes the same `active_generation_id` column with **no
environment check and no `unsafe_development` flag**, and its target may be in state `ready`
(`recall/generations.py:855`). So "no ungated generation becomes active in production" is not a
property this system has. `promote()`'s message says the refusal stands "until certification gates
land", which is accurate: it is a placeholder, not a safety property.

**F3. The legacy `chunks` table records enough to establish a binding, not merely assert one.**
It has no `source_sha256` column, but `recall/index.py:810` stamps into every chunk's metadata:
`content_hash`, `index_fingerprint`, `embedding_profile`, `context_mode`, `context_version`, `ord`
and `file`, all written **at embed time**, so checking them is verification rather than
reconstruction.

🔁 **Corrected 2026-08-18 by measurement.** An earlier version of this paragraph said
`embedding_profile` "is the identity of the embedder that produced the vector". **It was not**, at
the tree this was measured against: the fallback returned the literal string
`bge-small-symmetric-v1` for any model without a registered profile, so a 1024 dimensional corpus
carried a 384 dimensional profile's id.

🔁 **Fixed upstream, 2026-08-18, by #370**, which this measurement prompted. `_fallback_profile_id`
(`recall/embeddings.py:699`) now derives `unregistered__{model}__{dimension}__{kind}`
(`recall/embeddings.py:750`) instead of claiming a registry id it does not have.

⚠️ **That does NOT restore `embedding_profile` as an adoption check, and the design still must not
use it.** Every corpus indexed *before* #370 carries the old literal, which is exactly the
population an adoption path exists to read. A fix to the writer does not retroactively repair rows
already written. Only `content_hash` is load bearing here, and the accessor that returns it is
`PgVectorStore.source_raw_hashes` (`recall/store.py:2093`), **not** `source_content_hashes`
(`:2075`), which coalesces `index_fingerprint` first and therefore returns the defective identifier.

⚠️ **`content_hash` is media type dependent since `bd582316`.** A markdown source is hashed as
decoded, newline normalised, NUL stripped text re encoded as UTF-8 (`recall/index.py:671` and
`:690`); any other media type is hashed as **raw bytes** (`:692`). Any adoption path must branch the
same way, or it will refuse every markdown file with CRLF or a BOM.

**F4. The first run wizard is half built and already solves the hardest part.**
`recall/wizard/queryset.py`'s `generate_offline` produces a labelled query set from the corpus with
no human and no network. (The module also offers an LLM generator, which uses both; only the
offline one has that property, and only it was measured.) Its docstring names the reason it exists:
the hand labelled set "is the step that makes calibration too hard to be worth doing, and it is the
step a first-run wizard has to remove". It is not wired into the CLI.

## 2. The diagnosis

`RECALL_ENV` is one string carrying at least six unrelated policies:

1. **Ingestion source.** Production refuses local filesystem indexing (`recall_mcp/service.py:1769`, `recall/cli.py:2043`).
2. **Auth.** Production refuses static bearer tokens (`recall_mcp/auth.py:366`).
3. **Store class.** Production selects `GenerationStore`, at **three** sites, not one:
   `recall_mcp/server.py:627`, `recall/cli.py:2082`, and the `generation_mode` parameter threaded
   into `StoreRegistry` (`recall_mcp/stores.py:154`), whose value is `generation_mode and not
   enterprise` and therefore also encodes the control plane interaction.
4. **Retrieval legs.** Production disables the learned sparse leg (`recall/retriever.py:187`).
5. **Promotion permission.** Production refuses it outright (`recall/generations.py:796`).
6. **Generation creation.** Production requires a verified pipeline identity and refuses
   `allow_unverified` (`recall/generations.py:286`, `:288`), which an adopted generation cannot satisfy with
   an unpinned default embedder.

Policies 1, 2 and 4 are genuinely process wide: they are about what this deployment is allowed to
touch. Policies 3, 5 and 6 are not. Whether a *tenant* has a generation worth serving, and whether a
*generation* is fit to be the active one, are per tenant and per generation facts that happen to be
read off a process wide variable.

That is the whole bind. Policy 5 says "promote only in development"; policy 3 says "serve only in
production"; and because both read the same string, the two cannot be satisfied at once. Nobody
designed that. It fell out of overloading one variable.

The corollary matters more than the bind: **`RECALL_TRUST_MODE=development` is not dishonest because
it serves without a calibration. It is dishonest because it serves with a threshold of 0.5 that was
never measured against anything.** A first run path does not need to relax the trust gate. It needs
to give the gate a number that is bound to something.

## 3. The four questions, answered

### Q: Should "usable without a calibration" be a property of the environment?

**No. It is a property of the tenant, and it should be recorded in the database.**

An environment variable is invisible, process wide, and belongs to whoever last edited a unit file.
Readiness is already per tenant everywhere else in this codebase, and `recall/readiness.py`'s module
docstring argues the point for a different reason: "A per-tenant fault must cost exactly one
tenant". The same argument applies to a per tenant relaxation, and an operator must be able to *see*
it.

So: add `serving_mode` to `recall_tenant_state`, with three values.

| `serving_mode` | Means | `trust_state` on the wire |
|---|---|---|
| `certified` | A published, certified calibration binds to the active generation. | `trusted` |
| `provisional` | A generation is active and a **measured but uncertified** threshold is bound to it. | `provisional` |
| absent | No active generation. | `refused` |

`provisional` is a new `TrustState` and a new `CalibrationStatus`. It is deliberately **not**
`degraded`, because `degraded` currently means "we ran with a number bound to nothing" and
`provisional` means "we ran with a number measured on this corpus, on a query set no human
labelled".

What this buys, stated precisely: a stray exported variable can no longer make a tenant *look*
certified, because `serving_mode` is read from the database. It does **not** close the escape hatch.
`RECALL_TRUST_MODE=development` survives untouched (section 6), so a stray export can still relax a
strict gate.

### Q: Can promotion and serving be available under the same setting?

**Yes, by removing the environment term from both rather than by aligning them.**

Given F2, the current gate is not protecting anything, so replacing it costs no safety. Replace it
with the certification gate its own error message promises:

```
promote(generation_id, *, provisional_reason: str | None = None)

  certified published calibration bound to this generation  -> allowed, serving_mode = 'certified'
  no such calibration, provisional_reason given             -> allowed, serving_mode = 'provisional',
                                                               reason recorded in the audit log
  no such calibration, no reason                            -> UnsafePromotion
```

`unsafe_development=True` becomes `provisional_reason="..."`, which is strictly more informative: a
boolean records that somebody opted in, a reason records *what they were doing*.

⚠️ **This must be applied to `rollback()` in the same change**, and doing so needs a decision the
design does not make. `rollback()` today takes no arguments (`recall/generations.py:843`), so it
cannot record a reason, and its target's calibration may have gone stale or superseded in the
meantime. Under the certification rule it would either refuse, which blocks incident recovery
exactly when it is needed, or grant `provisional` with no reason, which is the weakness this design
uses to argue against `unsafe_development=True`. See the open questions.

Then delete the `generation_mode` branch. Two corrections to an earlier version of this section,
both found by audit:

⛔ **It is three sites, not one** (policy 3 above), and the `StoreRegistry` parameter cannot simply
be deleted because it also encodes `and not enterprise`. Naming only `server.py:627` would have been
the same "fixed one writer, left the other" failure this design levels at `promote()`.

⛔ **"A tenant with no generation gets `INDEX_NOT_READY`, already implemented" is false on the
search path.** `readiness.py:110` is a different entry point that receives `generation_id` as an
argument. On search, `GenerationStore.generation_binding()` raises `NoActiveGeneration`, which is
swallowed by the broad `except Exception` in `trusted_search` and re raised as
`DEPENDENCY_UNAVAILABLE` (`recall/trust.py:687`), whose advice text calls that condition "an
outage, not an empty result". Mapping `NoActiveGeneration` to `INDEX_NOT_READY` is therefore a **prerequisite** of
this change, not a consequence of it.

⛔ **"GenerationStore always" breaks development ingestion over MCP.** The `recall_index` tool writes
through the same store object the branch selects, and `GenerationStore.upsert` raises
`ImmutableGenerationError`. Store selection has to be split by direction, a read store chosen per
tenant readiness and a separate legacy write store for ingestion, or the MCP index tool has to be
redirected. Section 6's claim that policy 1 is untouched does not survive without that split.

### Q: Is there an honest install time calibration binding that does not need a full build?

**Yes, and `recall/cli.py:2037` was right to refuse the version that would have been dishonest.**

The comment says: "Resolve that by deciding where install-time calibration binds, not by reinstating
the line below." My answer is that **there is no honest process global calibration and there should
not be one.** A threshold is a property of the tuple (tenant, generation, pipeline, corpus, query
set). Any artifact that does not carry all five is `legacy_unbound`, and the strict policy is right
to refuse it.

The install time binding is the existing generation bound path, with the human removed from the one
step that made it impractical:

1. Adopt or build a generation (section 4).
2. `generate_offline` produces the labelled set from the corpus itself. Already written, not wired up.
3. `CalibrationRepository.calibrate(generation_id, ...)` fits and binds. This already works against
   a `ready` generation (F1), so it does not require promotion and does not re embed anything.
4. The artifact records `query_set_provenance: "generated"`, and a generated set certifies to
   `PROVISIONAL`, never `CERTIFIED`.

Step 4 is the honesty mechanism. The threshold is a real measurement of a real corpus, which 0.5 is
not. It is weaker evidence than a human labelled set, because the questions were invented by the
same process being scored. Both facts belong in the wire payload.

🔁 **Corrected 2026-08-18 by measurement, and the correction matters.** I assumed a generated set
would *fail* the certification bar, which is why a separate status looked necessary. **It passes**,
on both corpora measured. So `PROVISIONAL` cannot be justified as "the statistics are too weak". It
is justified **only** by provenance, which `docs/CALIBRATION.md:169-176` states independently: a
certified calibration "does not mean the labelled set was a good one".

⚠️ **`query_set_provenance` as described is caller asserted and forgeable**, and adding it is a
checksum covered schema change. See the open questions.

### Q: Should a first run adopt a legacy `chunks` corpus without re embedding?

**Yes, and F3 says the binding can be established rather than asserted, which is the condition the
question sets.**

Writing `recall_chunks_v1` rows from `chunks` rows would fabricate `source_sha256` and
`object_version_id`, which is what `recall/lineage.py:269` refuses for `file://` objects. The escape
is that the legacy metadata was written at embed time and can be checked against the world now.

`recall generation adopt --from-chunks` proceeds per source:

1. Read `metadata->>'content_hash'` via `source_raw_hashes`. Absent means **not adoptable**.
2. Read the file at `metadata->>'file'`. Missing or unreadable means not adoptable.
3. Re derive the hash **exactly as the indexer does for that media type**: decoded, newline
   normalised, NUL stripped text for markdown (`recall/index.py:671`, `:690`), raw bytes otherwise
   (`:692`). Not equal means the file changed since indexing: not adoptable.
4. 🔁 **Corrected 2026-08-18 by measurement.** This step originally compared
   `metadata->>'embedding_profile'` to the configured embedder's profile id. **That check does not
   work** (F3), and `index_fingerprint` inherits the defect because `_index_fingerprint` hashes the
   same value (`recall/index.py:447`). Neither stored field may gate adoption. The check is the
   attestation sample below.
5. Only then copy the vector, setting `source_sha256 = content_hash` and
   `object_version_id = content_hash`, the rule `recall/lineage.py:269` enforces for local files.

Two properties this gives, stated as the `file://` comment states its own:

- **What it buys is detection of divergence, never prevention.** A file rewritten between step 3 and
  step 5 is not caught. That is the same guarantee a `file://` manifest already offers.
- **`content_hash` proves which bytes were read, not which chunker ran.** So adoption **must** draw
  a **pipeline attestation sample**: re embed N chunks chosen at random from the enumerated
  population and compare to the stored vectors. Given step 4, this is the only sound *embedder*
  check available. Measured on a 20 chunk sample: 20 of 20 at cosine 1.0000 against an off diagonal
  control of 0.709 max, running in seconds. ⚠️ It does **not** check the chunker, because it re
  embeds the stored chunk text, which is downstream of chunking.

An adopted generation is `provisional` by construction, carries an `unverified_reason` naming its
origin plus the five way adoption census (verified / changed / missing / metadata absent /
unreadable) in its manifest, and can only reach `certified` through a calibration against a human
labelled set. Adoption never manufactures certification; it manufactures a *starting point*.

## 4. The resulting first run

```
recall init <dir>
```

One command, no environment variables, and every step refusable with a real reason:

1. Index into `chunks` if needed, or detect an existing corpus.
2. Adopt into a generation, verifying every source. Report the five way census. Re embed only what
   failed verification.
3. Generate a query set. Calibrate. Publish.
4. Promote with the certification gate. A generated query set lands `serving_mode = 'provisional'`.
5. Write `.mcp.json`.

Search **would then work**, in strict mode, with a proposed `trust_state: "provisional"` on every
result and a threshold measured on the user's own corpus, and `RECALL_TRUST_MODE` would not need to
be set. None of that exists yet.

## 5. What was measured

Pre registered before the first run; full record, controls and scope limits in the pre registration.

- **Adoption fidelity. Confirmed, on a narrowed criterion.** 1,080 of 1,080 sources and 8,716 of
  8,716 chunks verified on the `content_hash` conjunct (the registered `embedding_profile` conjunct
  was withdrawn as unsound, not met). Attestation 20 of 20 at cosine 1.0000 against a 0.9999 bar.
  A **proxy** for adoption cost, excluding attestation, ran in 3.3 s against an estimated ~10 hours
  for a full rebuild. One registered sub prediction is **not tested**: there were no failures.
- **Provisional threshold quality. Half confirmed, half falsified.** On a held out human labelled
  set the generated threshold cut false confidence from 1.00 to 0.05, that is 20 of 20 down to 1 of
  20 (bar was 0.10 absolute), at a false abstain cost of 1 of 20, matching a threshold fitted on the
  human labels and no worse on false abstain, across all 20 seeds. **Falsified:** the generated set
  *does* reach the certification bar. On the fixture, AUC 1.0000 [1.0000, 1.0000] at 20 per class;
  on the memory corpus, 0.9656 [0.9243, 1.0000] at 40 per class. Both clear the 20 per class sample
  floor and both interval lower bounds clear the 0.90 separability bar, so both certify. ⚠️ The
  memory corpus figures carry a `k=1` retrieval confound (3 of 10 probe queries disagreed with
  `k=200`); the fixture result is the clean one.
- ⚠️ Two method deviations are on the record: the corpus was substituted, because no human labelled
  set exists for the registered one, and the primary ran at half the registered sample size.

## 6. Open questions this design does not answer

Recorded rather than invented. Each needs a decision before implementation.

1. **The enterprise control plane is a second activation surface.** `ControlPlane.set_route()` and
   `cutover()` (`recall/control_plane.py:802`) write `recall_tenant_routes.active_generation`, and
   `StoreRegistry._get_generation` consults the route **before** the tenant state path. Under
   `RECALL_ENTERPRISE_CONTROL_PLANE=1` the proposed gate governs nothing. Either those two take the
   same certification rule, or `serving_mode` is not the single per tenant readiness answer and
   enterprise keeps the old semantics.
2. **`rollback()`'s signature and refusal semantics**, per Q2 above.
3. **Is `serving_mode` authoritative or advisory?** It is a cached derivative of something currently
   recomputed per query by `CalibrationRepository.resolve()`. Both `forget()` and `publish()`
   invalidate a calibration without touching tenant state, so a tenant can read `certified` while
   the live resolver says stale.
4. **Adding `provisional` to `TrustState` hits an exhaustive whitelist that raises**, not a defaulted
   mapping: `recall/reasoning.py:1461` rejects anything outside `{trusted, degraded, refused}`, on a
   versioned API whose version is unbumped, plus several `!= "trusted"` comparisons that would
   silently downgrade.
5. **The strict gate is binary on `CERTIFIED`.** `code_for_status` returns a failure code for
   everything else via a fail closed default (`recall/trust_policy.py:109`), so a new
   `CalibrationStatus.PROVISIONAL` is silently classified `CALIBRATION_UNCERTIFIED` and strict
   refuses it. "Search works in strict mode with `trust_state: provisional`" is unreachable without
   changing that, and `code_for_status`'s return type is part of a documented stable API.
6. **The adoption copy is under specified.** `chunk_ordinal`, `source_uri`, the root that resolves
   the relative `metadata->>'file'`, and the NOT NULL `tsv` (legacy is `GENERATED ... 'english'`,
   v1 uses the pipeline's declared regconfig) all need a stated origin, as do the NOT NULL columns
   of `recall_generations` and the `file://` manifest entries.
7. **How an adopted generation reaches `ready`**, which `promote()` requires. The only existing
   transition into `ready` is `_validate()`, which compares every row against a stored manifest.
   There is also a `legacy_unverified` state that nothing downstream accepts.
8. **Transaction and failure semantics for adoption.** `gc()` only reclaims `retired` or `failed`
   generations, so a partially adopted one is reachable by nothing.
9. **The attestation sample needs a sample size rule.** 20 of 8,716 is 0.23 percent coverage and
   cannot detect a partially contaminated corpus, which is the hazard it is now the sole defence
   against. N as a function of the smallest fraction worth detecting, stratified by source, with the
   cosine bar and the abort rule stated here rather than only in the pre registration.
10. **`query_set_provenance` must be unforgeable**, and its addition is an artifact version bump and
    a migration.
11. **Two published contracts enumerate the trust states exhaustively** (`docs/USING_WITH_CLAUDE.md`,
    `docs/REASONING_CONTRACT.md`) and a pinned test asserts the current set.

## 7. What this design does not do

- It does not touch policies 1, 2 or 4, **subject to the `recall_index` correction in Q2**.
  `RECALL_ENV` survives for those and should be renamed per axis so it cannot be re overloaded.
- It does not lift the strict default. `provisional` is a *third* answer, not a relaxation.
- It does not make a generated query set equivalent to a labelled one, and says so on the wire.
- It does not fix the 0.5 constant's remaining user, `RECALL_TRUST_MODE=development`, which stays as
  the explicit escape hatch for an uncalibratable corpus.
