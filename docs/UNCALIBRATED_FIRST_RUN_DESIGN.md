# Serving a corpus before it has a calibration

**Status:** design, 2026-08-18. Nothing here is implemented. The measurements it depends on are
pre registered in `docs/preregistrations/2026-08-18-uncalibrated-first-run.md`.

The goal is unchanged: someone must be able to index and search before they have labelled queries.
What follows argues that `RECALL_ENV=development` is the wrong mechanism for that, not because it is
badly implemented, but because it answers a question about a **process** when the question is about
a **tenant**.

## 1. What I confirmed in the code

Every site below was read on 2026-08-18 at `55397af5`. The quoted text is verbatim. Line numbers
have drifted from the report and are corrected here.

| Claim | Site (corrected) | Verdict |
|---|---|---|
| Server builds `GenerationStore` only in production | `recall_mcp/server.py:627` | confirmed |
| Missing `generation_id` degrades to `"legacy"` | `recall_mcp/service.py:906` **and `:1315`** | confirmed, **two** call sites |
| `promote()` refuses in production, needs a flag otherwise | `recall/generations.py:762-768` | confirmed |
| No generation means `INDEX_NOT_READY` regardless of calibration | `recall/readiness.py:110` | confirmed |
| `calibration = None` is deliberate, and names an open design question | `recall/cli.py:2021-2033` | confirmed |
| Legacy `chunks` has nothing to reuse | `recall/store.py:259` (`DEFAULT_TABLE = "chunks"`) vs `recall_chunks_v1` | confirmed |

### Four findings the report did not have, which change the answers

**F1. Promotion is not required, for either calibration or serving.**
`CalibrationRepository._generation` accepts states `{"ready", "active", "retired"}`
(`recall/calibration_v2.py:349`). `GenerationStore.pin_generation` accepts the same three
(`recall/generation_store.py:146`). And `SERVABLE_ACTIVE_STATES = frozenset({"ready", "active"})`
(`recall/control_plane.py:34`), so the enterprise control plane **already treats `ready` as
servable**. The only thing `promote()` does that nothing else does is set
`recall_tenant_state.active_generation_id`, which is the *default selection*, not the permission.

**F2. The promotion gate does not hold the invariant it claims to hold.**
`rollback()` (`recall/generations.py:812`) writes the same `active_generation_id` column with **no
environment check and no `unsafe_development` flag**, and its target may be in state `ready`
(`recall/generations.py:824`). So "no ungated generation becomes active in production" is not a
property this system has. `promote()`'s message says the refusal stands "until certification gates
land", which is accurate: it is a placeholder, not a safety property. Treating it as one is what
created the bind.

**F3. The legacy `chunks` table records enough to establish a binding, not merely assert one.**
It has no `source_sha256` column, but `recall/index.py:707` stamps into every chunk's metadata:
`content_hash` (sha256 of the source bytes actually read, `recall/index.py:621`),
`index_fingerprint`, `embedding_profile`, `context_mode`, `context_version`, `ord` and `file`.
`embedding_profile` is the identity of the embedder that produced the vector. These were written
**at embed time**, so checking them is verification, not reconstruction.
`PgVectorStore.source_content_hashes` already queries them (`recall/store.py:2050`).

**F4. The first run wizard is half built and already solves the hardest part.**
`recall/wizard/queryset.py` generates a labelled query set from the corpus with no human and no
network, and its docstring names the exact reason it exists: the hand labelled set "is the step that
makes calibration too hard to be worth doing, and it is the step a first-run wizard has to remove".
It is not wired into the CLI. Only `recall/wizard/inventory.py` is (`recall/cli.py:1513`).

## 2. The diagnosis

`RECALL_ENV` is one string carrying five unrelated policies:

1. **Ingestion source.** Production refuses local filesystem indexing (`recall_mcp/service.py:1769`, `recall/cli.py:1893`).
2. **Auth.** Production refuses static bearer tokens (`recall_mcp/auth.py:348`).
3. **Store class.** Production selects `GenerationStore` (`recall_mcp/server.py:627`).
4. **Retrieval legs.** Production disables the learned sparse leg (`recall/retriever.py:187`).
5. **Promotion permission.** Production refuses it outright (`recall/generations.py:763`).

Policies 1, 2 and 4 are genuinely process wide: they are about what this deployment is allowed to
touch. Policies 3 and 5 are not. Whether a *tenant* has a generation worth serving, and whether a
*generation* is fit to be the active one, are per tenant and per generation facts that happen to be
read off a process wide variable.

That is the whole bind. Policy 5 says "promote only in development"; policy 3 says "serve only in
production"; and because both read the same string, the two cannot be satisfied at once. Nobody
designed that. It fell out of overloading one variable.

The corollary matters more than the bind: **`RECALL_TRUST_MODE=development` is not dishonest because
it serves without a calibration. It is dishonest because it serves with a threshold of 0.5 that was
never measured against anything.** `recall/cli.py` says so in band, "an UNCERTIFIED demonstration
threshold of 0.5". A first run path does not need to relax the trust gate. It needs to give the gate
a number that is bound to something.

## 3. The four questions, answered

### Q: Should "usable without a calibration" be a property of the environment?

**No. It is a property of the tenant, and it should be recorded in the database.**

An environment variable is invisible, process wide, and belongs to whoever last edited a unit file.
Readiness is already per tenant everywhere else in this codebase, and `recall/readiness.py`'s module
docstring argues the point at length for a different reason: "A per-tenant fault must cost exactly
one tenant". The same argument applies to a per tenant relaxation. It must cost exactly one tenant,
and an operator must be able to *see* it.

So: add `serving_mode` to `recall_tenant_state`, with three values.

| `serving_mode` | Means | `trust_state` on the wire |
|---|---|---|
| `certified` | A published, certified calibration binds to the active generation. | `trusted` |
| `provisional` | A generation is active and a **measured but uncertified** threshold is bound to it. | `provisional` |
| absent | No active generation. | `refused`, `INDEX_NOT_READY` |

`provisional` is a new `TrustState` alongside `trusted` / `degraded` / `refused`, and a new
`CalibrationStatus` alongside the eight that exist. It is deliberately **not** `degraded`, because
`degraded` currently means "we ran with a number bound to nothing" and `provisional` means "we ran
with a number measured on this corpus, on a query set no human labelled". Collapsing them would
throw away the only distinction the redesign creates.

Consequence worth naming: a process can no longer relax another operator's tenant by having a stray
`RECALL_TRUST_MODE` exported. That is the failure mode recorded in `recall_mcp/auth.py:344`, "a
developer setting `RECALL_ENV=development` permanently in their shell".

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

No environment term at all, in any branch. `unsafe_development=True` becomes
`provisional_reason="..."`, which is strictly more informative: a boolean records that somebody
opted in, a reason records *what they were doing*.

⚠️ **This must be applied to `rollback()` in the same change.** F2 is an instance of a failure I
have shipped before: fixing one writer of a column and leaving the other. `rollback()` is the second
writer and must set `serving_mode` from the same rule, or the gate is decorative on the path that
actually reaches it.

Then delete the `generation_mode` branch at `recall_mcp/server.py:627`. The server should build a
`GenerationStore` **always**, and let the per tenant readiness answer decide what happens, which is
where the answer lives anyway. A tenant with no generation gets `INDEX_NOT_READY` from
`recall/readiness.py:110`, unchanged, which is the correct answer and is already implemented.

### Q: Is there an honest install time calibration binding that does not need a full build?

**Yes, and `recall/cli.py:2032` was right to refuse the version that would have been dishonest.**

The comment says: "Resolve that by deciding where install-time calibration binds, not by reinstating
the line below." My answer is that **there is no honest process global calibration and there should
not be one.** A threshold is a property of the tuple (tenant, generation, pipeline, corpus, query
set). Any artifact that does not carry all five is `legacy_unbound`, and the strict policy is right
to refuse it. So `recall calibrate`'s process global JSON should be deprecated rather than re read.

The install time binding is the existing generation bound path, with the human removed from the one
step that made it impractical:

1. Adopt or build a generation (section 4).
2. `recall/wizard/queryset.py` generates the labelled set from the corpus itself. Already written,
   already reasoned about, not wired up.
3. `CalibrationRepository.calibrate(generation_id, ...)` fits and binds. This already works against
   a `ready` generation (F1), so it does not require promotion and does not re embed anything: it
   runs queries against vectors that already exist.
4. The artifact records `query_set_provenance: "generated"`, and a generated set certifies to
   `PROVISIONAL`, never `CERTIFIED`.

Step 4 is the honesty mechanism and it is the part I would defend hardest. The threshold is a real
measurement of a real corpus, which 0.5 is not. It is weaker evidence than a human labelled set,
because the questions were invented by the same process that is being scored. Both facts belong in
the wire payload. Suppressing the second would make this worse than the status quo, because it would
launder a generated measurement into the word "certified".

### Q: Should a first run adopt a legacy `chunks` corpus without re embedding?

**Yes, and F3 says the binding can be established rather than asserted, which is the condition the
report correctly set.**

The trap named in the brief is real: writing `recall_chunks_v1` rows from `chunks` rows would
fabricate `source_sha256` and `object_version_id`, which is exactly what `recall/lineage.py:269-274`
refuses for `file://` objects. The escape is that the legacy metadata was written at embed time and
can be checked against the world now.

`recall generation adopt --from-chunks` proceeds per source:

1. Read `metadata->>'content_hash'`. Absent means **not adoptable**, full stop. There is no
   inference available, and `store.source_content_hashes` already reports these as `""`.
2. Read the file at `metadata->>'file'`. Missing or unreadable means not adoptable.
3. sha256 the bytes now on disk. Not equal to `content_hash` means the file changed since indexing:
   **not adoptable**, re embed that source or drop it.
4. Compare `metadata->>'embedding_profile'` to the configured embedder's profile id. Not equal means
   the vectors came from a different model: not adoptable.
5. Only then copy the vector, setting `source_sha256 = content_hash` and
   `object_version_id = content_hash`, which is precisely the rule `recall/lineage.py:269` enforces
   for local files.

Two properties this gives, stated as the `file://` comment states its own:

- **What it buys is detection of divergence, never prevention.** A file rewritten between step 3 and
  step 5 is not caught. That is the same guarantee a `file://` manifest already offers, and the
  design should not claim more.
- **`content_hash` proves which bytes were read, not which chunker ran.** So adoption also draws a
  **pipeline attestation sample**: re embed N chunks chosen at random from the enumerated population
  and compare to the stored vectors. Passing is evidence the recorded pipeline identity is the one
  that ran. This costs seconds. Its bar and its failure mode are pre registered.

An adopted generation is `provisional` by construction, carries
`unverified_reason: "adopted from legacy chunks"` plus the four way adoption census in its manifest,
and can only reach `certified` through a normal calibration against a human labelled set. Adoption
never manufactures certification; it manufactures a *starting point* that a calibration can bind to.

## 4. The resulting first run

```
recall init <dir>
```

One command, no environment variables, and every step is refusable with a real reason:

1. Index into `chunks` if needed, or detect an existing corpus.
2. Adopt into a generation, verifying every source. Report the census: verified, changed, missing,
   unadoptable. Re embed only what failed verification.
3. Generate a query set. Calibrate. Publish.
4. Promote with the certification gate. A generated query set lands `serving_mode = 'provisional'`.
5. Write `.mcp.json`.

Search then works, in strict mode, with `trust_state: "provisional"` on every result and a threshold
that was measured on the user's own corpus. `RECALL_TRUST_MODE` is not set, and does not need to be.

## 5. What I would measure before believing any of this

Pre registered in `docs/preregistrations/2026-08-18-uncalibrated-first-run.md`, predictions written
before the first run:

- **Adoption fidelity.** At least 95 percent of a remote read only Postgres `memory` tenant adopts with a
  verified binding; failures dominated by edited files rather than absent metadata; attestation
  sample reproduces to cosine 0.9999; under 5 minutes against ~10 hours for a full build.
- **Provisional threshold quality.** The generated threshold beats the 0.5 constant on false
  confidence rate by at least 0.10 absolute on a **held out human labelled** set, and does not reach
  the certification bar.

If the second is falsified, the honest response is to keep the constant and label it, not to ship a
status that dresses it up.

## 6. What this design does not do

- It does not touch policies 1, 2 or 4. Ingestion source, auth mode and the learned sparse leg stay
  process wide, because they genuinely are. `RECALL_ENV` survives for those and only those, and
  should be renamed per axis so that a future reader cannot re overload it.
- It does not lift the strict default. Strict stays the default and refusal stays the behaviour when
  there is nothing to stand on. `provisional` is a *third* answer, not a relaxation of the first two.
- It does not make a generated query set equivalent to a labelled one, and says so on the wire.
- It does not fix the 0.5 constant's remaining user, `RECALL_TRUST_MODE=development`, which stays as
  the explicit escape hatch for someone who wants text out of an uncalibratable corpus.
