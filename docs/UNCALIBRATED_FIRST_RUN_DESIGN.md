# Serving a corpus before it has a calibration

**Status:** design, 2026-08-18; **partly implemented, 2026-08-20.** This used to say "Nothing here
is implemented", and that line survived one commit past the point where it stopped being true, which
is how the certification gate came to contradict section 6 decision 2 without anybody noticing.

What has landed, and where:

| From this document | Landed as |
|---|---|
| F2, the ungated activation window | `promote` requires CERTIFIED in production (`recall/generations.py`, `require_certified_for_production`) |
| Section 6, decision 2 | `rollback(*, provisional_reason=...)` never refuses, and records the target's status |
| The artifact attestation of 6c | `embedder_artifact_digest` / `artifact_tree_sha256` in `recall/embeddings.py` |

What has **not**: the per-tenant `serving_mode` this document argues for, and strict refusal at read
time. Everything below is otherwise unchanged, and remains a proposal with measured
support for two of its claims. Section 6 answers the eleven questions an audit found it had left
open; two of those answers change the design and one dissolves a question that rested on a
falsified premise. Sections 6c and 6d then design the two attestations that were still missing.
Measurements are pre registered in `docs/preregistrations/`:
`2026-08-18-uncalibrated-first-run.md`, `2026-08-18-chunker-attestation.md` and
`2026-08-18-extraction-attestation.md`. Source citations are re measured whenever this file is
edited, against the commit the edit lands on; see the note in the first pre registration about
how fast they drift.

The goal is unchanged: someone must be able to index and search before they have labelled queries.
What follows argues that `RECALL_ENV=development` is the wrong mechanism for that, not because it is
badly implemented, but because it answers a question about a **process** when the question is about
a **tenant**.

## 1. What I confirmed in the code

| Claim | Site | Verdict |
|---|---|---|
| Server builds `GenerationStore` only in production <!-- cite-anchor: if generation_mode: --> | `recall_mcp/server.py:759` | confirmed. 🔁 **Line corrected 2026-08-25, and it was wrong before it drifted.** It read `:678`, which was a `retrieval_profile` logging argument, not the branch; the anchor is added so the next edit above it moves the pointer rather than silently invalidating it |
| Missing `generation_id` degrades to `"legacy"` | `recall_mcp/service.py:999` | confirmed. A second site uses the same default but maps it to `None` immediately after, so the two do not behave identically |
| `promote()` refuses in production, needs a flag otherwise <!-- cite-anchor: def promote --> | `recall/generations.py:1239` | 🔁 **no longer true.** Confirmed when written. `promote()` now admits a generation whose published calibration certified and is still bound, and `unsafe_development` is refused in production rather than being the other way through. See F2 |
| No generation means `INDEX_NOT_READY` **at the readiness endpoint** | `recall/readiness.py:116` | confirmed, but this is **not** the search path. See Q2 |
| `calibration = None` is deliberate, and names an open design question | `recall/cli_commands/index_search.py:406-418` | confirmed |
| Legacy `chunks` has no `source_sha256` **column** | `recall/store.py:376` (`DEFAULT_TABLE`) vs `recall_chunks_v1` | confirmed as stated, and **narrower than "nothing to reuse"**: the metadata carries `content_hash`, which is what F3 is about |

### Four findings that change the available answers

**F1. Promotion is not required, for either calibration or serving.**
`CalibrationRepository._generation` accepts states `{"ready", "active", "retired"}`
(`recall/calibration_v2.py:495`). `GenerationStore.pin_generation` accepts the same three
(`recall/generation_store.py:150`). And `SERVABLE_ACTIVE_STATES = frozenset({"ready", "active"})`
(`recall/control_plane.py:35`), so the enterprise control plane **already treats `ready` as
servable**. What `promote()` adds over calibration and serving is that it sets
`recall_tenant_state.active_generation_id`, the *default selection* rather than the permission, and
even that is not exclusive to it (see F2).

**F2. The promotion gate does not hold the invariant it claims to hold.**
`rollback()` (`recall/generations.py:1321`) <!-- cite-anchor: def rollback --> writes the same `active_generation_id` column with **no
environment check and no `unsafe_development` flag**, and its target may be in state `ready`
(`recall/generations.py:1365`) <!-- cite-anchor: GenerationState.RETIRED -->. So "no ungated generation becomes active in production" is not a
property this system has. `promote()`'s message says the refusal stands "until certification gates
land", which is accurate: it is a placeholder, not a safety property.

🔁 **Closed 2026-08-20**, and then **partly reopened and re-closed the same day**, which is the more
useful record.

`promote()` now requires a calibration that is published, that certified, and whose pipeline and
corpus fingerprints still match the generation — the gate its message had been promising.

⛔ **The first version of that fix also gated `rollback()`, and that was wrong.** The reasoning was
this finding's own words: building the gate into `promote()` alone leaves the window open beside
the new door. But section 6 decision 2, in this same document, had already settled the question the
other way, and the fix was written without reading it. Three ways the refusal bit, each found
independently in the audit: `forget()` bricked rollback permanently by making every calibration
stale; upgrading bricked it, because every generation an existing install serves was promoted under
`development` and has no published calibration; and there is no override to reach for, so the
remaining routes are a mid-incident recalibration or flipping `RECALL_ENV`.

**The resolution is that this finding named its invariant slightly wrong.** "No ungated generation
becomes active in production" is not the property worth having, because rollback is the incident
path and a refusal there buys nothing an operator will not route around. The property is **"no
generation becomes active without the operator being told what they are activating"**, and rollback
satisfies it by *reporting*: `generation_rolled_back` carries the resolved calibration status and an
optional `provisional_reason`. Prevented, no; hidden, never.

Pinned by `tests/test_calibration_v2.py::test_rollback_never_refuses_and_records_what_it_activated`,
which replaced the test that pinned the reversed behaviour, and by
`::test_a_rollback_onto_a_certified_target_records_no_provisional_reason`, so an undegraded rollback
cannot start looking degraded.

**F3. The legacy `chunks` table records enough to establish a binding, not merely assert one.**
It has no `source_sha256` column, but `recall/index.py:955` stamps into every chunk's metadata:
`content_hash`, `index_fingerprint`, `embedding_profile`, `context_mode`, `context_version`, `ord`
and `file`, all written **at embed time**, so checking them is verification rather than
reconstruction.

🔁 **Corrected 2026-08-18 by measurement.** An earlier version of this paragraph said
`embedding_profile` "is the identity of the embedder that produced the vector". **It was not**, at
the tree this was measured against: the fallback returned the literal string
`bge-small-symmetric-v1` for any model without a registered profile, so a 1024 dimensional corpus
carried a 384 dimensional profile's id.

🔁 **Fixed upstream, 2026-08-18, by #370**, which this measurement prompted. `_fallback_profile_id`
(`recall/embeddings.py:982`) now derives `unregistered__{model}__{dimension}__{kind}`
(`recall/embeddings.py:982`) instead of claiming a registry id it does not have.

⚠️ **That does NOT restore `embedding_profile` as an adoption check, and the design still must not
use it.** Every corpus indexed *before* #370 carries the old literal, which is exactly the
population an adoption path exists to read. A fix to the writer does not retroactively repair rows
already written. Only `content_hash` is load bearing here, and the accessor that returns it is
`PgVectorStore.source_raw_hashes` (`recall/store.py:2471`), **not** `source_content_hashes`
(`recall/store.py:2453`), which coalesces `index_fingerprint` first and therefore returns the defective identifier.

⚠️ **`content_hash` is media type dependent since `bd582316`.** A markdown source is hashed as
decoded, newline normalised, `_strip_nul` text re encoded as UTF-8 (`recall/index.py:822`
and `recall/index.py:803`); any other media type is hashed as raw `source_bytes` (`recall/index.py:832`). Any adoption path must branch the
same way, or it will refuse every markdown file with CRLF or a BOM.

**F4. The first run wizard is half built and already solves the hardest part.**
`recall/wizard/queryset.py`'s `generate_offline` produces a labelled query set from the corpus with
no human and no network. (The module also offers an LLM generator, which uses both; only the
offline one has that property, and only it was measured.) Its docstring names the reason it exists:
the hand labelled set "is the step that makes calibration too hard to be worth doing, and it is the
step a first-run wizard has to remove". It is not wired into the CLI.

## 2. The diagnosis

`RECALL_ENV` is one string carrying at least six unrelated policies:

1. **Ingestion source.** Production refuses local filesystem indexing (`recall_mcp/service.py:2506`, `recall/cli_commands/index_search.py:208`).
2. **Auth.** Production refuses static bearer tokens (`recall_mcp/auth.py:376`).
3. **Store class.** Production selects `GenerationStore`, at **three** sites, not one:
   `recall_mcp/server.py:760` <!-- cite-anchor: if generation_mode: -->, `recall/cli_commands/index_search.py:298` <!-- cite-anchor: generation_mode -->, and the `generation_mode` parameter threaded
   into `StoreRegistry` (`recall_mcp/stores.py:154`), whose value is `generation_mode and not
   enterprise` and therefore also encodes the control plane interaction.
4. **Retrieval legs.** Production disables the learned sparse leg (`recall/retriever.py:423`). <!-- cite-anchor: wants_learned -->
5. **Promotion permission.** Production once refused `promote()` outright; it now requires a published, certified, still-bound calibration (`recall/generations.py:1239`) <!-- cite-anchor: def promote -->. 🔁 Updated 2026-08-20.
6. **Generation creation.** Production requires a verified pipeline identity and refuses
   `allow_unverified` (`recall/generations.py:398`, `recall/generations.py:409`), which an adopted generation cannot satisfy with
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

So: record `serving_mode` beside the generation pointer, with three values.

| `serving_mode` | Means | `trust_state` on the wire |
|---|---|---|
| `certified` | A published, certified calibration bound to the active generation, fitted on a **human labelled** query set. | `trusted` |
| `provisional` | The same, but the query set was **machine generated**. The statistics are real; the questions were invented by the process being scored. | `provisional` |
| absent | No active generation. | `refused` |

`provisional` is a new `TrustState`. It is deliberately **not** `degraded`, because `degraded`
means "we ran with a number bound to nothing", and it is deliberately **not** a new
`CalibrationStatus`: see decision 5, which is the correction the measurement forced. Certification
is a *statistical* verdict and provenance is a *separate axis*; the measured result was that a
generated set passes the statistical bar, so folding provenance into `CalibrationStatus` would
have conflated two independent facts.

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
design does not make. `rollback()` today takes no arguments (`recall/generations.py:1321`), so it
cannot record a reason, and its target's calibration may have gone stale or superseded in the
meantime. Under the certification rule it would either refuse, which blocks incident recovery
exactly when it is needed, or grant `provisional` with no reason, which is the weakness this design
uses to argue against `unsafe_development=True`. **Resolved in decision 2.**

Then delete the `generation_mode` branch. Two corrections to an earlier version of this section,
both found by audit:

⛔ **It is three sites, not one** (policy 3 above), and the `StoreRegistry` parameter cannot simply
be deleted because it also encodes `and not enterprise`. Naming only `server.py:627` would have been
the same "fixed one writer, left the other" failure this design levels at `promote()`.

⛔ **"A tenant with no generation gets `INDEX_NOT_READY`, already implemented" is false on the
search path.** `readiness.py:110` is a different entry point that receives `generation_id` as an
argument. On search, `GenerationStore.generation_binding()` raises `NoActiveGeneration`, which is
swallowed by the broad `except Exception` in `trusted_search` and re raised as
`DEPENDENCY_UNAVAILABLE` (`recall/trust.py:748`), whose advice text calls that condition "an
outage, not an empty result". Mapping `NoActiveGeneration` to `INDEX_NOT_READY` is therefore a **prerequisite** of
this change, not a consequence of it.

⛔ **"GenerationStore always" breaks development ingestion over MCP.** The `recall_index` tool writes
through the same store object the branch selects, and `GenerationStore.upsert` raises
`ImmutableGenerationError`. Store selection has to be split by direction, a read store chosen per
tenant readiness and a separate legacy write store for ingestion, or the MCP index tool has to be
redirected. Section 6's claim that policy 1 is untouched does not survive without that split.

### Q: Is there an honest install time calibration binding that does not need a full build?

**Yes, and `recall/cli_commands/index_search.py:418` was right to refuse the version that would have been dishonest.**

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
checksum covered schema change. **Resolved in decision 10.**

### Q: Should a first run adopt a legacy `chunks` corpus without re embedding?

**Yes, and F3 says the binding can be established rather than asserted, which is the condition the
question sets.**

Writing `recall_chunks_v1` rows from `chunks` rows would fabricate `source_sha256` and
`object_version_id`, which is what `recall/lineage.py:292` refuses for `file://` objects. The escape
is that the legacy metadata was written at embed time and can be checked against the world now.

`recall generation adopt --from-chunks` proceeds per source:

1. Read `metadata->>'content_hash'` via `source_raw_hashes`. Absent means **not adoptable**.
2. Read the file at `metadata->>'file'`. Missing or unreadable means not adoptable.
3. Re derive the hash **exactly as the indexer does for that media type**: decoded, newline
   normalised, `_strip_nul` text for markdown (`recall/index.py:822`, `recall/index.py:803`), raw
   `source_bytes` otherwise (`recall/index.py:832`). Not equal means the file changed since indexing: not adoptable.
4. 🔁 **Corrected 2026-08-18 by measurement.** This step originally compared
   `metadata->>'embedding_profile'` to the configured embedder's profile id. **That check does not
   work** (F3), and `index_fingerprint` inherits the defect because `_index_fingerprint` hashes the
   same value. 🔁 **Corrected: #381 changed that.** `_index_fingerprint` now hashes
   `embedding_profile(embedder).fingerprint()` (`recall/index.py:516`), which covers model name
   and dimension, so a fingerprint computed *today* does distinguish models. It does not help
   here: every fingerprint **already stored** was computed under the old formula, and those are
   the rows adoption reads. Neither stored field may gate adoption. The check is the
   attestation sample below.
5. Only then copy the vector, setting `source_sha256 = content_hash` and
   `object_version_id = content_hash`, the rule `recall/lineage.py:292` enforces for local files.

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

## 6. Decisions on the eleven questions

Answered 2026-08-18. Each states the decision, the reason, and what it costs. Two of them
(1 and 9) change the design rather than merely completing it, and one (5) dissolves rather than
resolves: the question rested on a premise the Q2 measurement had already falsified.

### 1. The enterprise control plane: same rule, and the mode moves to the pointer

**Decision: `serving_mode` is a column on the row that holds the generation pointer, not on the
tenant.** That means both `recall_tenant_state` and `recall_tenant_routes.active_generation`. One
function, `activation_decision(conn, tenant, generation_id, provisional_reason) -> ServingMode`,
computes it, and **all three activation paths call it**: `promote()`, `rollback()`, and
`ControlPlane.cutover()` / `set_route()`.

**Why the pointer and not the tenant.** A mode stored on the tenant while the *route* selects the
generation can drift from the generation it describes, and `StoreRegistry._get_generation` resolves
the route first (`recall_mcp/stores.py:150`). Putting the mode in the same row as the pointer makes
drift unrepresentable rather than merely unlikely. This is the same reasoning F2 applies to
`promote()` and `rollback()`, taken one step further: **do not enumerate writers by discipline when
you can make the datum travel with the thing it describes.**

**What it costs.** A migration on two tables, and `cutover()` gains a parameter. `cutover()`'s
existing gate checks parity and outbox lag, not calibration, so this genuinely adds a refusal it did
not have. That is the point of the question.

### 2. `rollback()` takes a reason and never refuses

**Decision: `rollback(*, provisional_reason: str | None = None)`, and it never refuses on
certification grounds.** When the target has no certified calibration it activates as `provisional`
and records the reason, defaulting to `"rollback: incident recovery"` when the caller gives none.

**Why not refuse.** Rollback is the incident path. A gate that blocks recovery precisely when
recovery is needed is worse than serving a `provisional` answer that says so on the wire, and an
operator facing a bad generation will route around a refusal in a way nobody audits. **Refusing here
would trade a visible degradation for an invisible workaround.**

**The honest cost:** rollback can silently downgrade a tenant from `certified` to `provisional`.
That is correct, and it must be **reported**: the audit event records the downgrade, and
`recall status` shows it. Prevented, no; hidden, never.

### 3. `serving_mode` is ADVISORY. The live resolver stays authoritative

**Decision: `CalibrationRepository.resolve()` remains the runtime answer. `serving_mode` records
what was decided at activation time and why.** Where they disagree, the resolver wins and the
disagreement is itself reportable.

**Why.** `resolve()` re-derives the lineage comparison on every query, which is what catches a
`forget()` that rewrote `corpus_fingerprint` (`recall/generations.py:1497`) or a `publish()` that
superseded the artifact (`recall/calibration_v2.py:978`). A cached mode cannot catch either.
Making it authoritative would require every current and future invalidator to update it, which is
exactly the growing-enumeration failure this design criticises in F2. **A cache that must be
invalidated by an open-ended set of writers is a bug with a schedule.**

**What is therefore NOT claimed:** `serving_mode` is not a fast path and must not be read to skip
`resolve()`. It buys operator visibility and an audit record, which is what the question about
environment variables was really asking for.

### 4. Add `provisional` to the reasoning whitelist, and bump the API version

**Decision: `recall/reasoning.py:1637` accepts `{trusted, degraded, refused, provisional}`, and
`REASONING_API_VERSION` goes 1 → 2.** The several `!= "trusted"` comparisons keep their current
behaviour and become an explicit named set, `_CERTIFIED_STATES = frozenset({"trusted"})`.

**Why the comparisons keep their behaviour.** `require_certified_evidence` means certified. A
provisional result must not satisfy it. The change is that the intent is *stated* rather than
inferred from a string comparison, so the next person adding a state has to make a decision instead
of inheriting one by default.

**Why the version bump.** A client written against version 1 may enumerate the three states. Adding
a fourth is additive for a client that treats unknown states as untrusted and breaking for one that
does not, and the contract never told them which to do. Version 2 says it: **unknown `trust_state`
must be treated as untrusted.**

### 5. Dissolved: `PROVISIONAL` is not a `CalibrationStatus`

**Decision: do not add it. `code_for_status` and `_STATUS_CODES` are untouched, so the stable
failure-code API is untouched and the fail-closed default stays exactly as it is.**

The question assumed provisional was a *weaker certification*. The Q2 measurement falsified that:
a generated query set **passes** `Calibration.certified`, and `publish()` accepts the artifact
(`recall/calibration_v2.py:980` only refuses `not artifact.certified`). So certification and
provenance are **two independent axes**, and the original framing conflated them.

So: `CalibrationStatus` keeps meaning "is this artifact bound and statistically sound". Provenance
lives on the artifact as `query_set_provenance` (decision 10). `trust_state` is then computed from
the pair, which needs no new failure code:

```
failure_code is not None              -> refused / degraded, exactly as today
None and provenance == human          -> trusted
None and provenance == generated      -> provisional
```

⚠️ **One thing this does NOT give for free, and the design was wrong to imply it did.** A strict
policy must still decide whether it *accepts* a provisional answer. `TrustPolicy` gains
`accept_provisional: bool = False`, so strict-by-omission still refuses. `recall init` writes the
acceptance explicitly into the local configuration rather than defaulting it on. That keeps the
first run free of manual environment fiddling **without** a fail-open default, and it matches
`TrustPolicy.development()`'s existing rule: "Explicit opt-in for local workflows. Never reachable
by omission."

### 6. The adoption column mapping, stated

**Decision:** every `recall_chunks_v1` column has a named origin, and anything without one makes the
source not adoptable.

| v1 column | Origin | Note |
|---|---|---|
| `source_uri` | `file://` + the absolute resolved path | Root comes from the project's recorded index root, not guessed |
| `object_version_id` | `metadata->>'content_hash'` | The `file://` rule: version_id **is** the digest (`recall/lineage.py:292`) |
| `source_sha256` | `metadata->>'content_hash'` | Verified against disk first |
| `chunk_ordinal` | `int(metadata->>'ord')` | **Required alongside `content_hash` in step 1**; absent means not adoptable, matching `_write_source` (`recall/generations.py:641`) |
| `text`, `embedding`, `metadata` | copied | The vectors are the whole point |
| `tsv` | **recomputed**, never copied | `to_tsvector(<pipeline fts_language>::regconfig, text)`, exactly as `_write_source` does. Legacy is `GENERATED ALWAYS ... 'english'`, so a copy is right only by coincidence when the adopted pipeline declares English |

`recall_generations`' NOT NULL columns come from a **synthesized manifest built from the verified
set**: one `ManifestObjectV1` per adopted source with `uri`/`sha256`/`version_id` from the
verification, `size` and `media_type` from disk. This is not fabrication, and the distinction is the
same one `lineage.py` draws: the manifest records **what was checked against disk at adoption
time**, which is precisely the guarantee a `file://` manifest offers anywhere else.

### 7. No new state. Adoption uses the existing lifecycle

**Decision:** `create()` (→ `building`) → adopt-copy → `validate()` (→ `ready`) → `promote()`.

**Why not `legacy_unverified`,** which already exists in the state CHECK
(`recall/migrations/sql/0008_generation_foundation.sql:7`): **nothing downstream accepts it.**
`CalibrationRepository._generation` takes `{ready, active, retired}` and `pin_generation` the same,
so an adopted generation parked there could never be calibrated, which defeats the entire purpose.
A state that no consumer accepts is not a lifecycle stage, it is a dead end.

`validate()` then works unmodified, because it compares chunk `source_uri` / `source_sha256` /
`object_version_id` against the manifest. ⚠️ **Say plainly what that check is worth here:** both
sides derive from the same verified digest, so validation confirms the copy is *internally
consistent*, not that the bytes were re-checked. The disk check in step 3 is the evidence; this is
an integrity check on the copy.

⛔ **Constraint this surfaces:** `create()` refuses `allow_unverified` in production
(`recall/generations.py:398`), and an adopted generation is unverified by construction. **Adoption
is therefore development-only under the current gate**, which is policy 6 in section 2 and is a
second reason that gate should move off the environment. The first-run design depends on it.

### 8. One transaction, and failure uses the existing compensating path

**Decision:** one transaction per generation for the copy, and any failure calls
`GenerationManager.fail(generation_id, reason)`.

**Why this needs nothing new:** `fail()` moves `building`/`validating` → `failed`
(`recall/generations.py:824`), and `gc()` already reclaims `failed`. The abandoned-generation
problem the question raised only exists for a generation left in `building`, which is exactly what
this prevents. `validate()` already wraps itself this way, so adoption inherits a tested shape
rather than inventing one.

**Sources that fail verification:** the generation still adopts, and the failures are re-embedded
into the same generation before `validate()`. Refusing wholesale on one changed file would make
adoption useless on any live corpus. What is refused wholesale is an **attestation** failure, which
is different in kind: see decision 9.

### 9. The attestation sample size, and the measured sample was too small

**Decision:** `n = ceil(ln(alpha) / ln(1 - p))`, sampled **by source** and not by chunk, with
`alpha = 0.05` and `p` the smallest contaminated fraction worth detecting.

| smallest fraction detectable | n at 95% | n at 99% |
|---:|---:|---:|
| 20% | 14 | 21 |
| 10% | 29 | 44 |
| 5% | 59 | 90 |
| 2% | 149 | 228 |
| 1% | 299 | 459 |

⛔ **This retires a number the measurement reported.** The 20-chunk sample detects only a
contaminated fraction of **13.9% or larger** at 95% confidence. As the *sole* embedder check
(which decision 4 of the adoption path makes it) that is not adequate, and the pre-registration's
"20 of 20" should be read as a feasibility and cost result, not as coverage. **Default `p = 0.05`,
so n = 59.** At the measured ~0.05 s per chunk that is about three seconds.

**Stratify by source, not chunk.** Contamination arrives per source, since it is a re-index of some
files under a different model. Sampling chunks uniformly over-weights long documents and can draw
all 59 from a handful of sources.

**Abort rule: a single sample below the bar aborts the whole adoption.** Not a majority. One failure
means the recorded pipeline identity is wrong for at least one source, and nothing in the legacy
metadata says which other sources share its provenance.

**Bar: cosine ≥ 0.9999, and if the platform cannot reproduce it, refuse rather than lower it.** A
lowered bar cannot distinguish "different ONNX execution provider" from "different model", which is
the only thing the check exists to tell apart. Measured margin: 20 of 20 at 1.0000 against an
off-diagonal control of 0.709 max.

### 10. Provenance is ATTRIBUTABLE, not unforgeable

**Decision:** `query_set_provenance` is stamped **server side from the entry point that stored the
query set**, is part of the artifact's immutable payload and therefore inside its checksum, and
takes three values, not two.

| value | Written when |
|---|---|
| `generated` | The set came from `recall/wizard/queryset.py` |
| `human_asserted` | A file was supplied via `recall calibration calibrate --queries FILE` |
| `human_reviewed` | As above, plus a recorded reviewer identity and timestamp |

Only `human_reviewed` yields `trust_state: trusted`. `human_asserted` yields `provisional`, the
same as `generated`.

**Why not claim unforgeability.** The operator owns the database; any local scheme is defeatable by
writing the row directly. Claiming otherwise would be the exact failure this project's `lineage.py`
refuses for `file://` objects, where the comment says plainly that what a local path buys is
**detection of divergence, never prevention**. So the honest guarantee is: **the default is safe,
the claim is attributable, and asserting a human label is a distinct recorded act rather than the
absence of a flag.** That is the same standard `recall rewrite` already applies, where nothing
reaches corpus metadata without a named human.

**Cost:** a column on `recall_calibration_query_sets`, an artifact-version bump, and a migration.

### 11. Additive, with the contracts and the pinned test updated

**Decision:** the change is additive for clients that treat unknown states as untrusted, and the
contract is amended to *require* that. Concretely: `docs/USING_WITH_CLAUDE.md`,
`docs/REASONING_CONTRACT.md`, and the test pinning `trust_state in {"trusted", "degraded"}` all move
to the four-state set, and `REASONING_API_VERSION` 2 (decision 4) is what a client checks to know
which set to expect.

**The pinned test is not an obstacle, it is the mechanism.** It exists so that adding a state is a
deliberate act with a visible diff, which is what is happening here.

## 6b. What remains genuinely open

Two things, and neither blocks implementation:

- **Whether enterprise deployments want decision 1 at all.** Adding a certification gate to
  `cutover()` is a real behaviour change for an existing operator, and that is a product call rather
  than an engineering one. The engineering answer is that the gate belongs there; whether to ship it
  default-on for existing tenants is not mine to make.
- 🔁 **The chunker gap is now designed and measured. See section 6c.** It was open when this
  section was written; the measurement is
  `docs/preregistrations/2026-08-18-chunker-attestation.md`.

## 6c. The chunker attestation

Decision 9 sizes the *embedder* check. This is its counterpart, and it is a different shape than
expected, because the measurement said so.

### It is an IDENTIFICATION, not a verification

**The legacy table records no chunker at all**: not the algorithm, not `max_chars`, not `overlap`.
`_index_fingerprint` carries no chunker CONFIGURATION either (`recall/index.py:464`), which is why
re indexing a corpus does not repair a chunker change: the skip guard reports it unchanged.

🔁 **Corrected 2026-08-18 after `79a0d6ed`, which is the commit that made the previous wording
wrong.** This used to read "`_index_fingerprint` has no chunker term either". #381 widened that
fingerprint to hash the whole `EmbeddingProfile`, which covers `chunker_version`
(`recall/embeddings.py:414`), so a field of that name is now in the hash. It is inert: it belongs to
the EMBEDDING profile, is defaulted to `chunk-text-v1` at both definitions and set by nothing else,
and the `Indexer`'s actual chunker (`recall/index.py:584`) never reaches it. Measured against
`79a0d6ed`, one file and one embedder, varying only the chunker: `chunk_text(800, 80)` gives one
chunk, `chunk_text(60, 10)` gives four, `chunk_code` gives one, and **all three produce the
identical index fingerprint**. So the conclusion below is untouched and only the sentence needed
narrowing. ⚠️ Forward hazard: `chunker_version` is now key material, so anything that starts setting
it per chunker turns a chunker change into a forced re embed.

So there is no stated value to check. The attestation **re derives the body with `parse_frontmatter`
(`recall/frontmatter.py:190`), re chunks it with each of a fixed candidate set, and compares the
result against the stored chunks ordered by `ord` as an exact string list equality.** What it
produces is the chunker's identity, or a refusal.

⛔ **The candidate set is fixed before the run and is never widened to make something fit.** A
search that keeps broadening until a configuration reproduces the corpus will always succeed, and
what it finds is a configuration that never ran. That is curve fitting wearing a verification's
clothes. When no candidate reproduces a source, the answer is "not identifiable", and the source is
re embedded rather than adopted.

### It is EXHAUSTIVE, not sampled, and that is the asymmetry worth naming

Measured: **0.16 s to re chunk all 1,058 usable sources** with one candidate, against 1.82 s merely
to hash the same corpus. All four candidates ran in under a second.

**So the chunker check covers every source, and needs no sample size rule.** The embedder
attestation needs one (decision 9) only because inference is expensive. Chunking is pure string
work, so where the embedder check can give a statistical bound, the chunker check gives a complete
answer. Two checks, two costs, two different guarantees, and the design should not force them into
one shape.

### Outcomes

| Candidates reproducing ALL usable sources | Meaning | Action |
|---|---|---|
| exactly one | Identified | Record it as the `ChunkerIdentity`, **verified rather than asserted** |
| more than one | Observationally equivalent on this corpus | Record the set, adopt under the canonical member, and say the identity is under determined |
| none | Not identifiable | Refuse adoption; re embed |

Measured on the memory corpus: exactly one, `chunk_text(max_chars=800, overlap=80)`, reproducing
**1,058 of 1,058**. The other three reproduced 50.57, 4.06 and 2.93 percent, so the comparison
discriminates rather than accepting everything.

### Per source, because a corpus can hold more than one chunker

⚠️ **This is the consequence the measurement forced, and it is not obvious.** Since
`_index_fingerprint` carries no chunker configuration (see the correction above: as of `79a0d6ed`
it carries a `chunker_version` string that no chunker change moves), `recall index` **skips** a file
whose content and embedder are unchanged even when the chunker has changed underneath it. An incrementally built corpus can
therefore legitimately contain chunks from several chunker eras, and that is the expected result of
any chunker change rather than an exotic case.

So the attestation runs **per source** and the outcomes above are evaluated over the whole corpus:

- all sources identify the **same** chunker: adopt the generation with that identity;
- sources identify **different** chunkers: the corpus cannot become one generation with an honest
  `PipelineIdentity`, which carries exactly one chunker. Adopt the majority set and **re embed the
  remainder**, which is the same disposition step 3 already applies to a source whose bytes changed.

### Failure disposition differs from the embedder check, deliberately

A single embedder attestation failure **aborts the whole adoption** (decision 9), because nothing in
the legacy metadata says which other sources shared the failing one's provenance. A chunker mismatch
is **local and diagnosable**: the source is named, its bytes are already verified, and re embedding
it is cheap. So a chunker mismatch rejects a source, not the run. **A systematic mismatch, more than
half the corpus, aborts** and reports that the candidate set does not describe this corpus.

### What it does not prove

- **Observational equivalence, not identity.** A different implementation producing identical output
  on these sources is indistinguishable here. Same standard as cosine 1.0 for the embedder, and it
  should be claimed no more strongly.
- 🔁 **Markdown only, and the non markdown case is now designed in section 6d.** It was out of
  scope when this section was written; the measurement is
  `docs/preregistrations/2026-08-18-extraction-attestation.md`.
- **The body rule can move under it.** `parse_frontmatter` changed once, and the fix carries a
  version marker (`_BODY_RULE_VERSION`, `recall/generations.py:148`) precisely because the same bytes then yielded a
  different body. A corpus indexed before such a change reports a chunker mismatch when the real
  difference is upstream of the chunker. The attestation should therefore report the body rule
  version alongside its verdict, so the two causes are distinguishable.

1. **The enterprise control plane is a second activation surface.** `ControlPlane.set_route()` and
   `cutover()` (`recall/control_plane.py:803`) write `recall_tenant_routes.active_generation`, and
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
   mapping: `recall/reasoning.py:1637` rejects anything outside `{trusted, degraded, refused}`, on a
   versioned API whose version is unbumped, plus several `!= "trusted"` comparisons that would
   silently downgrade.
5. **The strict gate is binary on `CERTIFIED`.** `code_for_status` returns a failure code for
   everything else via a fail closed default (`recall/trust_policy.py:110`), so a new
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

## 6d. The non markdown extraction attestation

Section 6c attests the chunker for markdown. This is the counterpart for everything else, and it
reaches a **weaker** conclusion on purpose.

### Why it cannot be the same check

Markdown body derivation is pure Python inside this repository, so a chunker mismatch is
diagnosable: the code that would differ is versioned by the repo. Extraction is not.
`extract_document` (`recall/extraction.py:169`) dispatches to **six third party libraries** and, for
five suffixes, to an **external LibreOffice binary** (`_extract_with_libreoffice`,
`recall/extraction.py:743`). Those libraries
are declared with open lower bounds in an optional extra (`pdfplumber>=0.11` and friends), and
LibreOffice is not a Python dependency at all.

⛔ **So "the same recall version" says nothing about what extraction produces**, and a re extraction
mismatch is ambiguous between four causes rather than being evidence of a defect.

### Measured: comparison is possible, which is necessary and not sufficient

17 of 17 formats, including `.pdf` and the five LibreOffice ones, are **byte identical across two
independent processes**, and **none** leaks a temporary path, working directory, user name or date
into its output. So a re extraction comparison is meaningful rather than noise.

⚠️ **That is the weakest of the good outcomes and must not be read as more.** It was measured on one
machine, one OS and one set of library versions, on fixtures whose extracted text ran from 6 to 124
characters. Determinism across *versions* is the question an adoption path actually faces and is
**not** tested. This limitation was committed in writing before the run.

### The mechanism: an `ExtractionIdentity`, recorded at index time

The precedent is already in the tree and is deliberate. `EmbeddingProfile.dependencies`
(`recall/embeddings.py:416`) carries the inference library version as key material, and its
docstring says a `fastembed` upgrade costs a re embed on purpose, "because ONNX runtime changes are
free to move the last bits of a vector and a cache cannot tell". **The identical argument applies to
`pdfplumber` and to LibreOffice**, and extraction has no equivalent:
`STRUCTURED_DOCUMENT_VERSION` (`recall/extraction.py:143`) versions recall's own block shape and
says nothing about the libraries.

So record, per extracted source:

```
ExtractionIdentity(
    structured_document_version,   # already exists
    extractor,                     # "pdfplumber", "libreoffice", "stdlib-email", ...
    dependencies,                  # resolved versions, as EmbeddingProfile.dependencies does
    external_tool,                 # ("libreoffice", "26.2.5.2") or None
)
```

⚠️ **Implementation constraint, found the hard way:** `soffice --version` exits 0 and prints
**nothing** on Windows. A naive implementation records an empty version, compares equal across
upgrades, and defeats the identity it was added for. Read the binary's file version metadata or the
`version.ini` beside it, and **refuse to record an identity when the version cannot be determined**
rather than recording a blank.

### The verdict is four way, not a boolean

| Stored identity vs current | Re extraction | Verdict |
|---|---|---|
| same | matches | **verified** |
| same | differs | **defect**, and loud: at one identity, extraction is deterministic |
| different | matches | verified, plus evidence the version change is output neutral for this corpus |
| different | differs | ⚠️ **not attestable in this environment**. Refuse or re embed, and claim nothing |

That bottom row is what the chunker attestation does not need, and it is the whole difference.
**"Cannot tell" has to be a first class outcome**, because collapsing it into failure would report a
routine LibreOffice upgrade as corpus corruption, and collapsing it into success would be the
fail open this design exists to remove.

### Retroactively: not attestable, and that is the honest answer

Nothing today records an extraction identity, so **no existing non markdown corpus can be attested**,
however deterministic extraction turns out to be. This is the same shape as the `embedding_profile`
defect in F3: fixing a writer does not repair rows already written.

**Rule for adoption:** a non markdown source with no recorded `ExtractionIdentity` is **not
adoptable** and must be re embedded. Only markdown gets the complete answer of section 6c.

### Should extraction identity join `_index_fingerprint`?

**Yes, and the cost is real.** Section 6c documents what happens without an *effective* term:
the fingerprint carries a `chunker_version` string that no chunker change moves, so a corpus
silently accumulates chunks from several chunker eras. Leaving extraction out reproduces that defect
one level down, with a worse blast radius, since an extractor upgrade can change every PDF in the
corpus at once.

⚠️ **State the requirement as a behaviour, not as the presence of a field.** The chunker case is the
warning: a field named after the thing exists and is inert, so "there is a chunker term" was true and
useless at the same time. An `ExtractionIdentity` term earns its place only if changing an extractor
version changes the fingerprint, and that is what a test must pin.

The price is that a `pdfplumber` patch bump re extracts and re embeds every PDF. That is the same
price `EmbeddingProfile.dependencies` already charges for a `fastembed` bump, and it was accepted
there for the same reason. Naming it here so the trade is chosen rather than inherited.

### A defect found while measuring

`.msg` appears in the LibreOffice branch at `recall/extraction.py:184` but is **unreachable**:
`extract_document` matches `.msg` earlier at `:176`. A deployment without `python-oxmsg` therefore
gets an extraction error where the code appears to offer a fallback. Filed separately; it is not
part of this design.

## 7. What this design does not do

- It does not touch policies 1, 2 or 4, **subject to the `recall_index` correction in Q2**.
  `RECALL_ENV` survives for those and should be renamed per axis so it cannot be re overloaded.
- It does not lift the strict default. `provisional` is a *third* answer, not a relaxation, and
  decision 5 keeps a strict policy refusing it unless the operator opts in explicitly.
- It does not make a generated query set equivalent to a labelled one, and says so on the wire.
- It does not fix the 0.5 constant's remaining user, `RECALL_TRUST_MODE=development`, which stays as
  the explicit escape hatch for an uncalibratable corpus.
- It does not verify the CHUNKER. Decision 9 sizes the embedder check only; the re chunk
  counterpart is named in section 6b and not designed here.
