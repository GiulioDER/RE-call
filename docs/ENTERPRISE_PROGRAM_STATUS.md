# Enterprise retrieval program status

Rolling handoff between sessions of the enterprise retrieval program described in
[ENTERPRISE_RETRIEVAL.md](ENTERPRISE_RETRIEVAL.md). Newest session first. Each entry records what
landed, what was measured, and what is blocked, so the next session can start without
re-deriving state.

`docs/*.md` is deliberately outside `claim_gate.py`'s `GATED_DOCS`, so figures here carry no
evidence markers. If this file is ever promoted into the gate, every number below needs a marker
before that change can go green.

---

## 2026-08-05, embedding profile identity: one registry, a cache that cannot alias, the Qwen rejection published

### Session ledger

| # | Item | Outcome |
|---|---|---|
| 1 | Collapse the two profile vocabularies into one registry | done, `recall/embedding_registry.py`; both dict literals are gone |
| 2 | Asymmetric semantics, with the legacy `embed()`-only fallback preserved | done, the declared encoder mode is now the dispatch key |
| 3 | Cache keyed on the complete immutable identity | done, `EmbeddingProfile.fingerprint`; `cache_key` refuses a bare profile ID |
| 4 | Offline enforcement, proven rather than asserted | done, unit tests plus a real-loader run on VPS2 |
| 5 | Preserve and publish the Qwen rejection | done, registry record plus `ENTERPRISE_RETRIEVAL.md`, tied together by a test |
| 6 | Prove every new test can fail | done, 67 of 67; 66 by mutation here, 1 on VPS2 because it skips on Windows |

This is backlog session 8 (items 22, 23, 24), plus item 9 from session 4 and the
symlink-escape half of item 29, taken ahead of session 3.
The previous entry names session 3, the outbox drain, as what to start with. It is untouched and
still first in the backlog.

### Two other sessions landed work while this one ran

`origin/master` moved four times during this session: #192 and #191/#190 (dependabot), then
**#197** (OIDC wiring, 22 CCA audit fixes) and **#196** (erasure and control-plane fixes). #196 and
#197 together touched 26 files, four of which this session also touched: `.env.example`,
`docs/ENTERPRISE_RETRIEVAL.md`, `recall/generations.py` and `tests/test_generations.py`.

The branch was rebased onto `cd0cbe0`. Three of the four auto-merged; `tests/test_generations.py`
was an add/add conflict at the tail, where both sides appended tests. It was resolved by keeping
**both**, upstream's first, and the resolution script asserts every test function from each side is
present by name in the result and that no conflict marker survives, rather than trusting a reading
of the diff. Nine upstream tests and one of this session's are all present.

Everything below was re-verified on the rebased tree, not on the tree the work was written against.

### What landed

**One registry.** `recall/embedding_registry.py` owns `profile_id`, `model_name`, `dimension`,
`query_mode`, `passage_mode`, `context_mode`, `normalization`, `instruction_version`,
`chunker_version`, the artifact digest and the backend, for all six registered IDs. The two
literals it replaces were `recall_mcp/service.py:113-120` (profile to context version, six entries)
and `recall/context.py:37-41` (profile to context mode, three entries plus a silent default).

Two properties do the work, and both are tested:

* `context_version` is **derived** from `context_mode` rather than declared next to it, matching
  what `Indexer.__init__` already enforces. The two values cannot disagree because there is only
  one of them.
* A profile added to the registry needs no second edit. The test adds a seventh profile to the
  registry alone and asserts that both `context_policy_for_profile` and `make_embedder` already
  know it. That test is red against the old arrangement, which is what makes it a guard rather
  than a description.

`make_embedder` is now environment parsing only. `RegisteredProfile.build` is the single
construction site, and `RegisteredProfile.identity` the single constructor of a runtime
`EmbeddingProfile` for a registered ID.

**Asymmetric semantics.** `query_mode` and `passage_mode` are dispatch keys handed to the backend,
not documentation of one: `FastEmbedEmbedder` resolves `getattr(model, mode)` and refuses a mode
the backend does not have, rather than falling back to the symmetric encoder. Dimension discovery
goes through the passage encoder, and a probed width that contradicts the registry refuses
startup. `GenerationManager.build` was still indexing with `embedder.embed`; it now uses
`embed_passages`. The `Embedder` protocol is unchanged and an `embed()`-only embedder still works
through both helpers, which six tests pin.

**Cache identity.** `EmbeddingProfile.fingerprint()` is a SHA256 over the complete identity, and
`cache_key` now takes the profile rather than its ID. The signature deliberately refuses a bare
string: the old call passed the profile ID, so the unsafe call is the one that used to be correct,
and a `str | EmbeddingProfile` union would have let every existing caller keep the weaker key.

This answers backlog item 24 as a side effect. `normalization`, `instruction_version`,
`chunker_version` and `dependencies` were read by nothing; they are now key material, so a change
in any of them re-partitions the cache instead of silently serving vectors produced under the old
value.

⚠️ **The change invalidates every existing embedding cache entry.** Old entries were keyed by
profile ID and are simply not found. The cost is one re-embed of whatever was cached; nothing is
served under a key that no longer describes it.

**Offline enforcement.** Artifact verification moved ahead of the backend import, so a missing or
tampered tree fails the same way whether or not the optional extra is installed. Startup is proven
to complete with connections refused, and to refuse on a missing artifact, a checksum mismatch, a
malformed digest, an empty tree and a symlink escaping the artifact root.

**The Qwen rejection.** Registered, marked rejected, and carrying the measurement that decided it.
`RejectionRecord` refuses to be constructed without one, because a verdict with no number is an
opinion the next session re-litigates. A test asserts `ENTERPRISE_RETRIEVAL.md` carries every
number the registry carries, so neither copy can be deleted quietly.

### What was measured

**The real loader, offline, on VPS2.** Root SSH, not qwen-mcp: this program lives in
`/opt/recall-enterprise`, which is outside qwen-mcp's four file roots. The check ran against
fastembed 0.8.0 and the artifact provisioned at
`/opt/recall-enterprise/models/bge-fastembed-cache`, from `/var/tmp/recall-profile-check`
(scratch, left in place). `/opt/recall-enterprise` was not modified.

| Check | Result |
|---|---|
| `artifact_tree_sha256` of the provisioned tree | `9a443d711e06…c919c`, **equal to the digest recorded in `manifest.json` on 2026-08-03** |
| Socket block fires (positive control) | yes, on `create_connection`, `getaddrinfo` and `socket.connect` |
| `bge-small-symmetric-v1` loads with connections refused | yes, dim 384, both encoders return 384 |
| `bge-small-asymmetric-v1`, `bge-small-context-section-v1` | both load, dim 384 |
| Checksum mismatch | refused |
| Missing artifact | refused |
| Deployment identity fingerprint | `c992f9f018d68570acb82d27233599cc20a2d65ed485b0e2fcfb18201cceacc7` |

The digest agreement is worth naming: the manifest value was produced by a different tool months
earlier, so this is an independent confirmation of the tree-hash implementation, not a
self-comparison.

**Finding: `bge-small-asymmetric-v1` is asymmetric in name only.** Under fastembed 0.8.0,
`embed`, `query_embed` and `passage_embed` return **byte-identical** vectors for
`BAAI/bge-small-en-v1.5` (`embed == query == passage`, cosine 1.0). `TextEmbedding.query_embed`
delegates to the model implementation, and this one applies no query prefix.

Nothing in this session depends on that being false, and nothing here changes it. What it means:

* the three context profiles and `bge-small-asymmetric-v1` currently differ from
  `bge-small-symmetric-v1` in the passage TEXT they embed, not in the encoder they use;
* the dispatch still has to be correct, because `qwen3-embedding-0.6b-384-v1` does use a distinct
  instruction-prefixed query encoder, and because a future fastembed or model could add one
  without changing a profile ID;
* a claim that this deployment uses a distinct query encoder for BGE would be **false**. If a BGE
  query instruction is wanted, that is a new experiment and needs registering. It was not done
  here: new model behaviour is out of scope for this session.

**Finding: the socket block has to be a `socket.socket` subclass.** Replacing `socket.socket` with
a function looks stricter and is unusable: `ssl` builds `class SSLSocket(socket)` at import time,
so the swap turns any later `import ssl` into a `TypeError`. The first VPS2 run failed exactly
that way, because fastembed's import chain reaches `requests`. The local test now uses the same
instrument as the VPS2 check, so the unit test and the real run are measuring the same thing.

### Proving the tests can fail

67 tests across the six touched files plus one in `test_generations.py`, every one shown red.
Not a coverage number: a mutation harness applied one narrow change at a time, recorded which tests turned red,
and restored. It refuses to report anything if a mutation's search string is absent, so a silently
unapplied mutation cannot read as "the tests survived it".

| Route | Tests | How |
|---|---|---|
| Wide sweep, 57 mutations | 64 | one mutation per guard, across `embedding_registry`, `embeddings`, `cache`, `context`, `index`, `service`, `timing`, `generations`, `calibration`, `readiness`, and the published document |
| Two targeted mutations | 2 | see below |
| Run on VPS2 | 1 | `test_a_symlink_escaping_the_artifact_root_is_refused` **skips on Windows**, and a skipped test is never red. Run on Linux against the current module: the guard refuses, and with the guard removed it does not. A skip is not a pass, and this is the difference |

**The sweep found a test of mine that could not discriminate.** `make_embedder hardcodes the
artifact path variable` survived: the Qwen test set no digest, so `make_embedder` refused for the
missing digest rather than the missing path, and it passed identically whether or not the profile's
own variable was read. The test now supplies a real provisioned tree and a real digest, so the only
thing missing is `RECALL_QWEN_MODEL_PATH`, and the mutation kills it.

**Two survivors were bad mutations, not weak tests**, and both needed aiming at what the test
actually asserts. Relaxing `strict=True` on the FastEmbed side changed nothing the test reaches,
because that branch only runs when no digest is supplied, which never happens on the registry path;
all four strict resolves have to go before a missing artifact stops raising `FileNotFoundError`.
And dropping the relative path from the tree digest left the NUL separator behind, so an added file
still moved the hash by one byte; the separator has to go with it. Both kill their test once
aimed correctly.

⚠️ The harness restores with `Path.write_text`, which rewrites LF as CRLF on Windows, so it left
ten files dirty in `git status` with no content diff. Line endings were normalised afterwards and
`git status` verified back to exactly this session's files. A next session running it should
restore by BYTES.

### Gates run

| Gate | Result |
|---|---|
| `ruff check .` | clean |
| `mypy` | clean, 139 source files |
| `pytest -q` | 2263 passed, 35 skipped, 0 failed on the rebased tree (14 m 03 s, dedicated database) |

**The local suite needed a dedicated database, and then a FRESH one.** Two things bit, in order.

First, contamination: a `ConcurrentMigrator: another RE-call schema migrator is already running`
cascade across 74 tests, from another Python process on this machine holding the migration
advisory lock on the shared `localhost:5432` container. Nothing was killed, because a process that
might belong to a concurrent session is not mine to kill; the suite moved to a throwaway container
via `RECALL_TEST_DSN`.

Second, after the rebase, **2273 errors, all one cause**:
`MigrationChecksumMismatch: applied migration 0008_generation_foundation.sql checksum drift`. #196
edited an already-applied migration, and the throwaway database still had the previous version in
its ledger. That is the checksum guard working exactly as designed, not a defect in either change,
but it means **a test database provisioned before #196 must be recreated, not reused**. A next
session on this machine should expect both.

### Decisions a reader should be able to reverse

1. **A rejected profile warns, it does not refuse.** `make_embedder` logs a warning naming the
   verdict and the measurement, then builds it. Refusing outright (with an explicit override
   variable) would be the fail-closed choice and is one line; it was not taken because it changes
   operator-facing behaviour beyond what was asked. VPS2's active profile is
   `bge-small-symmetric-v1`, so nothing live is affected either way.
2. **Calibration is still keyed by profile ID alone.** `load_for` matches `Calibration.embedder`
   against the profile ID, so two different artifacts under one ID share a threshold. The
   fingerprint now exists and would be the stricter key. Not changed here: calibration artifacts
   on disk carry the ID, and calibration semantics were out of scope.
3. **The default `FastEmbedEmbedder()` still claims `bge-small-symmetric-v1` with a
   `legacy-unverified` digest.** That is a legacy identity sharing a registered ID. Left alone
   deliberately: every evaluation harness and published results table labels that embedder, and
   renaming it would move published labels and orphan existing calibration files. Readiness
   already fails it for the unpinned digest.
4. **`EmbedderIdentity` in `recall/lineage.py` overlaps with `EmbeddingProfile` and was not
   merged.** They serve different layers (manifest provenance against runtime encoder identity).
   Worth a deliberate decision at some point rather than drift.

### Standing blockers

| Blocker | Kind | Effect | Change |
|---|---|---|---|
| **No latency reference host.** VPS2 has 12 cores under a permanent load average near 8 from unrelated live production. | External dependency. Do not work around it. | Latency is **PENDING**; promotion blocked on latency grounds. Quality and safety gates still run. | unchanged. The VPS2 work this session was correctness only and cites no timing |
| **No production corpus.** | Open | Nothing may be claimed about enterprise-corpus behaviour. | unchanged |
| **No approved local generator confirmed.** | Open | The generator-neutral evidence path stays unexercised end to end. | unchanged |

### What the next session should start with

1. **Session 3 of the backlog, unchanged: give the migration outbox a drain.** `replay_pending`
   still has no producer and no operator command, and `cutover` still refuses forever after a
   crash between the event append and its completion. It remains the only item that can deadlock a
   production migration with no shipped workaround.
2. Decide item 1 above: should a rejected profile refuse to start?
3. If the asymmetric BGE profiles are meant to use a distinct query encoder, register that as an
   experiment. Today they do not, and the registry now says exactly which encoder each profile
   claims, so the gap is visible rather than assumed.

---

## 2026-08-05, gap matrix: 21 requirement areas audited against master

### Session ledger

| # | Item | Outcome |
|---|---|---|
| 1 | Audit 21 requirement areas against the merged program | done, every verdict carries a `file:line` or an explicit "no such symbol anywhere" |
| 2 | Confirm or refute eight inherited leads | done, 6 confirmed, 1 sharpened, 1 refuted |
| 3 | Decide the two-ledger question and record reasons | done, **keep separate**, with three follow-ups |
| 4 | Produce the ordered backlog mapped to sessions 3 to 11 | done, 37 items |
| 5 | Rebase onto #194, which landed the prior session's status file first | done, both entries kept |

Read-only against the code. Nothing was fixed. Deliverable:
[docs/superpowers/plans/2026-08-05-enterprise-gap-matrix.md](superpowers/plans/2026-08-05-enterprise-gap-matrix.md).

### A concurrent session wrote the entry below, and it landed first

The 2026-08-04 entry was **still being written while this audit ran**. It grew from 170 to 266 lines
mid-session, gaining an AUD-1 fix, a corrected "macro-averaged" label and a rebuilt discriminating
test. That session shipped it as #194, which merged before this one opened, so the 08-04 entry
reached `master` on its own and this section was rebased on top of it.

The add/add conflict that produced was resolved by keeping **both** entries, newest first, rather
than by taking a side. `master`'s header and its whole 08-04 entry are carried through **byte for
byte**, asserted rather than eyeballed: the resolution script compares the pre-entry and post-entry
line ranges against the merged blob and refuses to write if either differs.

The lesson for the next session that touches this file: it is a shared handoff, so two sessions will
race it again. Read `origin/master`'s copy immediately before editing, never a snapshot taken earlier
in the session, and append rather than rewrite.

### Corrections to the inherited snapshot

- **This file was untracked when this audit started.** The 2026-08-04 entry records creating it as
  done, but at that point it existed only as an untracked working-tree file on the local branch
  `bench/mtrag-symmetric-baseline`, in no ref. #194 has since committed it; see the note above.
- **`bench/mtrag-symmetric-baseline` was local and unpushed when this audit ran.** The MT-RAG runner
  commit existed on that branch only, with no remote and no PR. It has since shipped as #194 and is
  on `origin/master`; this session never touched it.
- **The audited tree is not PR 181.** Five PRs merged after it, four of which changed audited code:
  #182 calibration binding, #184 strict trust, #185 shared pool and tenant-scoped readiness, #189
  store latency instrumentation. Verdicts describe `origin/master` at `8147d96`, and the matrix says
  per row where the current state is mostly later work.
- **`recall/migrations/sql` holds 0001 to 0011, not 0001 to 0010.** `0011_calibration_binding.sql`
  arrived with #182.

### Verdict summary

| Verdict | Count | Areas |
|---|---|---|
| Implemented and tested | 1 | 17 |
| Implemented, insufficiently tested | 6 | 1, 3, 6, 7, 9, 16 |
| Partially implemented | 10 | 2, 4, 5, 10, 11, 12, 13, 14, 15, 20 (12, 13 and 14 are "implemented, zero tests") |
| Missing | 1 | 19 |
| Intentionally rejected / out of scope | 0 | none |

Areas 8 and 21 are counted above by their dominant state and neither fits one column cleanly: area 8
is implemented in code with its **required record** missing, and area 21 is partial with two live
contradictions. The matrix says so per row.

### The findings that change what a later session should do first

1. **`replay_pending` has no producer and no operator command.** `recall/control_plane.py:298` is its
   only reference in the whole repository, and `recall-enterprise` has no `replay` subcommand
   (`recall/enterprise_cli.py:22-41`). A crash between the event append and its completion
   (`recall/index.py:625` and `:631`) leaves a pending row, and `cutover` then refuses forever
   (`recall/control_plane.py:278-279`). The recovery mechanism exists, looks correct, and cannot be
   invoked. This is session 3.
2. **Erasure does not reach the outbox.** A pending event's payload holds full chunk text and vectors
   (`recall/index.py:614-623`) and is cleared only on completion (`recall/control_plane.py:261`).
   Nothing removes it on `recall_forget`. For a right-to-erasure path that is a policy question, not
   only a bug.
3. **The promotion gate has never been shown to pass.** `tests/test_promotion.py` is one test and it
   asserts `not promoted`. A gate only ever observed to fail is compatible with a gate that refuses
   everything.
4. **Three whole areas have zero tests**: route notification and polling (12), dual writes (13),
   atomic cross-generation forget (14). `ShadowIndexTarget`, `get_shadow`, `make_profile_embedder`,
   `delete_sources_across`, `invalidate_route`, `replay_pending`, `RetrievalOverloaded`, `stage_ms`,
   `check_enterprise_readiness` and `enterprise_cli` each have **no test reference anywhere** in
   `tests/`.
5. **The Qwen3 rejection record does not exist.** `git grep -in "qwen"` over tracked `*.md` returns
   nothing. PR 181 describes the profile as "promotion gated"; no gate and no record are in the
   repository.
6. **`check_enterprise_readiness`'s calibration branch cannot fire.** PR 181 passed
   `calibration=calibration`; commit `0341c15` (#182) removed the argument, and master calls the
   function with three (`recall_mcp/server.py:336-339`). The parameter defaults to `None`
   (`recall/readiness.py:160`), so every enterprise boot takes the `calibration is None` path at
   `:195-196`: a permanent degraded-readiness warning, and the identity-mismatch **failure** at
   `:198` is unreachable from the server. It reads as a calibration gate and is not one. Nothing
   caught it because the function has no test.

### Leads: what survived verification

| Lead | Verdict |
|---|---|
| Profile registry is a dict literal in `service.py::make_embedder`, partly duplicated in `context.py` | **confirmed**: `recall_mcp/service.py:113-120` and `recall/context.py:37-41`, two independent literals, already differing in extent |
| `cache_key` omits `context_version` and `artifact_digest` | **confirmed**: `recall/cache.py:26`; both are independently settable at `recall/embeddings.py:334-339` |
| `latency_budget_ms` is enforced nowhere; only the promotion gate reads it | **sharpened**: `git grep` returns three hits, all inside `recall/profiles.py`. The gate's budget is a separate caller-supplied field (`recall/promotion.py:33`). **Nothing** reads the profile's value |
| `recall/evidence.py` is complete but unexported and unreachable | **confirmed**: absent from `recall/__init__.py`; referenced only by `tests/test_evidence.py` and one prose line |
| `recall/promotion.py` has no producer outside its test | **confirmed**: `tests/test_promotion.py:8-13` is the only construction site |
| `FAST_PROFILE` and `QUALITY_PROFILE` share `candidate_k=20` | **refuted as a defect**: `docs/ENTERPRISE_RETRIEVAL.md:57-59` specifies "the same candidate pool" deliberately, so cost differences are attributable to the reranker alone |
| `readiness.py`, `validate_generation_parity`, `replay_pending`, `enterprise_cli.py` have no tests | **confirmed, one narrowed**: `tenant_readiness` / `process_readiness` (later session) *are* covered by `tests/test_tenant_readiness.py`; `check_enterprise_readiness` is not. The other three have no test reference at all |
| Two migration ledgers exist | **confirmed**: see the decision below |

### The ledger decision

**Keep `recall_schema_migrations` and `recall_schema_versions` separate. Do not merge.** Merging is a
one-way door guarded by committed checksums (`recall/schema.py:53-54`); the two ledgers have
different scoping (per target table with a `__global__` bucket, versus database-global) and different
lifecycles (the control plane must exist before any generation does); and the enterprise deployment
is opt-in, so merging would impose control-plane tables on every deployment.

The split must be made deliberate rather than accidental, which needs three things it does not have:
an advisory lock on `ControlPlane.apply_migrations` matching `recall/schema.py:24`; readiness that
verifies **both** ledgers rather than only the one `check_schema` covers; and a paragraph in
`docs/MIGRATIONS.md`, which currently never mentions the second ledger. Full reasoning in the matrix.

### Documentation contradictions found on master

Both are shipped, and each makes an operator following one document wrong according to the other.

- `docs/ENTERPRISE_RETRIEVAL.md:13` sets `RECALL_DSN` to the **migration** role, and
  `recall/enterprise_cli.py:14` reads it. `docs/MIGRATIONS.md:12` calls `RECALL_DSN` the deprecated
  fallback for the **serving** DSN, with `RECALL_MIGRATION_DSN` as the schema owner.
- `README.md:628-632` says strict calibration enforcement "has not landed yet"; #184 landed it and
  `README.md:205` already marks it ✅.

### Standing blockers

| Blocker | Kind | Effect | Change |
|---|---|---|---|
| **No latency reference host.** VPS2 has 12 cores under a permanent load average near 8 from unrelated live production. | External dependency. Do not work around it. | Latency is **PENDING**; promotion blocked on latency grounds. Quality and safety gates still run. | unchanged |
| **No production corpus.** Everything measured is the public MT-RAG release. | Open | Nothing may be claimed about enterprise-corpus behaviour. | unchanged |
| **No approved local generator confirmed.** | Open | The generator-neutral evidence path stays unexercised end to end. | **now blocks backlog item 11** (session 5): the evidence boundary can be made reachable and tested against a fake generator, but the real path stays unexercised until this resolves |

### What the next session should start with

Session 3 of the backlog: give the migration outbox a drain. Add `recall-enterprise replay`, test
`replay_pending` against the actual crash shape, test the `cutover` refusal branch, and decide what
erases a pending event's payload when a tenant invokes erasure. Everything else in the program can
wait behind a deadlock that has no shipped workaround.

Two items carried forward from 2026-08-04 and still open: the MT-RAG baseline has no
`results/ARTIFACTS.md` row (deliberately), and `bench/mtrag-symmetric-baseline` is still local and
unpushed.

---

## 2026-08-04 — MT-RAG symmetric baseline salvaged, validated, archived, runner committed

### Session ledger

| # | Item | Outcome |
|---|---|---|
| 1 | Salvage the untracked runner and the finished run off `/var/tmp` | done |
| 2 | Validate the frozen run against its preregistration | done, 8 checks, all PASS |
| 3 | Archive with a SHA256 manifest and a provenance note | done, 29 files |
| 4 | Commit the runner under version control | done, this branch |
| 5 | Create this status file | done |
| 6 | Fix AUD-1 (the "macro-averaged" label) | done, label corrected and pinned by a discriminating test |

No new benchmark run was performed. Implementation work was out of scope for the salvage itself;
the AUD-1 fix (item 6) was authorised separately after the audit surfaced it.

### Corrections to the inherited snapshot

Two facts in the session brief did not survive verification, and are corrected here so the next
session does not inherit them again.

- **`docs/ENTERPRISE_PROGRAM_STATUS.md` did not exist.** It was described as the authoritative
  prior-session handoff. It was absent from `origin/master` and from every ref. This file is its
  first version, not an update.
- **`summary.json` reports gap warnings on two arms, not one.** The brief named the single warning
  on `recall_default_last`. `sparse_last` reports 507 of 507. See the explanation below; it is
  expected behaviour, but a reader told to expect one warning would read the arm as broken.

Also worth carrying forward: the local checkout was on `codex/enterprise-retrieval-program` at
`1aa93ec` while `origin/master` had advanced to `8147d96`. This branch was cut fresh from
`origin/master`.

### Salvage record

The runner existed only as untracked files inside a `/var/tmp` checkout, and the results existed
only in `/var/tmp`. Both are now in two durable places.

| What | From | To |
|---|---|---|
| `benchmarks/mtrag/{README.md,__init__.py,run.py}`, `tests/test_mtrag_benchmark.py` | `/var/tmp/re_call_mtrag_20260803/RE-call` (rev `3d3c905`, untracked) | this repository, and `…/runner/` in the archive |
| `results/official_run_1/` (6 predictions, 6 metrics, manifest, summary) | `/var/tmp/re_call_mtrag_20260803/results/` | `/var/lib/recall-benchmarks/2026-08-04-mtrag-symmetric-baseline/results/` |
| `benchmark.log`, 5 index logs, `provision.log`, both shell drivers, `preregistered_manifest.json` | `/var/tmp/re_call_mtrag_20260803/` | same archive directory |

Archive root: **`/var/lib/recall-benchmarks/2026-08-04-mtrag-symmetric-baseline/`** on VPS2,
29 files, 61 MB, `MANIFEST.sha256` covering every file, `NOTE.md` carrying the provenance.
`sha256sum -c MANIFEST.sha256` passes. Every archived artifact was compared byte for byte against
its `/var/tmp` source at archive time. **The source directory was not deleted.**

VPS2 access for this work used root SSH, not qwen-mcp: qwen-mcp's file roots are
`/opt/sentiment_agent`, `/var/lib/qwen_agent`, `/var/log/qwen_agent` and `/etc/systemd/system`,
and this program lives outside all four.

### Validation verdicts

Every check ran against the archived copy, with the expected values taken from the preregistration
and the MT-RAG release rather than from the predictions themselves.

| # | Check | Verdict | Evidence |
|---|---|---|---|
| 1 | Archive is byte-identical to the `/var/tmp` source | **PASS** | 14 files compared by SHA256, zero mismatches |
| 2 | Release inputs match the manifest | **PASS** | `reference.jsonl` hashes to the manifest's `input_sha256.tasks` |
| 3 | Each of the six prediction files holds exactly the 507 frozen task IDs, once each | **PASS** | per file: 507 rows, 507 unique, 0 duplicates, 0 missing, 0 extra, against the ID set derived from `reference.jsonl` |
| 4 | Prediction SHA256 for the two known arms | **PASS** | `recall_default_last` = `d0f4ce2d…51676`, `recall_default_recent3` = `12cc5e3f…3fef1`, both as expected |
| 5 | Scored metrics use exactly the 332 judged tasks | **PASS** | all six arms: `overall.count` 332, domains clapnq 83 / cloud 86 / fiqa 58 / govt 105, and `per_query` keys set-equal to the qrel query IDs |
| 6 | The six arms match the preregistration exactly | **PASS** | names in order, and each `metrics.arm` equal field for field to the matching `frozen_arms` entry. The preregistration was not edited or relabelled |
| 7 | DB chunk counts per domain | **PASS** | clapnq 183 408, cloud 72 442, fiqa 61 022, govt 49 607, total 366 479 |
| 8 | Revisions and adapter identity | **PASS** | RE-call `3d3c905…`, MT-RAG `cc5b1d4…`, adapter SHA256 `a675d900…e2347` equal to the salvaged `run.py` |

The four previously unrecorded prediction hashes, now facts:

| arm | SHA256 |
|---|---|
| `recall_rerank_last` | `cfd5f9a48d59b36511f4d803770f3da1e47239c85c0f0c1984510b185f5bea2c` |
| `recall_rerank_recent3` | `6defd4f299de6d0880a1d3dd889f72f34f4794240749e2b77bc0321db5ad273d` |
| `dense_last` | `d7e0b61dbc78987231ff0740115115db4d783a3b56c8fd75dc3c7005c24430a1` |
| `sparse_last` | `3d1395294d5b676d9fb8748e9f512f0f7cd59a0adb8fc5d0deb3e078a3523333` |

#### Two findings worth carrying forward

**The DB count check first returned zero, and that was the tenant isolation model working.** A
plain `SELECT count(*)` on `recall_mtrag_bge_v1_<domain>` as the runtime role returns 0 rows, not
the counts above. The tables carry `FORCE ROW LEVEL SECURITY` with a `tenant_isolation` policy on
`current_setting('recall.tenant_id', true)`; with the GUC unset, the policy correctly matches
nothing. The true counts were confirmed three independent ways: `pg_stat_user_tables.n_live_tup`,
a count with `recall.tenant_id` set, and a count as the table owner. All rows are tenant
`default`. A future session that reads zero here should set the GUC before concluding the index is
empty.

**`gap_warning` on `sparse_last` fires on every query by construction.** `gap_warning` is computed
over the dense candidate pool (`recall/guards.py`, floor 0.50 cosine), and an empty candidate set
counts as a gap, fail-closed. `sparse_last` sets `use_dense=False`, so the pool is always empty and
the flag always fires. It means "no dense evidence was gathered", not "the corpus lacks an answer".
This is documented in `HybridRetriever`'s docstring. The single warning on `recall_default_last`
is the real kind: task `34d3cde930baaf8a80a37bede060c827<::>2` (govt, judged), query "Who was the
last Kolb to live in the house?", where every dense candidate scored below the floor. The other
four arms report zero.

### What was measured

Full results are in the archive; the headline is reproduced here for handoff only, and the
artifact is authoritative.

**These are POOLED means over all 332 judged queries, not macro averages.** Each domain counts in
proportion to its judged-query count. The distinction is worth 1.5 % to 6.6 % and the two
definitions do not agree on one arm ordering; see AUD-1 below, now fixed in the README and pinned
by a test. The `elapsed` column is per-arm total including prediction writing and scoring, not
retrieval time alone (finding AUD-4, open).

| arm | nDCG@5 (pooled) | Recall@5 (pooled) | nDCG@10 (pooled) | elapsed (s) |
|---|---|---|---|---|
| `recall_default_last` (primary) | 0.3701 | 0.4081 | 0.4048 | 416.0 |
| `recall_default_recent3` | 0.3205 | 0.3604 | 0.3667 | 587.3 |
| `recall_rerank_last` | 0.4227 | 0.4555 | 0.4661 | 18 468.1 |
| `recall_rerank_recent3` | 0.3173 | 0.3671 | 0.3676 | 20 480.8 |
| `dense_last` (ablation) | 0.3304 | 0.3556 | 0.3677 | 120.7 |
| `sparse_last` (ablation) | 0.2542 | 0.2936 | 0.2905 | 418.3 |

Run window 2026-08-04T08:21:20Z to 2026-08-04T19:36:21Z, elapsed 11 h 15 m. The two reranked arms
consumed 96 % of it.

**The p50 and p95 latency figures in the artifacts are diagnostic only.** They were measured on a
12-core host under unrelated live production at a load average of roughly 8. No promotion decision
on latency grounds may cite them.

### Runner now under version control

`benchmarks/mtrag/` and `tests/test_mtrag_benchmark.py` are committed on this branch. They existed
in no ref of the repository before it.

Two changes were made to the salvaged `run.py`. First, it failed `mypy` with two `var-annotated`
errors and `disallow_untyped_defs` applies to `benchmarks/`, so two local variable annotations were
added. Second, `score_predictions` gained a docstring as part of the AUD-1 fix below. **The
committed `run.py` is therefore not byte-identical to the adapter that produced the run**, so its
SHA256 no longer matches `adapter_sha256`. The byte-exact adapter is preserved at
`…/runner/run.py` in the archive, and that is the copy the manifest's `adapter_sha256` refers to.

Behaviour-neutrality was proven, not assumed. The annotation-only change was verified by compiling
both files and comparing every code object on opcodes, operands, names and argument layout;
fingerprints were identical. Adding the docstring then broke that equivalence in
`score_predictions` (a docstring lands in `co_consts`), so bytecode identity was **replaced by a
stronger check rather than quietly dropped**: both adapters were run over the real frozen
predictions for all six arms, and every figure was required to equal the archived metrics file.

> All six arms reproduce the archived scores exactly, to full float precision, under both the
> archived and the committed adapter (`committed == archived == frozen` for every arm). Example:
> `recall_default_last` overall nDCG@5 is `0.3700930547143303` from both.

Both proofs carry a negative control (mutate `0.0` to `1.0` in `ndcg_at` for the bytecode check;
reverse one task's context order for the reproduction check), each confirmed to report a
difference, so a green result is detection rather than a check that cannot fail.

Gates run on the change: `ruff check .` clean, `mypy` clean across 137 source files,
`pytest tests/test_mtrag_benchmark.py` 3 passed, and the claim-gate suite
(`test_published_numbers_have_artifacts`, `test_results_artifact_model_stack`,
`test_results_artifact_provenance`, `test_findings_crossrefs`) 235 passed with
`benchmarks/claim_gate.py` exiting 0. No `results/ARTIFACTS.md` row was added and no number was
published into a gated document.

### Audit of the committed runner (CCA, DEEP tier, no-fix)

The commit hook required a bug review, so the runner went through the tiered CCA pipeline at DEEP
(forced by the numeric-path flag: this code computes the published metrics). Eight auditors, 38 raw
findings, deduplicated and verified. **Nothing was fixed:** implementation work was out of scope for
this session, and changing the adapter would diverge it further from the frozen run. Everything
below is reported for a decision.

**Deterministic coverage for this run was NONE.** `cca_checks` is not installed in this repo or its
venv, so no static backend was available. Where a verdict below says "measured" or "read", that is a
command I ran; where it says "reasoned", it is LLM adjudication only.

| ID | Finding | Verdict | Basis |
|---|---|---|---|
| **AUD-1** | `benchmarks/mtrag/README.md` said the adapter reports "macro-averaged nDCG and Recall". `score_predictions` pools all 332 judged queries into one list and divides by 332, weighting each domain by its judged-query count (clapnq 83, cloud 86, fiqa 58, govt 105). This repo already defines macro as the *unweighted* mean of per-corpus values (`recall/promotion.py:63-71`). | **CONFIRMED, P1 → FIXED** | measured |
| **AUD-2** | `preregistered_manifest.json` records `embedder_model` but no reranker model or revision, although two of the six arms rerank. The archived reranker identity had to be recovered by reading `recall/rerank.py`, not from the artifact. | **CONFIRMED, P2** | read |
| **AUD-3** | `index_domain`'s completion guard (`final_count != seen`) compares row *counts*, never row *identity*. Re-running against a changed corpus under the same `--table-prefix` yields `final_count == seen` and the guard passes on a silently mixed index. | **CONFIRMED, P2** | reasoned |
| **AUD-4** | Per-arm `elapsed_s` is evaluated after prediction writing and after `score_predictions` (which re-parses all four qrels files), so it is not retrieval wall time. The `latency_ms` p50/p95 are unaffected: those come from the in-loop list. | **CONFIRMED, P2** | read |
| **AUD-5** | Neither `FastEmbedEmbedder` nor `CrossEncoderReranker` is constructed with the offline-enforcement arguments the library exposes, so the harness can fetch model weights over the network at runtime. Runtime model downloads are on this program's standing out-of-scope list. | **CONFIRMED, P2** | reasoned |
| **AUD-6** | `run.py` has no module docstring, and `argparse.ArgumentParser(description=__doc__)` therefore renders `--help` with an empty description. | **CONFIRMED, P3** | read |
| **AUD-7** | `p50`/`p95` use `ordered[int(p * (n-1))]`, which floors. At n=507 the reported p95 is the 481st of 507 values, the 94.87th percentile, and the bias is always toward the fast end. | **CONFIRMED, P3** | measured |
| **AUD-8** | `run_arm` constructs its reranker, four `PgVectorStore`s and the task list *before* its `try`, and `PgVectorStore.__init__` opens a connection eagerly (`recall/store.py:606`) with no `__del__`. A failure while opening store 3 leaks stores 1 and 2. Bounded: the exception aborts the run and the process exit closes the sockets, so this cannot accumulate. Both auditors rated it P1; downgraded on that mitigation. | **CONFIRMED, P3** | read |

Also confirmed and minor: the `DATABASE_URL` dotenv fallback is the only non-`RECALL_`-prefixed
config key in the codebase; the overall aggregation lacks the empty-list guard its per-domain twin
has; the duplicate-`_id` failure reports itself as a "partial/mixed index".

Dropped as false positives, with evidence: the "no TypedDict" and "god file" structural findings
(style preferences on a deliberately frozen benchmark adapter, and `ruff` plus `mypy` are green),
and the zip-bomb finding (the only path to a hostile archive is the operator pointing
`--mtrag-root` at their own tampered copy).

**AUD-1 in detail, because it is the one that touches a published label.** Recomputing every arm
both ways from the archived per-domain rows:

| arm | nDCG@5 pooled | nDCG@5 macro | delta |
|---|---|---|---|
| `recall_default_last` | 0.3701 | 0.3602 | −0.0099 (−2.67 %) |
| `recall_default_recent3` | 0.3205 | 0.3073 | −0.0132 (−4.11 %) |
| `recall_rerank_last` | 0.4227 | 0.4133 | −0.0094 (−2.21 %) |
| `recall_rerank_recent3` | 0.3173 | 0.3067 | −0.0106 (−3.35 %) |
| `dense_last` | 0.3304 | 0.3253 | −0.0051 (−1.53 %) |
| `sparse_last` | 0.2542 | 0.2374 | −0.0168 (−6.61 %) |

The pooled figure is higher than the macro figure for **every** arm and **every** metric, by 1.5 %
to 6.6 %. Arm ordering is preserved under nDCG@5 and nDCG@10, but **not under Recall@5**, where
`dense_last` and `recall_default_recent3` swap. So this is not purely cosmetic: on one of the three
reported metrics the two definitions disagree about which arm is better. The headline conclusion
(`recall_rerank_last` first, `recall_default_last` second) survives both definitions.

This finding did **not** go to the adversarial skeptic panel. Its verdict rests on a definition read
out of `recall/promotion.py` and on a measurement executed over the artifacts, so it is
artifact-backed, and the pipeline forbids routing an artifact-backed verdict to three LLM skeptics
to re-litigate.

**AUD-1 is fixed.** The label was corrected rather than the metric. Changing the aggregation would
have invalidated the archived run's comparability without a re-run, for a figure that is already
derivable: `domains[*]` is in every metrics file, so anyone wanting the macro average can compute
it from what is archived. Three things changed:

1. `benchmarks/mtrag/README.md` now states that `overall` is a pooled mean over judged queries,
   says explicitly that it is not a macro average in `recall/promotion.py`'s sense, and records
   the size of the gap.
2. `score_predictions` gained a docstring stating the averaging semantics at the place the
   arithmetic happens, so the next reader does not have to re-derive it from the loop.
3. `test_score_predictions_is_macro_average` was **renamed and its fixture rebuilt**. This was the
   load-bearing part. The old fixture used one judged query in each of two domains, where pooled
   and macro both give 0.5, so a test whose name asserted "macro" passed identically under either
   definition. It could not have failed. The replacement,
   `test_score_predictions_overall_is_pooled_not_domain_macro`, uses three judged queries in
   clapnq against one in cloud: pooled gives 0.75, domain macro gives 0.5, and the test asserts
   both the pooled value and that it differs from the macro of the domain rows.

The new test was proven able to fail: `score_predictions` was mutated on an isolated copy to
compute the domain macro, and the test's assertions were replayed against it. Green on the real
implementation, red on the mutant. The proof script refuses to report a result at all if the
mutation fails to apply, so a silently unapplied mutation cannot read as success.

### Standing blockers

| Blocker | Kind | Effect |
|---|---|---|
| **No latency reference host.** VPS2 has 12 cores under a permanent load average near 8 from unrelated live production. It cannot serve as the 16-vCPU idle reference environment. | External dependency. Do not work around it. | Latency is **PENDING**. Promotion is blocked on latency grounds. Quality and safety gates still run. |
| **No production corpus.** Everything measured so far is the public MT-RAG release. | Open | Nothing may be claimed about enterprise-corpus behaviour. |
| **No approved local generator confirmed.** The evidence boundary requires a local generator, and none has been confirmed approved. | Open | The generator-neutral evidence path stays unexercised end to end. |

### What the next session should start with

1. Decide whether the MT-RAG baseline earns a `results/ARTIFACTS.md` row. It was deliberately not
   added here. Adding one means satisfying `claim_gate.py` and the two artifact tests in the same
   commit, and a new gated section costs a marker per number.
2. Resolve the latency reference host as an external dependency, or record explicitly that
   promotion stays blocked. It is not solvable on VPS2.
3. Confirm whether an approved local generator exists. Until it does, the evidence boundary cannot
   be exercised end to end.
