# Changelog

This file keeps the release surface short. The full historical changelog lives at
[docs/archive/CHANGELOG_FULL.md](docs/archive/CHANGELOG_FULL.md).

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versioning is pre-1.0
`0.MINOR.PATCH`, so a minor bump may still break schema or API.

## [Unreleased]

## [0.9.8] (2026-08-22)

### Added

* **`recall calibration drift` says whether the corpus under a live calibration has moved far
  enough to need refitting.** Until now the drift question could only be asked *after* a rebuild:
  `resolve` compares fingerprints and answers `STALE` on any mismatch, which is a yes/no about
  identity rather than a statement about magnitude, and `carry-forward` needs a new generation to
  already exist. Nothing could be asked the question an operator actually has between rebuilds.

  Two tiers, and which one produced the verdict is always in the output. The **screen** is a
  manifest comparison over `(uri, sha256)`: no embedding, no retrieval, not even a model load. The
  **probe** replays the calibration's own stored labelled query set and measures what the frozen
  threshold now costs, per class, by the same two conditions `carry-forward` enforces. ⛔ **The
  screen firing is never reported as a verdict**: where the probe cannot run the strongest verdict
  is `recalibrate_recommended`, and the report names the check that was not made, so a directory can
  never reach `recalibrate_required` however total its delta.

* **`RECALL_AUTO_CALIBRATE` and `recall calibration auto` re-establish a calibration without asking
  for labels.** After a generation build: `off` does nothing, `warn` (the default) reports, `auto`
  additionally carries the threshold forward, or refits it on the same stored labelled evidence
  when it has to move. **Neither path loosens certification**, so the automation is in what gets
  run and never in what gets accepted, and neither invents questions: a tenant with no published
  calibration reports `skipped`, because deciding what the labelled questions should be is not a
  decision to make unattended.

* **`recall calibration carry-forward`** and the `corpus_delta` / `threshold_error_rates`
  primitives, rescued from an unmerged branch and rebased onto master.

### Changed

* ⚠️ **There is deliberately no corpus delta at which recalibration is demanded outright, and this
  reverses the design this feature started with.** Measured over 57 snapshots of three real corpus
  histories (`docs/preregistrations/2026-08-21-calibration-drift-trigger.md`): the frozen threshold
  first crossed the 0.10 error bound at a delta of **0.945** and never below it, so a delta-only
  rule at 0.25 fires on **56 of 57** snapshots and is right about **5**, a precision of **0.09**.
  The labels also proved far more durable than that rule assumed: at delta 0.981 only **27.5%** of
  the answerable queries' original evidence still existed and the false-abstain rate was **0.025**.
  What moved was the false-*confirm* rate, tracking corpus **growth** rather than change as such.

  This does **not** license raising `--max-corpus-delta` on carry-forward, and it was left at 0.25.
  That bound governs whether a threshold may be *inherited*, and what the numbers establish is that
  a delta is a poor alarm, not that a large delta is safe.


- `recall uninstall` also removes the desktop app's handoff file from the user config directory
  (`%APPDATA%/RE-call/runtime.json`), which the previous release looked for in the wrong place.
- `--selftest` resolves the embedder only when a model cache already exists, and says which branch
  it took. It was downloading model weights on a cold cache.

### Fixed

- **A desktop upload no longer silently shrinks the corpus.** Carried-forward files that live
  outside the upload staging directory (which is every file a wizard install indexed) are kept and
  the build reader is widened to reach them. Only a file whose bytes are genuinely gone is dropped,
  and the count is now named in the upload's own message rather than passing unmentioned.
- **A failed `docker compose down` is reported.** With the docker daemon unreachable, `recall
  uninstall` recorded no failure, printed `Removed N item(s).`, and deleted the stack file naming
  the containers that were still running — which then made them unnameable by the tool that left
  them. The stack file and `wizard.json` are now kept whenever the teardown fails, so the uninstall
  can be retried.
- **An uninstall no longer overwrites the install-time backup of the MCP client config.** It writes
  its own under a name that is never reused, and copies the source file's mode rather than creating
  it at the umask default, since that file carries bearer tokens.
- **A volume the stack declared `external` is never removed.** A fallback that derived the
  historical volume name could reinstate exactly the volume that had just been excluded as not
  ours. The fallback now applies only to a legacy stack that declares no volumes at all.
- **A failure inside `build()` no longer strands a generation.** The desktop upload's cleanup path
  called only `abandon`, which refuses any state but `ready`, so failures before `validate()` left
  a full copy of the corpus that `gc` could not collect.
- **A blank or relative data folder is refused** rather than silently becoming the process's
  working directory, and the terminal interview reports it as a refusal instead of a traceback.
- **`corpus_version` may no longer begin with `desktop-`.** That prefix decides which generations
  the desktop upload path abandons; a wizard corpus carrying it would have been reclaimed.



* ⛔ **The graphical installer could not open its own window, and 0.9.7 shipped that way.**
  `install_main` called `run_window(InstallerWindow(...))`. Python evaluates an argument before the
  call, so the window was constructed before `run_window` had created the `QApplication` it needs.
  Qt answers a widget built with no application by printing `QWidget: Must construct a QApplication
  before a QWidget` and aborting the process on its fatal handler: one line of stderr, no window,
  no dialog, exit `0xC0000409`. Every copy of the 0.9.7 installer did this, on every machine.
  `run_app` had the identical call shape, so the main desktop window carried the same defect.

  `run_window` now takes a **factory** rather than a window, so the application is created first and
  the ordering cannot be decided at a call site. Passing an already-built widget raises a `TypeError`
  naming the mistake instead of killing the process.

* ⛔ **The self-test that exists to prove the bundle runs was green for that build, and now cannot
  be.** `--selftest` constructed its own `QApplication` and then its own window, in the correct
  order, which is not the order the entry point used. It rehearsed a launch sequence nothing ships.
  Every desktop test in the suite had the same shape, and not by carelessness: once any test in a
  pytest session creates a `QApplication` it cannot be unmade, so no in-process test can observe a
  process that has none. The self-test now builds its window through `application_and_window`, the
  same code the entry point orders its launch with, and `tests/test_desktop_launch_order.py` adds a
  check that starts the real entry point in a **fresh interpreter** and asserts on the exit code,
  which is the only thing here that reproduces a double-click.


## [0.9.7]

### Added

* **Production promotion works, gated on certification.** `generation promote` under
  `RECALL_ENV=production` used to refuse outright with "unavailable in production until
  certification gates land". Those gates have landed: promotion now succeeds for a generation whose
  published calibration certified and is still bound to this pipeline and corpus, and refuses every
  other status by name (MISSING, DRAFT, UNCERTIFIED, STALE each need a different action from
  whoever hit it). `--unsafe-development-promotion` is refused there rather than ignored, so the
  development escape hatch cannot be carried into production by habit.

* **A desktop upload into a production tenant can reach CERTIFIED.** The gate above had no
  reachable door on that path: nothing in `generation_ingest` produced a calibration, so every
  upload ended built-and-validated but never live, with the CLI as the only route to a live corpus.
  It now calibrates and publishes before promoting, using the same query-set generator the
  installer uses. Measured end to end against a production tenant: a ten-file corpus certifies and
  goes live; a one-file corpus reports "no certifiable query set could be generated from 1
  chunk(s)" and stays ready — the corpus's own reason, not the gate's generic one.

### Changed

* ⛔ **`generation rollback` never refuses on certification grounds, and this reverses a gate that
  shipped for one release.** Rollback is the incident path, and a gate that blocks recovery
  precisely when recovery is needed trades a visible degradation for an invisible workaround. Two
  ways the refusal bit, both certain rather than hypothetical: `forget()` rewrites the corpus
  fingerprint of every generation of a tenant, so one erasure request left no rollback target ever
  again; and every generation an existing install is serving was promoted under `development` and
  has no published calibration, so upgrading would have removed rollback from all of them.

  The invariant is kept by REPORTING instead of preventing. `generation_rolled_back` now records
  the target's resolved calibration status and an optional `provisional_reason`, so a recovery that
  downgrades a tenant from certified to provisional is visible. Reasoning:
  `docs/UNCALIBRATED_FIRST_RUN_DESIGN.md`.

* **The certification gate follows the SERVING environment, not the build one.** `GenerationManager`
  takes `serving_environment`, defaulting to `environment`. The wizard is why: it builds every
  corpus under `development`, because a production build demands a verifiable embedder identity a
  bundled model does not have, then serves those tenants with `RECALL_ENV=production`. Keyed on the
  build environment, the gate ran on no tenant an install creates.

* **One definition of a pipeline identity.** `generation_ingest` assembled its own, hardcoding
  `provider="fastembed"` for every embedder and spelling out a chunker identity with an empty
  configuration. A generation built by the wizard and one built by a desktop upload therefore
  carried different pipeline fingerprints for the same pipeline, which is what makes a published
  calibration resolve STALE. Chunk boundaries are unchanged; only the recorded identity moves.

* **The `fastembed` extra is bounded to `<1`.** `embedder_artifact_path` walks a private attribute
  chain to find a model's own snapshot directory, and does so defensively — a release that renames
  it returns `None` rather than raising, which makes the pipeline identity unverified, which makes
  production builds refuse. An install that silently stops accepting uploads, with nothing pointing
  at the dependency that moved. Verified reachable on fastembed 0.8.0.

### Fixed

* **The certification gate decided from a different transaction than the one it authorised.** It
  opened its own connection while `promote` held the tenant and generation rows `FOR UPDATE`, so a
  concurrent `forget()` could invalidate the calibration between the verdict and the commit, and the
  connection was acquired while holding a lock. Measured: it is **not** a deadlock, which is what it
  looks like — a plain `SELECT` does not wait on `FOR UPDATE` under MVCC. Now resolved on the
  caller's transaction; a competing fingerprint rewrite waits for the promotion rather than racing
  it.

* **The artifact digest cache vouched for bytes that had changed.** Keyed by path with no
  invalidation, the first answer stood for the life of the process, so a re-download or a swapped
  model file kept a provenance claim no bytes on disk supported. Now keyed by file count, total size
  and newest mtime alongside the digest. This detects staleness, not tampering: someone able to
  write into the model directory can also set mtimes.

* **A generated stack could serve one version while every file on disk claimed another.** Adding a
  project inherits the existing stack's image tag so the new corpus runs the same recall as its
  siblings, but the Dockerfile was regenerated at the running version. Measured on a 0.9.1 stack
  under a 0.9.6 wizard: tag `recall-wizard:0.9.1`, Dockerfile `recall-rag==0.9.6`. Compose reuses a
  tag rather than building it, so the container would have started 0.9.1 in silence. The Dockerfile
  now follows the tag.

* **A refused upload no longer loses the previous upload's files or leaks its corpus.** A new build
  seeded its manifest from the ACTIVE generation, and a refused promotion never advances that
  pointer, so each upload dropped every earlier un-promoted upload's files. Measured against a real
  database: upload #2 reported 1 file where it should have reported 3. Builds it supersedes are now
  abandoned so `gc` can reclaim them.

* **Six operator documents claimed production promotion was blocked.** Each was true when written
  and falsified by the gates landing, and each was the document somebody reads before deciding what
  a command will do: `PRODUCTION.md`, `MIGRATIONS.md`, `CALIBRATION.md`, `GENERATIONS.md`,
  `FIRST_CALIBRATION.md`, `ENVIRONMENT.md`.

## [0.9.6]

### Added

* **The `documents` extra is now published.** It has existed in `pyproject.toml` for some time but
  no release ever carried it, so `pip install "recall-rag[documents]"` installed nothing and said
  so only in a warning:

      WARNING: recall-rag 0.9.5 does not provide the extra 'documents'
      --- pip exit: 0 ---

  pip exits **zero** for an extra a release does not provide, so every install of it succeeded and
  silently omitted `pypdf`, `pdfplumber`, `python-docx`, `openpyxl`, `python-pptx`, `xlrd`,
  `beautifulsoup4` and `python-oxmsg`. Anyone who installed it got a recall that accepted `.pdf`,
  `.docx`, `.xlsx` and `.pptx` files and extracted nothing from them, which reads as recall being
  bad at documents rather than as a missing dependency.

  Found while giving the Windows wizard's generated Docker stack an image it could build: the
  Dockerfile pins the running version, and a post-install import check turned the silent omission
  into a build failure.

### Changed

* ⚠️ **BREAKING, and it costs one full re-index: the incremental skip guard now keys on the whole
  embedding identity rather than the profile id.** `index._index_fingerprint` hashed
  `embedding_profile_id(embedder)`, which is ONE field of an `EmbeddingProfile`, so any two
  embedders sharing an id were the same embedder as far as the skip guard was concerned. It now
  hashes `EmbeddingProfile.fingerprint()`, bringing it into line with `recall/cache.py`, which has
  always keyed on the whole profile and whose docstring gives the reason: "The ID alone is not an
  identity".

  The hole was reachable. A 384-dimension corpus and a 1024-dimension corpus produced **equal**
  index fingerprints for the same file, so swapping the model left every file skipped, every vector
  stale, and the run reporting success with a skipped count. Ten identity fields now reach the
  fingerprint that did not before: model name, artifact digest, dimension, both encoder modes,
  normalization, instruction version, chunker version, context version and dependencies.

  **What you have to do.** Every `index_fingerprint` already stored in chunk metadata is now
  stale, so the next `recall index` over an existing corpus re-reads, re-chunks and **re-embeds
  every file, once**. Measured: the default `FastEmbedEmbedder()` moves from `a93f4428…` to
  `1832d370…`. There is no cache to soften this on the shipped paths, because nothing in `recall`
  or `recall_mcp` passes an `EmbeddingCache` to the `Indexer`; only the benchmarks do. Budget the
  re-embed on a metered embedder accordingly, and prefer to run it deliberately rather than
  discovering it inside an unattended job. Nothing is lost if you do not: the corpus keeps serving
  its existing vectors until it is re-indexed, since `index_fingerprint` is only ever compared to
  itself.

  **New re-index triggers, deliberately.** `dependencies` carries the inference-library version and
  the ONNX execution providers, so a fastembed upgrade or a CPU-to-CUDA move now re-fingerprints
  the corpus. That is the trade `EmbeddingProfile.fingerprint` already makes for the cache, for the
  same stated reason: a runtime change is free to move the last bits of a vector, and neither a
  cache nor a skip guard can tell. The two agree now instead of disagreeing.

  **`ContextPolicy.max_tokens` is now covered too**, in the same change and for the same reason.
  It selects a different rung of `contextual_passages`' degradation ladder, so two policies
  differing only in it build different passages and used to hash equal. It had been carried as a
  known gap on one ground only, that closing it re-fingerprints every corpus, and that cost is
  being paid here regardless; deferring it again would have charged a second full re-embed later
  for a one-term change. No shipped path sets it (`context_policy_for_profile` leaves it unset),
  so this widens the identity without moving any shipped corpus further than the paragraph above
  already does.

  ⚠️ `ContextPolicy.tokenizer` is deliberately **not** covered, and that is a decision rather than
  an oversight. It changes the passage exactly as `max_tokens` does, but a callable has no identity
  stable across processes: `__qualname__` collides for closures and lambdas and `id()` differs
  every run. An unstable term is far worse than a missing one, because the fingerprint would differ
  from itself and re-embed the whole corpus on every single run, silently and permanently. Closing
  it needs a caller-supplied stable tokenizer identity. A test pins the current behaviour so the
  limit is recorded rather than assumed.

### Fixed

* **An embedder built without a registered profile claimed `bge-small-symmetric-v1` whatever model
  it actually was.** `FastEmbedEmbedder.__init__` minted that literal (or `bge-small-asymmetric-v1`)
  unconditionally on the no-identity path, so the id was a label rather than a claim. Measured
  2026-08-18: a `fastembed:BAAI/bge-large-en-v1.5` embedder reported `dim=1024` under
  `profile_id='bge-small-symmetric-v1'`, whose registry entry is 384-dimensional, and a production
  corpus of 8,716 chunks had stored that pairing in its chunk metadata. The fallback id is now
  derived from the model name, dimension and encoder modes
  (`unregistered__BAAI__bge-large-en-v1.5__1024__symmetric`) unless the embedder genuinely is the
  model the legacy literal names, at the width the registry declares for it. The `/` is replaced
  because a profile id is interpolated into a result filename by
  `recall.eval.promotion.run.ArmConfig.key`, the same reason `SparseProfile` already did this.

  The embedding cache was never affected: `EmbeddingProfile.fingerprint` already covers
  `model_name` and `dimension`, so the two models keyed apart despite sharing an id. What was
  affected is `recall.index._index_fingerprint`, which hashes the profile id alone with no
  dimension term of its own. A bge-small corpus and a bge-large corpus therefore produced the
  **same** fingerprint for the same file, so the incremental skip guard treated a model swap as a
  no-op and left stale vectors in place. Verified by execution before the fix: the 384-dimension
  and 1024-dimension fingerprints were equal.

  ⚠️ **Scope of the change, which is narrower than it looks.** The default `FastEmbedEmbedder()`
  is bge-small at 384 and keeps `bge-small-symmetric-v1`, so its index fingerprints are
  byte-identical to before (verified) and no default corpus needs re-indexing. Registered
  enterprise profiles never reach this path at all. Only a corpus indexed with an *unregistered*
  model changes id, and for those the change is the repair: the next `recall index` run sees a
  different fingerprint and re-embeds, replacing vectors whose recorded provenance was false.
  Search keeps working in the meantime, because the stored `embedding_profile` metadata is
  reported as a diagnostic and never compared at read time. One thing to re-do deliberately: a
  calibration file fitted for such a corpus was written under the wrong id and will stop
  resolving, which is correct (a bge-small-keyed threshold does not transfer to bge-large cosines)
  but shows up as "uncalibrated" until it is re-fitted.

* **`recall generation build` recorded an overlap the chunker never used, and correcting it moves
  the pipeline fingerprint.** The chunker clamps overlap to `max_chars // 4`; the generation's
  `ChunkerIdentity` recorded what was asked for. So the record described a pipeline that did not
  run, and it was reachable with default arguments: `--max-chars 200` with the default overlap of
  80 chunked at 50 and recorded 80. Two configurations producing byte-identical chunks therefore
  fingerprinted differently, and a calibration binds to that fingerprint.

  ⚠️ **This is a deliberate break on rebuild, in exactly one region.** The recorded value changes
  only when `overlap > max_chars // 4`, which at the default overlap means any `--max-chars` below
  320. The default 800/80 is unchanged. Generations already in a database are immutable and
  unaffected, but rebuilding such a corpus with identical flags now yields a different
  `pipeline_fingerprint`, which costs the cross-generation chunk reuse keyed on that column and
  the binding of any calibration measured against the old generation. Re-run
  `recall calibration calibrate --publish` against the new generation. The pre-upgrade record was
  false, so there is no version of this that is both correct and non-breaking. `recall index` is
  unaffected: it takes neither flag and builds no pipeline identity.

* **The MCP server now honours `RECALL_TRUST_MODE`.** `docs/USING_WITH_CLAUDE.md` has told users to
  set it since the document was written, and it did nothing: the variable appeared nowhere in
  `recall_mcp`, and `search_memory` and `evidence_memory` were both called without `policy=`, so the
  service applied its strict default. Following the documented first-run path therefore produced
  `INDEX_NOT_READY` on every `recall_search` against a freshly indexed corpus, with the one
  documented remedy inert. The CLI honoured the same variable throughout, which is what let the gap
  survive: the same setting worked in one entry point and was silently ignored in the other.
  Strict remains the default, a misspelling such as `developmnet` still stays strict, and a relaxed
  server now logs a warning at every start rather than degrading quietly.

### Added

* `recall extract run|show --status-vocabulary W,X,Y` lets a corpus that states status in its
  own words, not the shipped memo set, be measured without every such claim being refused at a
  batch rung. It does not widen what `recall rewrite` may write: the write path still extracts
  under the shipped vocabulary and `route_relation` still refuses anything outside it.
* Promoted deterministic answer slot selection to a supported optional evidence path. Public
  `AnswerSlot` and `EvidencePolicy` exports, plus the LangChain and LlamaIndex evidence adapters,
  can require multiple answer components and abstain with `answer_slot_gap` when one is absent.
  Existing retrieval ordering remains the default. Beam selection and reasoning remain opt in.

## [0.9.5] (2026-08-15)

### Added

* Added model backed truth extraction: `recall/truth_extraction/`, turning memo prose into
  structured, quoted claims behind a refusing validation ladder. Off unless
  `RECALL_TRUTH_EXTRACTION=1`, runs on the ingest path only, never the query path. The
  extraction engine is a port with two implementations, a deterministic rules reference and an
  OpenAI compatible model engine (`pip install "recall-rag[extract]"`); whatever an engine returns
  clears the same ladder, so a model gains no ability to skip a rung.
* Added `recall extract run|show`, which reads a corpus and writes nothing, and
  `recall rewrite plan|apply|reject|verify`, which declares reviewed claims in corpus
  frontmatter. `recall rewrite apply` is a dry run by default and requires `--reviewer` and
  `--note` as argparse requirements, so the named human gate fires before any code runs.
* Added the `recall_rewrite_plan` MCP tool, read only. There is deliberately no
  `recall_rewrite_apply`: the MCP client is the model, and a reviewer id it can type is a field
  rather than a person. `recall_reasoning_proposals` and `recall reasoning proposals` gain
  `include_extracted`, defaulting to off so existing behaviour is byte identical.
* Added `recall extract run --cache PATH`, a persistent SQLite extraction cache, so re-ingesting
  an unchanged memo does not re-pay the engine for it. Entries are keyed on engine identity,
  engine revision, prompt revision, the file, its body and the corpus names, so an answer
  produced under one engine is never served for another. A path that is not a usable cache is
  refused before any engine call, a corrupt row is a miss and is re-paid, and a failed write is
  counted and reported rather than discarding the files already extracted. `--cache` was briefly
  a boolean because an earlier version accepted a PATH and ignored it; the flag came back when
  the persistence did. See [docs/EXTRACTION_CACHE_DESIGN.md](docs/EXTRACTION_CACHE_DESIGN.md).

### Fixed

* Fixed `recall extract run` aborting a whole corpus on one filename that is not valid UTF-8.
  A POSIX name arrives as a lone surrogate through `Path.glob`'s surrogateescape, and it raised
  twice: once hashing the cache key, which is computed for every file whether or not a cache is
  in use, and again printing the report, because reconfiguring stdout's encoding resets its
  error handler to strict. The first discarded every file already extracted; the second threw
  away a completed extraction at the last step, exiting 1 with empty output. Such a name is now
  reported with its bad bytes escaped.

### Changed

* **BREAKING: `PROPOSAL_SCHEMA_VERSION` moved from 1 to 2**, which rewrites **every** `ip_`
  proposal id in existence, including the checked in `results/reasoning_session3_proposals.json`.
  Version 2 adds `declares_validity` and `declares_status` to `ProposedRelation`, because a
  document asserting something about ITSELF is not a relation between two documents and forcing
  it into `references` would put a false relation into an audit record. The bump is the point:
  an id minted under a vocabulary that could not express validity must not be mistaken for one
  minted under a vocabulary that can. Anyone holding stored `ip_` ids must re-derive them.

### Fixed

* MTRAG Task B and C generation no longer scores an answer that the token ceiling cut off.
  `benchmarks/mtrag/generation.py` sent `--max-tokens` (512 by default) and never read
  `finish_reason`, so a truncated completion was written to the submission and judged as if the
  system had produced it. It now raises `CompletionTruncated`, unretried because the same ceiling
  cuts every further attempt, and the existing per-task quarantine keeps the task out of the
  submission and in the failures log.

### Added

* `recall setup` gains an optional reasoning arm step, asked after the entailment judge question
  and before the CLAUDE.md scaffold question. Answering yes writes four new environment
  variables: `RECALL_REASONING`, `RECALL_REASONING_MODEL`, `RECALL_REASONING_BASE_URL`, and
  `RECALL_REASONING_API_KEY`. Answering no writes `RECALL_REASONING=0` and nothing else, so
  "switched off" and "never configured" stay distinguishable in `.env`. The shipped reasoning
  tools do not read these variables yet; this writes the settings for a port the reasoning arm
  will use once it is built. See
  [docs/REASONING_MODEL_SELECTION_DESIGN.md](docs/REASONING_MODEL_SELECTION_DESIGN.md).

### Fixed

* Corrected `recall setup`'s refusal message for an embedder whose vector width conflicts with a
  table that already holds data. It previously pointed at a remedy that failed identically to the
  original problem. It now stops and tells you to choose an embedder matching the existing
  schema, or point setup at a fresh table name or database.

## [0.9.4] (2026-08-12)

Released straight from 0.9.2. 0.9.3 is deliberately skipped and will never exist.

Most of this release came from walking the documented quickstart on a clean machine as a new
user would, which found five defects on the path every new user takes.

### Added

* `recall setup` offers bge-base (768 dims) and bge-large (1024 dims) beside bge-small, each
  gated on having room for its own weights rather than on the shared download floor.
* `recall setup` refuses an embedder whose vector width does not match the table it will write
  to, naming the schema command that fixes it. Previously the mismatch surfaced on the first
  write, after the model had downloaded and the corpus had been read.
* `recall setup` lists retrieval options this machine cannot run yet, marked `(not installed
  yet)`, and prints what to install when one is chosen. They were previously hidden, which made
  the feature look absent and left no way to ask for it.
* `recall setup` scaffolds a `CLAUDE.md` and a `memory/` directory for the project, and indexes
  that directory once it exists.
* Added provider execution metadata for reasoning diagnostics and benchmark artifacts, including
  provider id, model id, model revision when available, token counts, latency, and monetary cost
  when providers expose it.
* Added a reviewed inference proposal promotion workflow with separate promoted fact records.
* Added experimental reasoning release notes covering opt in use, provider neutrality, citation
  constraints, CLI and MCP migration notes, serialized fields, limitations, and evaluation posture.

### Changed

* PostgreSQL 18 compliance is now declared and tested: local Docker uses
  `pgvector/pgvector:pg18`, CI runs schema migrations on PostgreSQL 16, 17, and 18, and the main
  integration job runs on PostgreSQL 18.
* Ported the MCP server to MCP Python SDK 2.x and raised the `mcp` extra floor to `mcp>=2,<3`.
* Raised the development Ruff range to `ruff>=0.16,<0.17` while keeping the prior lint baseline
  explicit in `pyproject.toml`.
* The shipped calibration sample now holds twenty answerable and twenty unanswerable queries.
  It previously held fourteen and five, below the certification floor, so following the
  documented path produced an explicitly uncertified threshold.
* The quickstart no longer assumes a clone. The compose file is inline, so `pip install` alone is
  enough to follow it, and the sample corpus that ships beside `recall/eval/queries.json` is now
  pointed at rather than left for the reader to discover.

### Fixed

* The quickstart's schema command failed on every fresh database. It passed a custom `--table`,
  but global migrations must be applied through the default target first, so a new user following
  the README exactly got `SchemaTooOld` from inside the library.
* The terminal video renderer writes its GIF before attempting the MP4, so a missing optional
  dependency no longer discards an asset that never needed it, and reports which file it did not
  write and why.

## [0.9.2] (2026-08-10)

### Added

- Added official MCP Registry metadata in `server.json`.
- Added the PyPI ownership marker required by the MCP Registry.
- Added the `recall-mcp` console script for registry clients using `uvx`.

## [0.9.1] - 2026-08-10

### Fixed

- Rebuilt the package description from the current GitHub README so PyPI no longer shows the stale
  pre-cleanup product copy or the incorrect MIT license sentence from the old buyer table.

## [0.9.0] - 2026-08-09

### Added

- Multi-query fusion through `HybridRetriever.search_fused(query, history, k, source)`.
- Reachable evidence boundary through package exports, CLI `--evidence`, MCP `recall_evidence`,
  and LangChain and LlamaIndex adapter methods.
- Request-time retrieval budgets, overload refusals, stage timings, and profile-sized MCP worker
  pools.
- FastEmbed resolved-provider reporting and provider-aware profile fingerprints.

### Changed

- Fast and quality retrieval profiles now carry separate concurrency and queue budgets.
- The quality reranker is pinned by artifact digest and refuses mismatched local trees.
- FastEmbed profile fingerprints changed, so profile-bound calibrations should be re-fitted.

### Fixed

- Restored package entry points, README sections, dependency declarations, and release smoke
  coverage that had been dropped during a merge.
- Split duplicate benchmark helper modules so pytest no longer collects colliding test names.

Full release detail: [docs/archive/CHANGELOG_FULL.md](docs/archive/CHANGELOG_FULL.md).
