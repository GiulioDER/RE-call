# Changelog

This file keeps the release surface short. The full historical changelog lives at
[docs/archive/CHANGELOG_FULL.md](docs/archive/CHANGELOG_FULL.md).

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versioning is pre-1.0
`0.MINOR.PATCH`, so a minor bump may still break schema or API.

## [Unreleased]

### Measured

* **Truth extraction is a reviewing aid, and the pre-registration says so before the numbers do.**
  The prose extraction experiment is scored in
  `results/truth_extraction/PREREGISTRATION-prose-extraction.md` and summarised as `RESULTS.md`
  §13. It is a **negative result**. The load-bearing public prediction, P10, registered that the
  model would refuse all four failure fixtures transplanted from the private corpus; it proposed
  an edge on two of them, and the registered decision table caps the feature at reviewing aid on
  that alone, whatever else holds. `recall rewrite apply` keeps its human gate for this reason
  rather than by caution.
* **Neither arm can be given a tier, and neither precision prediction is falsified.** R1 decided 8
  proposals and M1 decided 2 against a registered floor of 10, so both artifacts read
  `UNDERPOWERED`: "could not tell", not "the model is bad". Both intervals still overlap their
  predicted ranges. An earlier write up reported P1 as falsified by comparing R1's Wilson *upper*
  bound against 0.70, which is the decision rule's gate on the Wilson *lower* bound and not P1's
  floor of 0.60. That claim is withdrawn in both files.
* **The finding worth carrying is a mechanism, not a rate.** The model's characteristic error is
  reading a claim about something *inside* a document as a claim about the document. Both of its
  false positives on the adjudicated rows and both of its fixture proposals are partial scope:
  four for four across two corpora sharing no text, and the same error the rules it replaces made
  on the private corpus.
* Recall on PEPs is published as a **corpus fact** rather than a model result: 47 authored header
  edges, 8 restated in prose at either end, and only 3 restated by the document the extractor is
  actually given, so the operative ceiling is 0.064. Documents with a structured field for a
  relation use the field.

### Fixed

* The fixtures result is published as **P10**, the prediction it scores, rather than P7, which is
  a different registered prediction on a different instrument. The artifact is renamed
  `arm_P10_fixtures.json`, and `tests/test_prereg_authority.py` now derives the identifier from
  the pre-registration instead of carrying it from one line of code to the next.
* Invariant I5 ("labels frozen before arms, checked by the runner") is asserted for the first
  time. It was previously unassertable: the pre-registration and the results lived on two branches
  neither of which contained the other. The runner now refuses to start when the pre-registration
  is unreadable or the gold manifest digest has moved, and refuses to write an artifact generated
  at or before the pre-registration was authored.
* Corrected three documentation claims the shipped code contradicts: the truth extraction design
  document's "not yet implemented" status, "`status` is routable but no relation emits it yet" in
  three places, and `REASONING_OPERATIONS.md`'s claim that no proposal is promoted to corpus
  metadata by the CLI (`recall rewrite apply --apply` writes frontmatter, under a human gate).

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
