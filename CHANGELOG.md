# Changelog

This file keeps the release surface short. The full historical changelog lives at
[docs/archive/CHANGELOG_FULL.md](docs/archive/CHANGELOG_FULL.md).

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versioning is pre-1.0
`0.MINOR.PATCH`, so a minor bump may still break schema or API.

## [Unreleased]

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
