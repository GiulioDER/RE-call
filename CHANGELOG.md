# Changelog

This file keeps the release surface short. The full historical changelog lives at
[docs/archive/CHANGELOG_FULL.md](docs/archive/CHANGELOG_FULL.md).

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versioning is pre-1.0
`0.MINOR.PATCH`, so a minor bump may still break schema or API.

## [Unreleased]

### Added

* Added model backed truth extraction: `recall/truth_extraction/`, turning memo prose into
  structured, quoted claims behind a refusing validation ladder. Off unless
  `RECALL_TRUTH_EXTRACTION=1`, runs on the ingest path only, never the query path. The
  extraction engine is a port with two implementations, a deterministic rules reference and an
  OpenAI compatible model engine (`pip install recall[extract]`); whatever an engine returns
  clears the same ladder, so a model gains no ability to skip a rung.
* Added `recall extract run|show`, which reads a corpus and writes nothing, and
  `recall rewrite plan|apply|reject|verify`, which declares reviewed claims in corpus
  frontmatter. `recall rewrite apply` is a dry run by default and requires `--reviewer` and
  `--note` as argparse requirements, so the named human gate fires before any code runs.
* Added the `recall_rewrite_plan` MCP tool, read only. There is deliberately no
  `recall_rewrite_apply`: the MCP client is the model, and a reviewer id it can type is a field
  rather than a person. `recall_reasoning_proposals` and `recall reasoning proposals` gain
  `include_extracted`, defaulting to off so existing behaviour is byte identical.

### Changed

* **BREAKING: `PROPOSAL_SCHEMA_VERSION` moved from 1 to 2**, which rewrites **every** `ip_`
  proposal id in existence, including the checked in `results/reasoning_session3_proposals.json`.
  Version 2 adds `declares_validity` and `declares_status` to `ProposedRelation`, because a
  document asserting something about ITSELF is not a relation between two documents and forcing
  it into `references` would put a false relation into an audit record. The bump is the point:
  an id minted under a vocabulary that could not express validity must not be mistaken for one
  minted under a vocabulary that can. Anyone holding stored `ip_` ids must re-derive them.

### Added

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
