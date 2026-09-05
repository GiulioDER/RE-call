# RE-call 0.12.0 release preparation

Status: draft for review.

## Scope audited

This analysis uses the latest release tag and the current remote release line:

* Previous release: `v0.11.0`, commit `68e862f0`, dated 2026-08-29.
* Candidate source: PR [#583](https://github.com/GiulioDER/RE-call/pull/583), merged commit
  `b053994f18f0a2f61c267d784e868bf3bc15923c`, dated 2026-09-02.
* The audited range is `v0.11.0..origin/master` plus the merged PR #583 changes. The original
  remote tip contributed 25 first-parent commits; PR #583 contributes seven commits and 25 changed
  files.

The current checkout is `codex/image-toggle`, seven commits behind `origin/master`, and has
substantial uncommitted and untracked work. This document intentionally does not treat that work
as part of the release candidate. It is a safe review artifact until the branch is reconciled.

Audit commands:

```text
git log --first-parent --date=short --pretty=format:'%h %ad %s' v0.11.0..origin/master
git diff --stat v0.11.0..origin/master
git diff --name-status v0.11.0..origin/master
```

## Proposed public changelog

The following is the proposed `CHANGELOG.md` section for a provisional `0.12.0` release.

## [0.12.0] (2026-09-02)

### Added

* **Graph RAG serving is now integrated and explicitly bounded.** The deterministic Evidence Graph
  V1 remains separate from the authored supersession reasoning graph. `graph_expansion=one_hop`
  traverses the semantic graph only, starts from trusted retrieval, follows permitted directional
  relations, re-evaluates every candidate through the ordinary trust layer, and adds only trusted
  evidence. Graph metadata and model proposals never become evidence by themselves.

* **Graph expansion now performs its admission checks before projecting the graph.** Readiness,
  generation binding, trusted seed availability, and the selective gate are reported as whole
  expansion refusals. Candidate rejection counters describe only candidates that were actually
  discovered. A gated request therefore avoids the full graph projection and reports zero graph
  diagnostics when no graph was inspected.

* **The graph contract is documented precisely.** The reasoning graph and semantic graph now have
  separate schemas, relation vocabularies, consumers, and diagnostics. Automatic extraction creates
  `references` edges from links and wikilinks. Other semantic relations require explicit
  `recall_graph` declarations. `supersedes` is enforced upstream by trust and is not a semantic
  graph relation.

* **The opt in reasoning answer path is reachable.** The Python reasoning API, CLI, and MCP tool
  can receive an answer provider while the model free audit path remains model free. The release
  adds an OpenAI compatible answer provider for `/chat/completions`, preserves the local Ollama
  adapter, bounds paid answer calls, and routes from ordinary retrieval only when a blocked or
  superseded result makes reasoning actionable and an answer backend is actually configured.

* **Content addressed embedding reuse is enabled across indexing entry points.** The shared cache
  is bounded by `RECALL_EMBED_CACHE_MAX_MB`, defaults to 512 MB, uses packed float32 vectors, has
  LRU eviction, deduplicates repeated inputs, and degrades to a re-embed when cache storage fails.
  It is used by CLI indexing, generation builds, MCP ingestion, session close indexing, seeding, and
  setup. Benchmarks can still opt out by constructing an indexer without a cache.

* **Claude Code hooks can synchronize project memory to a hosted corpus.** Hosted credentials use
  the OS keychain where available and a protected file fallback where needed. Login, logout, refresh,
  access token caching, MCP transport, failure classification, retry policy, pending sync notices,
  and session end synchronization are included. Both project memory roots are considered, and the
  hook never deletes or rewrites the local source files.

* **Hosted synchronization can plan before uploading.** `recall_inventory` exposes source names and
  raw content digests for a tenant. The hosted planner compares those digests with local files and
  uploads only changed or unknown files. Truncated inventories are refused, deletion is opt in,
  and the planner itself performs no network or database operation.

* **A `UserPromptSubmit` memory hook is available through setup and the Claude plugin.** It uses
  local memo files and bounded BM25 ranking before a turn starts, so it can surface a prior decision
  before a plan is formed. It is separate from the write time hook, remains fail open, and has no
  database, network, or embedder dependency.

* **Codex is now a first class RE-call integration.** `recall setup` detects Codex and installs a
  packaged Codex plugin, MCP server, shared hook configuration, and the same durable memo contract
  used by Claude Code. The five lifecycle hooks cover startup, resume, clear, compact, prompt time,
  write time, pre compact refresh, and session end refresh. The Codex adapter delegates prompt and
  write handling to the shared implementation, so thresholds, project discovery, and fail open
  behavior stay aligned across both clients. The wheel force includes the Codex bundle, and the
  integration has its own installation and authenticated server coverage.

* **Hosted uploads preserve relative paths and converge on re-ingest.** Path traversal, absolute
  paths, Windows device names, invalid components, and length hazards are refused. Repeated uploads
  of the same source no longer duplicate the active corpus, and a changed source can supersede its
  previous content.

### Changed

* **Reasoning and retrieval descriptions now expose the actual decision boundary.** Retrieval does
  not silently escalate into model execution. It emits a library authored `NEXT:` recommendation
  only for non gap trust blocks or superseded matches, and only when the configured tool surface
  can answer.

* **Open indexing is safer on constrained hosts.** The default outer embedding batch is 64 chunks.
  Allocation failures identify `RECALL_INDEX_BATCH_CHUNKS` as the corrective setting. HTTP
  transports default to stateless MCP mode, with `RECALL_MCP_STATELESS` available when session state
  is required.

* **The session serving scripts now have a read only `verify` handshake.** It checks the server a
  session will actually launch, reports tenant and generation boundaries, avoids deployment state
  changes, and cleans up timed out child processes. The corpus status command now distinguishes
  database state from the process and tool surface a client will use.

* **Write time hook lifecycle and project binding are explicit.** The relay is integrated per
  session, setup passes the selected project root consistently, and the single machine wide hook
  configuration reports when a later project installation moves that binding.

* **Dependency maintenance was refreshed.** The release line updates the Claude Agent SDK,
  Pydantic, LangChain Core, OpenAI, and python dotenv pins, caps pypdf at the supported major, and
  holds the corresponding Dependabot ranges against the tested compatibility window.

### Fixed

* **Graph expansion no longer spends several seconds projecting a graph that the admission gate
  will discard.** Readiness retains precedence, and graph refusal counters are no longer reported
  as candidate rejection counts.

* **The reasoning answer port is no longer dead code.** Configured CLI and MCP reasoning queries can
  invoke the selected answer provider, while audit commands cannot spend a model call. Provider
  failures remain structured reasoning outcomes rather than leaking as unclassified exceptions.

* **MCP trust refusals now reach clients as structured tool errors.** The refusal code, calibration
  state, tenant, and generation remain visible instead of being flattened into an empty result or
  an opaque server failure.

* **Indexing now reports the actual allocation remedy.** The previous outer batch default could let
  the embedder request a large padded allocation and fail part way through a run. The failure now
  names the batch setting and leaves the existing batch intact.

* **A current state projection now marks unresolved supersession references as ambiguous.** Tests
  exercise both producer driven and direct consumer paths, so the guard is no longer reachable only
  in theory.

* **A prompt hook no longer runs an unrelated checkout by accident.** The installed module path is
  deterministic, missing optional event modules degrade to silence, and the deployed hook copy is
  checked for drift.

* **Hosted credential caches are account bound.** Switching between two configured accounts no
  longer risks sending one account's access token with the other project's memory. Login and logout
  now have distinct user facing behavior from silent hook events.

* **The credential screen is fail closed for new hosted uploads without disclosing matched text.**
  A scanner failure is reported as a broken screen rather than as a false positive. The measured
  corpus run found no matches in 1,383 real memos, which demonstrates noise on that corpus only and
  is not evidence that the corpus is safe from undiscovered credentials.

### Evaluation and limits

* The graph work is a serving and trust boundary improvement, not a new retrieval quality claim.
  The prior graph first experiment remains closed after rescuing zero of 15 frozen misses, and the
  answer provider has no quality evaluation beyond one correct live call. Graph marginal
  contribution and reasoning routing take up remain unmeasured.

* The committed refrozen scope probe remains an evaluation artifact, not a promoted feature. Both
  folder and facet arms rescued zero of 15 frozen misses. Facet scoping retained 23 of 31 controls,
  while baseline retained 19 of 31. Those results do not justify automatic scope selection or a
  change to the default retrieval path.

* Hosted credential screening measured zero findings on the author's corpus. Its positive controls
  caught planted fixtures, but there is no labelled external corpus, so the release must not claim
  complete secret detection.

## Graph RAG analysis

### What materially improved

The graph implementation now has a clear two structure model:

* The reasoning graph projects source and chunk lineage, especially authored supersession, for
  `recall_reasoning_projection` and current state.
* The semantic graph projects entities, mentions, diagnostics, and typed relations for
  `graph_expansion=one_hop`.

That distinction fixes the most important documentation and observability ambiguity. A healthy
authored supersession edge count does not imply that one hop has semantic relations to inspect.
Supersession is already enforced by the trust layer, while semantic traversal uses the relation
policy in `recall.semantic_graph`.

The one hop path now follows this sequence:

1. Read graph readiness.
2. Inspect the already available trusted retrieval seeds and the retrieval gap signal.
3. Refuse the whole expansion when admission is not satisfied.
4. Project and load the generation bound semantic graph only when admission passes.
5. Apply generation and relation direction checks.
6. Rank candidates with query cosine, seed corroboration, relation support, confidence, and a
   deterministic tie breaker.
7. Re run ordinary trust evaluation for every candidate.
8. Append only trusted candidates to the result.

The important release improvement is not only the order. The result now distinguishes a candidate
that was discovered and rejected from an expansion that never started. That makes diagnostics,
metrics, latency analysis, and future experiments interpretable.

### What did not improve and must not be claimed

The semantic graph is conservative by design. Links and wikilinks produce `references`; the other
relation kinds require explicit frontmatter declarations. It does not infer support or contradiction
from prose. The graph first quality result remains a null result, and the new release contains no
new positive retrieval measurement. The correct product claim is safer graph assisted evidence
expansion, not higher recall.

### Graph related commit coverage

| Commit | Release relevance |
|---|---|
| `10d43124` | Wires the reasoning answer port, provider adapter, bounded answer calls, and retrieval routing. |
| `d146a133` | Makes unresolved supersession state observable and tested. |
| `f63b9981` | Separates the two graphs and documents the actual one hop structure. |
| `d64ab5f2` | Moves graph admission before projection and separates expansion refusals from candidate rejections. |
| `fd7d8171` | Makes hosted source identity stable, enabling generation supersession rather than duplication. |
| `6f2ab6f8` | Adds the inventory needed by a hosted sync client to compare source digests. |
| `c643aa00` | Bounds open indexing and preserves trust refusal payloads through MCP. |

## Complete post release commit coverage

| Commit | Area | User facing result |
|---|---|---|
| `f653f223` | Evaluation | Refrozen folder and facet scope probe, with the falsified prediction recorded. |
| `3918a130` | Release tooling | Session close no longer reports completed preregistrations as pending. |
| `8f6b5d1f` | Indexing | Content addressed embedding cache serves every user facing indexing path. |
| `64626621` | Claude Code | UserPromptSubmit local memory retrieval is installed and documented. |
| `492501c6` | Operations | Read only serving verification handshake and safer session wrappers. |
| `10d43124` | Reasoning | Answer provider wiring, OpenAI compatible backend, routing, and rate bound. |
| `d146a133` | Trust state | Unresolved supersession references become observable ambiguous state. |
| `f63b9981` | Graph documentation | Exact graph traversed by one hop and actual relation extraction are documented. |
| `d64ab5f2` | Graph serving | Admission runs before projection and refusal accounting is corrected. |
| `fd7d8171` | Hosted ingestion | Relative paths are preserved and duplicate re ingest converges. |
| `6f2ab6f8` | MCP API | `recall_inventory` exposes raw source digests for sync clients. |
| `238e422b` | Dependencies | MCP and transformers ranges are held against tested caps. |
| `265baa62` | Hosted sync | Pure planner decides changed, unknown, and optional deletion work. |
| `1cd6c99e` | Dependencies | pypdf is capped at its supported major. |
| `fbd813b2` | Hosted sync | Credentials, transport, retry classification, and SessionEnd synchronization land. |
| `691cfe5d` | CI | Deploy guard enumerates the full package rather than a fixed module subset. |
| `f0a423c0` | Dependencies | python dotenv is refreshed. |
| `570a6eb1` | Dependencies | OpenAI is refreshed. |
| `ceae7336` | Dependencies | LangChain Core is refreshed. |
| `2280bca0` | Dependencies | Pydantic is refreshed. |
| `575d804c` | Dependencies | Claude Agent SDK is refreshed. |
| `5b76c57c` | Security | New hosted uploads are screened for credential shaped content. |
| `c643aa00` | Reliability | Safer indexing defaults and structured MCP trust errors. |
| `b0bb6286` | Evaluation | Pinned query construction prompt factor benchmark apparatus is prepared. |
| `2aca74ef` | Claude Code | Relay lifecycle, connection reuse apparatus, and project scoping are hardened. |
| `#583`, merge `b053994f` | Codex integration | Automatic Codex plugin, MCP, skills, lifecycle hooks, shared memo contract, and installation tests. |

## Release blockers and next steps

1. Reconcile this candidate with the dirty checkout in an isolated worktree. Do not reset or clean
   the current branch.
2. Decide whether the provisional version should be `0.12.0`. The graph and reasoning additions are
   opt in, but the minor bump is appropriate for the scale of the new API and schema surface under
   the project's pre 1.0 versioning rule.
3. Merge the proposed public section into the candidate branch's `CHANGELOG.md`, then add the
   `0.12.0` heading only after the content is accepted.
4. Run the release preflight from the candidate branch. `scripts/release.py` requires a clean tree,
   a complete changelog section, and synchronized version sites. It updates tracked version sites
   but does not commit, tag, push, or publish.
5. Verify all distribution paths: PyPI wheel contents, `recall setup`, Claude plugin hooks, and MCP
   registry metadata. The registry publication remains a separate manual workflow after PyPI.
6. Run the candidate tests and release smoke checks from the reconciled branch. A green test suite
   is not enough without checking the built wheel and the registry pin.
