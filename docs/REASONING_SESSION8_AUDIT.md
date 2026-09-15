# Reasoning Session 8 Audit

Date: 2026-08-10

Base revision: `bfc2739294f8fecac9535df4605702112eb84395`

Branch: `codex/reasoning-session-8`

## Release Decision

Decision: Retrieval plus experimental reasoning.

The release is not ready for a broader beta as a reasoning product. The bounded planner, graph
projection, proposal protocol, CLI commands, and MCP tools are integrated and tested. The measured
controls support traceable reasoning experiments over the defined Session 6 domains, not a general
reasoning claim.

## Audit Result

1. Trust boundary audit: Passed after fixes. The final code rejects mismatched proposal report
   generation and pipeline identity before answer generation, rejects proposal pipeline mismatches,
   keeps proposals outside trusted evidence, and requires trusted citations for answers.
2. Data isolation audit: Passed. Reasoning projection and query preserve tenant and generation
   identity. Source scoped reasoning query builds its graph from trusted source scoped retrieval.
   Legacy projections stay degraded in strict inspection mode.
3. Evaluation audit: Passed for committed deterministic controls. Session 1, Session 3, and
   Session 6 artifacts were regenerated and compared to committed artifacts with no diffs. Session 6
   threshold status reports all checks passed, including heldout, nearest neighbor, shuffled edge,
   and removed edge controls.
4. API compatibility audit: Passed after fixes. Existing retrieval contracts remain unchanged. New
   CLI and MCP reasoning surfaces are additive. New MCP tools now have explicit async, schema, scope,
   and authorization guard coverage.
5. Documentation audit: Passed with this release decision. Claims remain bounded to retrieval plus
   experimental reasoning. Broader reasoning and broader beta claims are rejected.

## Release Blocking Fixes

1. Removed ruff blockers in `recall/eval/reasoning_session6.py`.
2. Fixed mypy blockers in `recall/reasoning.py`, `recall_mcp/service.py`, and `recall/cli.py`.
3. Made `ChunkIterable` accept read only store identity properties.
4. Corrected public MCP proposal item fields so nullable proposal confidence and rule id serialize
   honestly.
5. Added authorization and async guard coverage for the four reasoning MCP tools.
6. Added regression tests for proposal report identity and proposal pipeline boundary checks.
7. Cleaned the local validation environment so `pip check` and direct `pip-audit` both pass.

## Final Validation

1. `python -m ruff check .`: passed.
2. `python -m mypy`: passed, 191 source files checked.
3. `python -m pytest -q --cov=recall --cov=recall_mcp --cov-report=term-missing --cov-fail-under=70`: 3599 passed, 33 skipped, 1 xfailed, coverage 86.67 percent.
4. `python -m pytest -q tests/test_mcp_concurrency.py tests/test_mcp_tool_authorization.py tests/test_reasoning_api.py tests/test_reasoning_session7.py`: 113 passed.
5. `python -m pytest -q tests/test_embeddings_cloud.py::test_voyage_roundtrip tests/test_learned_sparse.py`: 15 passed, 1 skipped.
6. `uv lock --check`: passed.
7. `uv export --all-extras --no-emit-project --format requirements-txt -o requirements.lock.txt` followed by `uvx pip-audit --requirement requirements.lock.txt --no-deps`: passed.
8. Direct `python -m pip_audit`: passed.
9. `python -m pip check`: passed.
10. `python -m build --outdir dist_session8` followed by `python -m twine check dist_session8\*`: passed.
11. Fresh wheel import check: passed for `recall`, `recall_mcp`, `trusted_search`, `reason`, and
    `ReasoningRequest`.

## Limitations And Roadmap

1. General reasoning is not claimed. The current evidence is a deterministic, frozen control suite
   and targeted integration tests.
2. Broader beta is blocked until held out external evaluation and adversarial controls exercise real
   provider backed reasoning, not only offline fixtures.
3. The default MCP reasoning query has no answer provider. It returns traceable reasoning state and
   refusal or review outcomes, not final generated answers.
4. Provider monetary cost is not normalized in the core metrics. Provider adapters must add numeric
   cost fields or metrics before any cost claim is made.
5. Inference proposals remain review candidates. Promotion into trusted corpus metadata needs a
   separate reviewed write path and its own audit.
6. Legacy stores without generation identity remain degraded in strict inspection mode.

## Follow Up Work

1. Add a held out external reasoning benchmark with model revision, corpus fingerprint, generation
   identity, cost, latency, and false positive measurements.
2. Add provider adapters that report monetary cost and model revision as library authored metadata.
3. Add an explicit reviewed promotion workflow for accepted inference proposals.
4. Add release notes that describe the reasoning features as experimental and opt in.
