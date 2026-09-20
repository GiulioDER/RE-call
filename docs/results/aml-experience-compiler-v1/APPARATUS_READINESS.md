# AML engineering experience compiler apparatus readiness

Date: 2026-09-17

Status: deterministic apparatus ready; no provider-backed retrieval or executable-task
measurement started

## Bound implementations

- RE-call implementation commit:
  `d45610d42d74d0aaca8cbee2d6aa77e24de79661`
- agent-memory-bench replay implementation commit:
  `4a0e2c06344fcee74e67d8b6ce9554b6c7b1e00a`
- RE-call `uv.lock` SHA256:
  `c7d0396f68d2ad9de8e05bcbc5b3286e588fd9eda8b3250897c17eb20c4d672e`

The replay recognizes `E0_raw`, `E1_compiled`, and `E2_compiled_raw`, checks the exact served
variant before deleting or adding corpus data, and rejects drift in the corpus, task population,
served commit, model identity, prompt digests, task queries, and post-retrieval label hashes.

## Deterministic evidence

RE-call checks:

- `41 passed, 1 skipped` in `tests/test_aml_hosted.py`
- `12 passed` across release manifest, promotion gate, and hosted preflight tests
- `29 passed, 21 skipped` in `tests/test_store.py`
- Ruff, Python compilation, dependency lock consistency, and whitespace checks passed

The skipped tests require the registered PostgreSQL integration environment and were not reported
as passes.

Benchmark checks:

- `18 passed` across the hosted adapter, replay, existing A-arm selector, and new experience
  representation selector
- Ruff, Python compilation, and whitespace checks passed

Red proof receipts cover arm isolation, exact source spans, unsupported factual fields, source
session concentration, E-arm promotion and fallback, and paired product identity.

## Measurement hold

The local Voyage credential is stale. VPS2's configured Voyage credential passed the RE-call
serving verification, but these two implementation commits have not been deployed there. Starting
a provider-backed replay from the local checkout would therefore either fail authentication or
measure old code. No such run was started.

The next valid measurement step is to deploy the bound RE-call commit to an isolated VPS2
evaluation service, expose one E variant at a time with separate purged namespaces, and produce
the three immutable replay artifacts before running the mechanical selector.
