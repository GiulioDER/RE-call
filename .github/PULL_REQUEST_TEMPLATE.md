## What this does and why

<!-- The "why" matters more than the "what" here — see CONTRIBUTING.md and the commit-message
     convention in `git log` (`type(scope): what changed — the reason`). -->

## Area touched

<!-- Retrieval / trust layer / embedders / MCP server / CLI / eval harness / multi-tenancy / other -->

## Checklist

- [ ] `ruff check .` passes
- [ ] `pytest -v` passes against a real pgvector database (`docker compose up -d --wait` — no mock
      DB in this project)
- [ ] If a dependency changed: `uv lock` was run and the updated `uv.lock` is included (CI runs
      `uv lock --check` as a hard gate)
- [ ] New behavior has a test that would fail without this change — and the test asserts the actual
      invariant, not a value a shortcut fix could also satisfy (see the README's "Engineering"
      section for examples from this repo)
- [ ] If this changes a published number (README claims table, `results/FINDINGS.md`,
      `docs/WRITEUP.md`) — the doc is updated in this PR, not left to drift
- [ ] If this changes calibration, retrieval quality, or the trust layer — `make eval` was re-run
      and the result reviewed, not assumed unchanged

## Evaluation impact (if applicable)

<!-- Paste the relevant row(s) from results/RESULTS.md or results/SCALE.md before/after, or state
     "no retrieval/trust-layer impact". -->
