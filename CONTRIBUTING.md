# Contributing

RE-call is a solo-maintained, pre-1.0 project. Contributions are welcome, but keep the register the
README uses: direct, technical, no marketing — a claim you can't measure is a claim you shouldn't
make.

## Set up

```bash
git clone https://github.com/GiulioDER/RE-call.git
cd RE-call
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"
```

`.[dev]` installs the engine plus `pytest`, `pytest-timeout`, `ruff`, and `mcp` — enough to run the
suite and the linter. Optional extras (`fastembed`, `voyage`, `rerank`, `entail`, `eval`, `finetune`,
`pool`) layer in as needed; see `pyproject.toml` for what each unlocks.

## Run the test suite — needs a real pgvector database

**The tests hit a real PostgreSQL + pgvector instance. There is no mock DB.** Bring one up with
Docker before running `pytest`:

```bash
docker compose up -d --wait      # or: make db-up
pytest -v                        # or: make test
```

`tests/conftest.py` reads `RECALL_TEST_DSN` (falling back to the local dev container at
`postgresql://recall:recall@localhost:5432/recall`) and refuses to run if that DSN is the same as
`RECALL_DSN` or points at a non-local host without `RECALL_ALLOW_REMOTE_TEST_DB=1` — **the suite
`DROP TABLE`s**, so this is a safety check, not friction to work around. Each test gets its own
uuid-named table via the `make_store` / `cli_table` fixtures and cleans it up on teardown; nothing
you own is at risk from a normal `pytest` run against the dev container.

Tests that need an optional extra (the real-model `rerank`/`entail` cases, the one Voyage test)
self-skip when the extra or its API key isn't present — that's why CI's `pip install -e ".[dev]"`
still passes without installing every extra. If you're touching one of those paths, install the
relevant extra locally so your change is actually exercised, not skipped.

## Lint

```bash
ruff check .        # or: make lint
```

CI runs this as a hard gate (`ci.yml`'s `test` job). Line length is 100 (`pyproject.toml`
`[tool.ruff]`), target `py311`.

## Keep `uv.lock` in sync

CI's `audit` job runs `uv lock --check` before scanning dependencies with `pip-audit` — a drifted
lockfile means the audit is scanning stale versions, so it's a hard failure, not a warning. If you
change a dependency in `pyproject.toml`, regenerate the lock before you push:

```bash
uv lock
```

Commit the updated `uv.lock` alongside the `pyproject.toml` change.

## Run the evaluation harness (optional, for retrieval/trust-layer changes)

```bash
make eval                                                     # ablations + trust + near-miss
python -m recall.eval.scale --embedder hashing --filler 50000  # scale + latency
```

`make eval` needs the same Docker Postgres as the test suite and runs key-free with the local
`fastembed` embedder; it adds a Voyage row only if `VOYAGE_API_KEY` is set. It writes
`results/RESULTS.md` and charts — if your change touches retrieval, trust verdicts, or calibration,
re-run it and look at whether the numbers moved before claiming they didn't.

## Before opening a PR

- `ruff check .` and `pytest -v` both pass locally.
- `uv lock --check` passes if you touched dependencies (or you ran `uv lock` and committed the
  result).
- New behaviour has a test that would fail without the change — see the README's "Engineering"
  section for what a *good* regression test in this repo looks like (asserts the invariant a naive
  fix could satisfy vacuously, not just a final count).
- If a claim in the README, `results/FINDINGS.md`, or `docs/WRITEUP.md` changes because of your
  PR — a number moves, a caveat needs updating — update it in the same PR. A stale published number
  is the failure mode this project exists to catch; don't reintroduce it in its own docs.
- Commit messages describe *why*, not just *what* — see `git log` for the house style
  (`type(scope): what changed — the reason`, and `fix: N audit fixes from CCA` for batched
  audit-driven fixes).

## Reporting bugs / requesting features

Use the issue templates — they ask for the PostgreSQL/pgvector version, which embedder, and (for a
retrieval bug) the query plus the returned verdict/confidence, because "search returned the wrong
thing" is nearly unactionable without that.

## Security issues

Do not open a public issue for a security concern — see [SECURITY.md](SECURITY.md) for private
reporting via GitHub Security Advisories.
