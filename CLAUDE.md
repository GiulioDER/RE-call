# recall: working rules

This file is loaded into every session in this repository. It records the rules that have been
paid for in lost work. The maintainer's own operating rules (their hosts, their session tooling,
their research workflow) live in a `CLAUDE.local.md` outside this repository and load alongside
this file on their machine only.

## Research records stay out of this repository

Pre-registrations, run traces and experiment write-ups are kept in the maintainer's private
research archive, not here. What belongs in `results/`, `docs/results/` and
`docs/preregistrations/` is only what a published claim cites: `benchmarks.claim_gate` and
`tests/test_published_numbers_have_artifacts.py` check that every marked number in a published
document resolves to a committed artifact, and that is the bar for adding one. Do not commit a raw
trace, a checkpoint or a run log to make a result reproducible; cite the artifact the claim needs.

## This repository is public: private content never enters the tree

⛔ **A trace of a run over a private corpus must not keep its text.** Experiment traces committed
between 2026-08-25 and 2026-09-16 captured whole private notes as retrieval-candidate text, and
with them host addresses, SSH host-key fingerprints, personal emails and another project's
operational notes; 73 files had to be scrubbed (`docs/results/REDACTION-2026-09-26.md`). Nothing
warned, because GitHub secret scanning looks for credential formats only.

- Record ids, ranks, scores and hashes of text, never the text. `scripts/redact_memo_traces.py`
  shows the placeholder form.
- `python scripts/check_public_tree.py` refuses addresses, fingerprints, server IDs, personal
  mailboxes, memo content, real account names in home paths, a tracked `CLAUDE.local.md`, and the
  terms of a private deny list, in every tracked file; `--staged` checks a commit before it is
  made. It is the `public-tree` job in CI and part of the required merge gate.
- The deny list is never in this tree. CI reads it from the `PUBLIC_TREE_DENYLIST` secret; locally,
  point `RECALL_PUBLIC_TREE_DENYLIST` at your copy. Account names and deny-list terms are
  RATCHETED by `scripts/public_tree_baseline.json`: a file may hold no more than its recorded
  count, so moving a file out, or cleaning one, lets the count fall and it never rises unreviewed.
- An exception is an exact value with a reason in `scripts/public_tree_allowlist.txt`, never a
  pattern or a whole file.
- Host addresses, account names and key names come from the environment in scripts, never as
  literal defaults.
- The sdist is built from an allowlist in `pyproject.toml` (`tests/test_sdist_allowlist.py` pins
  it). Adding a top-level directory to it publishes that directory on PyPI.

## Docker: one database per checkout

**The rule: a session that runs the test suite starts its own container and removes it when it is
finished. No session ever points the suite at a container it did not start.**

The suite DROPs tables. When two checkouts shared one container they dropped each other's tables
mid-run, and the failures that came back described the other session's timing rather than anything
about the code under test.

| Container | Port | Who owns it | Test suite may use it |
|---|---|---|---|
| `recall-sess-<hash>` | 5400 to 5919, on 127.0.0.1 | the checkout that started it | yes, this is the one |
| `recall-db-1` | 5432 | shared, from `docker compose up` at the repo root | **no**, demos and manual work only |

- `tests/conftest.py` has **no default DSN**. With `RECALL_TEST_DSN` unset the DB tests skip and
  say so. When unset, `TEST_DSN` points at port 1, where nothing listens on any platform, so a path
  that slips past a `requires_db` mark fails loudly instead of quietly writing to somebody else's
  data.
- **`docker compose up` from a worktree strands containers.** Compose names the project after the
  directory, so each worktree gets a different container, and deleting the worktree leaves it
  running. Use `scripts/session-db.sh`, which labels the container with its checkout path, so
  `scripts/session-db.sh orphans` can find these afterwards. Its tests stub docker:
  `bash scripts/session_db_tests.sh`.
- A second `docker compose up` cannot work anyway: `docker-compose.yml` binds host port 5432, so
  only one such stack can exist at a time.

## The embedding cache is on by default

Every indexing entry point (`recall index`, `generation build`, the MCP write path, the setup
wizard, seeding) consults a content-addressed SQLite cache before embedding. It lives under the
platform cache directory (`$XDG_CACHE_HOME/recall/embeddings.sqlite`, else `~/.cache/recall/`), and
`RECALL_EMBED_CACHE` moves it or switches it off; `RECALL_EMBED_CACHE_MAX_MB` bounds it, default
512 MB, LRU past that. It is keyed on the text and the complete embedder identity, so it covers the
rebuilds that pipeline-fingerprint reuse cannot. Deleting it costs one re-embed; a corrupt or
unwritable cache degrades to no cache with a warning.

⚠️ **`Indexer(cache=...)` defaults to None on purpose.** Eval harnesses and benchmarks construct
`Indexer` directly, and a cache under a run that is measuring embedding cost would corrupt the
measurement silently. The test suite disables the cache session-wide
(`tests/conftest.py::_disable_shared_embedding_cache`) for the same reason, and because a
process-independent cache is a hidden channel between tests.

## Testing

### Red proof is required for new tests

Every new or materially changed behavior test must first be run against the pre-fix implementation
or a deliberate plausible mutation of the production code. It must fail in the intended assertion,
not through `ImportError`, collection, fixture setup, timeout, network, or an uncollected test. Then
restore the implementation and run that exact test green. Record the test node ID, the mutation or
baseline, the production symbol or line targeted, and the failure reason in the test docstring or
pull request. A test that has only been seen green is not regression evidence.

```bash
eval "$(scripts/session-db.sh up)"
python -m pytest tests/ -q -n 3          # or: make test
scripts/session-db.sh down
```

### A test that spends API credit is opt-in, never gated on a key alone

**`RECALL_RUN_PAID_TESTS=1` is the only switch that lets a test call a paid API.** A key in the
environment is not consent to spend: anyone who uses the hosted embedder or the extraction engine
has one exported. A test that calls a paid endpoint must skip unless the opt-in AND the key are
set, and say `spends <provider> credit` in its skip reason. Pair it with an offline test that
replaces only the transport, so the behaviour stays covered on every run. CI's test jobs receive no
API keys at all.

⚠️ An offline Voyage test must stub the SDK's PRESENCE as well as the transport. CI installs `dev`
without `voyageai`, and `recall.embeddings._voyage_client_class` refuses to build `VoyageEmbedder`
when `find_spec("voyageai")` is None, even though it never imports the SDK. A spec-less module in
`sys.modules` is the double it accepts; reproduce CI locally with `sys.modules["voyageai"] = None`.

### Running the suite

- Use exactly three xdist workers locally: `make test`, or `python -m pytest -n 3`. CI uses
  `-n auto` on its fixed four-vCPU runner. `make test-serial` is for reading a red run in order.
- Parallel runs are safe because `tests/conftest.py::_isolate_xdist_worker` gives every worker a
  database of its own inside the checkout's container. Read that docstring before changing the
  worker count or the DSN shape.
- **Read the skip count before calling a run green.** A few dozen skips is healthy; several hundred
  means the DB tests never ran, and the reason is printed in the skip text.
- `tests/test_entailment.py::test_qnli_judge_separates_answering_from_adjacent_text` downloads a
  model from HuggingFace, so a green run needs the network. Check the failure text before assuming
  a regression.
- Lint is `python -m ruff check .`. **Never run `ruff format`**: most files do not conform to it
  and CI only ever runs `ruff check`.
- Types are a CI gate too, and ruff does not check them: `python -m mypy` (or `make typecheck`).
- `docs/ARCHITECTURE_MAP_GENERATED.md` carries a fingerprint of the package source bytes, and CI
  refuses a stale one. Regenerate it with `python tools/architecture_map.py` from a checkout with
  LF line endings; a CRLF checkout computes a value CI never reproduces.

## Git

- Stage by pathspec, never `git add -A` or `git add .`.
- Commits must be signed and `master` refuses merge commits, so integrate by squash.
- Before assuming a regression, check whether your branch is simply behind.
