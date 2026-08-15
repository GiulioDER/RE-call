# recall: working rules

This file is loaded into every session in this repository. It records the rules that have been
paid for in lost work, and the two commands a session opens and closes with.

## Open a session

```bash
scripts/session-open.sh
```

It is read-only apart from writing `.mcp.json`, and it prints what it found. Do the same three
things by hand if you prefer:

1. **Know which branch you are on.** A worktree's directory name is not its branch, and the local
   `master` ref here is routinely stale. Ask git, and diff against `origin/master`, never `master`.
2. **Generate this checkout's MCP config** with `scripts/session-mcp.sh`, once per checkout.
3. **Start a database only if you are going to run the DB-backed suite**, with
   `eval "$(scripts/session-db.sh up)"`.

## Close a session

```bash
scripts/session-close.sh
```

It removes this checkout's own database container, reports anything left uncommitted, and lists
orphaned containers from checkouts that no longer exist. It never touches another session's
container.

## Docker: one database per checkout

**The rule: a session that runs the test suite starts its own container and removes it when it is
finished. No session ever points the suite at a container it did not start.**

The suite DROPs tables. When two checkouts shared one container they dropped each other's tables
mid-run, and the failures that came back described the other session's timing rather than anything
about the code under test. Hours went into debugging tests that were never broken.

| Container | Port | Who owns it | Test suite may use it |
|---|---|---|---|
| `recall-sess-<hash>` | 5400 to 5919, on 127.0.0.1 | the checkout that started it | yes, this is the one |
| `recall-dogfood` | 5433 | shared, long-lived corpus for the `recall` MCP server | **no** |
| `recall-db-1` | 5432 | shared, from `docker compose up` at the repo root | **no**, demos and manual work only |

Consequences that follow from the rule:

- `tests/conftest.py` has **no default DSN**. With `RECALL_TEST_DSN` unset the DB tests skip and
  say so. They do not fall back to port 5432, because that fallback is what caused the collisions.
  When unset, `TEST_DSN` points at port 1, where nothing listens on any platform, so a path that
  slips past a `requires_db` mark fails loudly instead of quietly writing to somebody else's data.
- **`docker compose up` from a worktree is the thing that strands containers.** Compose names the
  project after the directory, so each worktree gets a *different* container, and deleting the
  worktree leaves it running with nothing left that knows to remove it. Use
  `scripts/session-db.sh`, which labels the container with its checkout path, so
  `scripts/session-db.sh orphans` can find these afterwards. It currently finds real ones.
- A second `docker compose up` cannot work anyway: `docker-compose.yml` binds host port 5432, so
  only one such stack can exist at a time.

## MCP servers

`scripts/session-mcp.sh` generates `.mcp.json`. Run it once per checkout, since every worktree is
a separate project root to the MCP client.

- **`recall`** serves recall's own `docs/` and **`recall-memory`** serves this project's memory
  store, both out of the `recall-dogfood` corpus on port 5433 (tenants `default` and `memory`,
  1688 and 308 chunks). These are the only servers whose corpus is this project.

  ⚠️ **Their `recall_search` currently refuses**, and this is by design rather than a
  misconfiguration. `recall_mcp/server.py` never passes a `TrustPolicy`, so the service defaults to
  strict and **ignores `RECALL_TRUST_MODE`**: the docstring is explicit that a server degrading by
  omission would degrade in production. An uncalibrated corpus therefore returns
  `INDEX_NOT_READY`. Lifting it needs a real calibration bound to an immutable generation
  (`recall calibration calibrate --generation G --queries FILE --publish`), not a flag.

  Until then the CLI is the working path, because the CLI *does* honour the env var:

  ```bash
  RECALL_DSN=postgresql://recall:recall@127.0.0.1:5433/recall RECALL_EMBEDDER=fastembed \
    RECALL_TRUST_MODE=development python -m recall.cli --tenant memory search "your question"
  ```

  Drop `--tenant memory` to search `docs/` instead. Rebuild either corpus with the recipe in
  `scripts/session-mcp.sh`. **Index each tenant separately:** re-indexing prunes sources that have
  vanished from disk, so pointing both corpora at one tenant deletes the other.
- **The remaining `http` servers are internal infrastructure for a different project.** They are
  reachable and useful for that system, but be clear-eyed here: their code and docs search does
  **not** search recall, and their read-only SQL tool does **not** reach recall's tables. Do not use
  them to answer questions about this repository. Their names, hosts and purpose are described in
  `~/.claude/recall-mcp-secrets.json`, outside this tree.

`.mcp.json` is **gitignored and must stay that way.** This repository is public on PyPI, GitHub and
the MCP registry, and **both halves of an internal server are disclosure**: the bearer token
obviously, and the host address too, because an inventory of which machines exist and what runs on
them is worth something on its own. Neither lives in the tree.
`scripts/session-mcp.sh` refuses to write the file at all unless the ignore rule is already in
place, rather than writing it and warning afterwards.

## Testing

```bash
eval "$(scripts/session-db.sh up)"
python -m pytest tests/ -q
scripts/session-db.sh down
```

- The full suite takes about **12 minutes** with a database and about 5 without.
- **Read the skip count before calling a run green.** Roughly 22 skips is healthy. Several hundred
  means the DB tests never ran, and the reason is printed in the skip text.
- Lint is `python -m ruff check .`. Bare `ruff` on this machine is an old 0.6.9; `python -m ruff`
  is the pinned 0.16.x. **Never run `ruff format`**: 348 of 406 files fail it and CI only ever runs
  `ruff check`.

## Git

- `git add -A` and `git add .` are blocked by a hook. Stage by pathspec.
- Commits must be signed and `master` refuses merge commits, so integrate by squash.
- Before assuming a regression, check whether your branch is simply behind: another session landing
  your commits and a real revert look identical from here.
