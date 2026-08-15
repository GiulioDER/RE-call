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
| `recall-sess-<hash>` | 5400 to 5520, on 127.0.0.1 | the checkout that started it | yes, this is the one |
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

- **`recall`** serves recall's own `docs/` out of the `recall-dogfood` corpus. This is the only
  server whose corpus is this project. Reach for it first when the question is what recall already
  does or has already decided. Rebuild the corpus with the recipe in `scripts/session-mcp.sh`.
- **`code-rag`, `qwen-mcp`, `qwen-vps3`, `vps3-lite`, `mcp-pg-ops`** index `/opt/sentiment_agent`
  and query the `sentiment_agent` database. They are reachable and useful for that host, but be
  clear-eyed here: `code_search` on those servers does **not** search recall, and `db_query_ro`
  does **not** reach recall's tables. Do not use them to answer questions about this repository.

`.mcp.json` is **gitignored and must stay that way.** It carries bearer tokens, and this
repository is public on PyPI, GitHub and the MCP registry. The tokens live in
`~/.claude/recall-mcp-secrets.json`, outside the tree. `scripts/session-mcp.sh` refuses to finish
if the ignore rule has gone missing.

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
