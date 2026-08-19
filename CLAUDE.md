# recall: working rules

This file is loaded into every session in this repository. It records the rules that have been
paid for in lost work, and the two commands a session opens and closes with.

## One session, one workspace

**Never work in the main checkout, and never in a worktree another session holds.**
`scripts/session-open.sh` enforces both and stops rather than warning.

```bash
scripts/session-space.sh new <short-name>   # your own worktree, off origin/master, claimed
```

Why this is a mechanism and not advice: measured 2026-08-15, this repository had **twenty
worktrees, five of them holding uncommitted work, and the main checkout itself dirty on a branch two
months behind master**. Two sessions in one checkout share an index and a working tree, so one
stages the other's files, a rebase moves HEAD under a session that is mid-edit, and `git log` looks
fine afterwards because the subjects match. The recorded cost includes a force-push that erased
three already-pushed commits.

- The **main checkout is refused outright**. It is shared by definition: every worktree resolves its
  objects through it, and it is the directory every session reaches by habit.
- Each worktree carries a **claim** naming the session that holds it, written to that worktree's own
  git directory, so it is never committed and dies with the worktree. A second session is told whose
  it is and how old the claim is, instead of discovering the collision through a lost edit.
- A claim goes stale after 12 hours **with no live process**. Liveness is asked of Windows, not of
  `kill -0`, which cannot see a Windows pid from Git Bash and reported a running `claude.exe` as
  dead. "Cannot tell" counts as alive: a false alive costs a `release --force`, a false dead costs
  somebody's in-flight work.
- If `session-space.sh` is **missing**, `session-open.sh` stops anyway. A checkout old enough to
  lack it is exactly the stale shared one, so an absent guard must not read as a passing guard.

`scripts/session-space.sh whose` prints the current claim. `release --force` takes over one you are
certain is abandoned.

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

`scripts/session-mcp.sh` generates `.mcp.json` **and records the client's approval for it**. Run it
once per checkout, since every worktree is a separate project root to the MCP client.

⛔ **Writing the file is only half of it, and the other half is silent.** A project-scoped server
stays *pending approval* until the client records it under
`projects[<dir>].enabledMcpjsonServers` in `~/.claude.json`, and a non-interactive session can
never answer that prompt. Measured 2026-08-17: **306 tracked projects on this machine, zero with
an approved server**, while `claude mcp list` reported both recall servers as `⏸ Pending approval`
with the file on disk in front of it. The symptom is a session with no `recall` tools, which is
also the symptom of a missing file, a dead corpus and a broken server, so it was misdiagnosed as a
write-ordering race for a day (`docs/preregistrations/2026-08-16-sessionstart-hook-mcp-ordering.md`).

**Diagnose with `claude mcp list` before touching anything else.** It names the state directly.
`scripts/session_mcp_approve.py` does the approval, carries only server names into the client
config (never a URL or a token), and will not reverse a server you have explicitly disabled.

- **`recall`** serves recall's own `docs/` and **`recall-memory`** serves this project's memory
  store, both out of the `recall-dogfood` corpus on port 5433 (tenants `default` and `memory`,
  1688 and 308 chunks). These are the only servers whose corpus is this project.

  They run with `RECALL_TRUST_MODE=development`, set inside the `env` block that
  `scripts/session-mcp.sh` writes rather than left to your shell: a stdio server launched with an
  explicit `env` does not inherit exported variables, so a corpus that searches fine from the
  terminal answered `INDEX_NOT_READY` through the client. Both corpora are uncalibrated and bound
  to no generation, which a strict server correctly refuses. **That setting is right for a local
  dogfood index and wrong for anything else**; lifting the refusal properly needs a calibration
  bound to an immutable generation (`recall calibration calibrate --generation G --queries FILE
  --publish`).

  The same thing from the CLI:

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

## Guards that will interrupt you, and why

Sources live in `scripts/`, deployed copies in `~/.claude/hooks/`. Each has a `*_tests.py` beside
it, and each test file has been mutation-tested: the guard was broken on purpose and the named
test watched to go red. A guard nobody has watched fail has not been tested.

⚠️ **The source and the deployed copy drift.** `session_start_hook.py` and its deployed twin
already differ by 57 lines, and nothing reports it. The newer test files assert that the deployed
copy matches the source, so a forgotten redeploy fails a test instead of silently disabling a
guard. Add that assertion to any hook you write.

- **`preregistration_guard.py`** denies a measurement command while anything under
  `docs/preregistrations/` or `benchmarks/PREREGISTRATION.md` is uncommitted. An uncommitted
  prediction has no timestamp anyone can trust. It matches at **command position** only, so
  reading, grepping or committing a message about a benchmark is untouched, and it enforces only
  "nothing is uncommitted"; it cannot verify a record exists for your specific question, and says
  so rather than implying otherwise. Escape: the bare word `ALLOW_UNREGISTERED_MEASUREMENT`.
- **`test_receipt.py`** records what each test run in this session actually reported, and prints
  it back at `git push` time: counts, age, and the command. It does not block. It exists because
  "I ran the tests" and "the tests ran" have been different things here, and because a run with
  several hundred skips is the documented false-green signature (roughly 22 skips is healthy).
  A receipt is per session id: another session's run is not your evidence.

## Git

- `git add -A` and `git add .` are blocked by a hook. Stage by pathspec.
- Commits must be signed and `master` refuses merge commits, so integrate by squash.
- Before assuming a regression, check whether your branch is simply behind: another session landing
  your commits and a real revert look identical from here.
