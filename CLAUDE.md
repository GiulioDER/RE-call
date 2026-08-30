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
- **A directory that still exists is not a checkout.** Measured 2026-08-20: 30 of 33 containers on
  this machine were compose stacks rather than session containers, and five of them belonged to a
  worktree that had been removed leaving its empty directory behind. `orphans` called that machine
  clean, because its test was `[ -d ]`. It now asks for a `.git` entry, reports a remnant as such,
  and prints `CHECK` rather than `ORPHAN` for a path too long for Windows to resolve, since a false
  `ORPHAN` line invites `docker rm -f` on a checkout somebody is using. Re-measure with
  `bash scripts/session_db_tests.sh`, which stubs docker and needs no daemon.
- A second `docker compose up` cannot work anyway: `docker-compose.yml` binds host port 5432, so
  only one such stack can exist at a time.

## Embedding: on VPS2, one at a time, bounded

**Standing instruction from the user, given 2026-08-22.** Three rules, and the reason each one is a
mechanism rather than advice.

### 1. Embedding runs on VPS2, not on this workstation

The corpus the MCP servers answer from lives on VPS2, and until today the model ran HERE and only
the rows crossed the network. That was a considered choice and it was wrong on the binding
constraint: this workstation has 12 GB of RAM with roughly 5 GB free, VPS2 has 48 GB with 35 GB
free (measured 2026-08-22, `ssh vps2 free -m`), and what actually killed a run was an onnxruntime
`bad allocation` on the 987 memo store, not CPU. CPU was the thing the old note optimised for.

```bash
cd ~/.claude/recall-vps2 && bash sync_memstores.sh     # ship the files
bash index_memory_vps2.sh                              # embed THERE, under the guards below
```

The one local corpus that stays local is the `recall-dogfood` container on port 5433, because its
database is on this machine and shipping vectors to it from VPS2 would be the same trip backwards.
It is small, and it is the exception that has to be named out loud rather than assumed.

### 2. Never two indexers against one corpus, and the check comes first

`recall.index_lock.single_writer` holds a session-scoped Postgres advisory lock keyed on
`(table, tenant)` for the whole of `Indexer.index_path`, so every caller is covered: the CLI, the
`SessionEnd` hook through `recall.setup.index_memory_directory`, and any script that builds its own
`Indexer`. A second run polls for 20 seconds and then REFUSES, naming the holder's host and pid.

It cannot go stale: a session lock dies with its backend, so a killed indexer leaves nothing behind
and there is no `--force` to get wrong. Escape hatch, for two runs you have confirmed are disjoint:
`RECALL_INDEX_ALLOW_CONCURRENT=1`, which logs a warning naming itself.

Two indexers are not merely untidy. `index_path` reads "what is already indexed" once, decides what
to skip, and prunes what is no longer on disk; interleave two of those and the second replaces rows
the first has just written and decided to skip.

### 3. Never two embedding processes on the embedding host

The advisory lock serialises indexing into one corpus. It knows nothing about an embedding run that
writes nowhere (a benchmark, a calibration), and those share the host's memory rather than the
corpus. On VPS2 that is a `flock` on `~/recall-repos/.locks/embed.lock`, taken by
`bin/index_memory.sh`, which reports who holds it rather than queueing behind them.

### 4. What bounds the memory, since VPS2 also runs live trading services

| Bound | Where | Why |
|---|---|---|
| `RECALL_INDEX_BATCH_CHUNKS=64` | VPS2's `~/recall-repos/.env` | fastembed pads a batch to its longest member, so 256 long chunks ask onnxruntime for 4.3 GB. 64 survived the longest documents in this corpus. |
| `MemoryMax=8G`, `MemorySwapMax=0` | the `systemd-run --scope` in `bin/index_memory.sh` | a cgroup limit is enforced by the KERNEL against the whole process tree, so the job is killed rather than the host |
| `CPUQuota=250%`, `nice -n 15` | same scope | thread caps do not work: `OMP_NUM_THREADS=3` still measured 515% CPU and `taskset -c 9-11` still measured 479%, because onnxruntime sizes its own pool and re-pins its own threads |

⛔ `RECALL_FASTEMBED_BATCH` is named as the fix in older notes and is read NOWHERE in this package.
Exporting it is a no-op. `RECALL_INDEX_BATCH_CHUNKS` is the knob that exists, and `recall index
--batch-chunks` is its per-run form.

### 5. Which model, and when the bounds above bind anything

The memory corpus is built with **`voyage:voyage-4`**, a hosted model, since 2026-08-22. Not for
speed: the corpus and the server that answers from it must agree, and they did not. The tenant held
`bge-small` and `bge-large` rows while the client queried it with `voyage-4`, and **nothing raised,
because all three emit 1024 dimensions** — pgvector computes a cosine happily and returns a
confidently ranked list that means nothing. It converged on voyage-4 rather than on bge-large
because the sibling tenants already use it, and because the alternative puts a 1.4 GB ONNX model
inside every stdio server session on that host, resident, with a cold start on the first query.

**So the bounds in the table above bind nothing on a hosted run**, where the batch is a request size
rather than an allocation. They are kept because they are what stands between a local-model run and
the host, and that is not hypothetical there: VPS2's main Postgres cluster has been down since
2026-08-19, `Result: oom-kill`. A dimension match is not a model match, and free memory today is not
a memory bound.

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

- **`recall-memory`, `recall-code` and `recall-docs`** serve this project's own corpora: its memory
  store, its generated-code index and its `docs/`. They are the only servers whose corpus is this
  repository.

  🔁 **Rewritten 2026-08-30, and every part of what stood here was wrong.** It said both servers ran
  out of a local `recall-dogfood` corpus on port 5433 with `RECALL_TRUST_MODE=development`, and
  explained at length why that relaxed setting was correct. Measured that day, three faults, each
  already documented as a fault elsewhere in this file or in the global notes, and none of them
  checked by anything:

  | Fault | Symptom | Where it was already written down |
  |---|---|---|
  | the 5433 container was removed on 2026-08-25 | `ConnectionRefused` | "Nothing should be pointed at 5433" |
  | an MCP `env` block REPLACES the environment | `ModuleNotFoundError: anyio`, before the DB is even reached | nowhere; this one was new |
  | `RECALL_TRUST_MODE=development` on a certified corpus | a trusted answer reported `degraded`, `calibrated` forced false | "actively wrong ... because a relaxed gate never errors, nothing reports it" |

  **The cost was a full working session with no memory layer at all**, during which the same
  decisions were re-derived and the same context re-explained, because a session cannot tell a dead
  server from a corpus with nothing to say. Both are silence.

  The servers now run **where the corpora are, over ssh stdio, under strict trust**. Strict is
  expressed by the ABSENCE of `RECALL_TRUST_MODE`: setting it to any string is precisely how a
  certified corpus ends up served relaxed while the config claims otherwise. Host, interpreter and
  paths come from `recall_corpus` in `~/.claude/recall-mcp-secrets.json`, never from the tree.

  ⚠️ **Each tenant must be served with the embedder its ACTIVE generation was built with.** The
  three here use three DIFFERENT 1024-dimension models, and a mismatch does not error: pgvector
  computes a cosine over whatever produced the vectors and returns a confidently ranked list that
  means nothing. Read them from the corpus rather than from this table, which will rot:

  ```bash
  psql "$RECALL_DSN" -c "select tenant_id, pipeline_identity->>'embedder' from recall_generations where state='active'"
  ```

  ⛔ **`.mcp.json present (N servers)` is not a health check**, and it reads exactly like one: it
  printed a truthful, reassuring count at every session start for days while every server in the
  file was dead. `scripts/session_memory_check.py` actually talks to the server and separates DEAD
  (nothing answered) from DEGRADED (answering un-gated) from QUIET (trusted, and genuinely had
  nothing). `scripts/session-open.sh` runs it; `RECALL_SKIP_MEMORY_CHECK=1` skips it when offline.
  Verify by hand in about ten seconds:

  ```bash
  python scripts/session_memory_check.py --all --quiet
  ```
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
python -m pytest tests/ -q -n 4          # or: make test
scripts/session-db.sh down
```

- 🔁 **Corrected 2026-08-23: run it in PARALLEL. Serial is 50 minutes; `-n 4` is 14.**
  Measured that day on this workstation, same commit (`ec6ab9a0`), same container, back to back,
  at 6,563 tests: **serial 49:58** (52:27 of wall clock, collection and interpreter start
  included), **`-n 4` 14:05** (14:20 wall), and `-n 6` **twice, at 16:45 and 21:08, with a worker
  killed both times**. So four workers is **3.7× faster** on wall clock, and six is not reliably
  faster than four while being reliably less stable on a 12 GB machine: that is why `make test`
  defaults to four rather than to `auto`, which would ask for twelve.

  ⚠️ **Those three numbers are one run each, on a box that is not idle.** `\Processor(_Total)\%
  Processor Time` read 47% with nothing of mine running, three other session containers were up,
  and the two `-n 6` runs differ by 4.5 minutes from each other. Treat anything under about 1.3×
  as noise. Re-measure with `time` around the whole invocation rather than reading pytest's line,
  which excludes interpreter start.

  **`pytest-xdist` is now in the `dev` extra, and `tests/conftest.py::_isolate_xdist_worker` is
  what makes it safe.** Every worker gets a DATABASE of its own inside this checkout's container,
  because the workers are separate processes sharing one database otherwise, and three things
  collide there silently: the shared `chunks` table, the migration ledger, and the cluster-wide
  `recall_rls_probe` role. Read that docstring before changing the worker count or the DSN shape;
  the first parallel run failed six tests purely because the rewritten DSN was in libpq keyword
  form and five tests take `TEST_DSN` apart as a URL.

  ⚠️ **Parallel is for a green run, serial is for reading a red one.** `-n` interleaves the output
  of four workers, and a failure's traceback arrives detached from the progress line above it.
  `make test-serial` is the ordered form; a single failing file is faster to re-run on its own
  either way.

- **Moving `voyageai` out of module scope buys 44 seconds on ONE file and about 7 on the suite.**
  `tests/test_embeddings_retry_after.py` imported it at module scope, and it imports
  `langchain_text_splitters`, which imports `transformers` and `torch`. Three modules did this;
  they now take the session-scoped `voyageai_sdk` fixture in `tests/conftest.py`.

  Measured back to back against `99e52d13`, warm, same 6,599 tests:

  | | master | branch |
  |---|---|---|
  | that one file, alone | **45.33s** | **1.04s** |
  | whole-suite collection | 55.6 / 59.6 / 59.1s | 50.9 / 54.8 / 50.0 / 51.3s |

  So it is worth having when you are ITERATING on that file, and close to noise on a full run,
  because by the time pytest reaches a file named `test_e…` another module has already pulled
  `transformers` in and voyageai only adds its margin.

  🔁 **Corrected the same day it was written. This first said "collection was 154s and is now
  75.1s", a 79 second saving, and that pair was invalid**: 154.10s was the first collect of the
  day on a cold page cache, and 75.14s was hours later with `torch` resident. A cold master
  against a warm branch measures the cache, not the change. It also said the cost was paid by
  "every `pytest` invocation ... including `pytest tests/test_cli.py`", which is simply false:
  pytest imports only the modules it collects. **Take both halves of a before/after back to back,
  in both orders, or do not report a pair.** Full accounting:
  `docs/preregistrations/2026-08-23-test-suite-wall-clock.md`.

  ```bash
  python -m pytest tests/ -q --collect-only | tail -1
  ```

- ⚠️ **Windows Smart App Control can block `pyarrow` for a few hours and then stop, with nothing
  changed.** `ImportError: DLL load failed while importing lib: An Application Control policy has
  blocked this file`, which fails
  `tests/test_bench_beam_candidate_k.py::test_the_shipped_local_reranker_is_reachable_without_a_cloud_call`
  and skips three `sentence-transformers` tests whose own guards catch `ImportError`, so the whole
  signature is `1 failed, 37 skipped` where 34 skips is healthy. `pytest.importorskip` deliberately
  does NOT skip the first one, because a module that is installed and broken is not a module that
  is absent, and that is the right behaviour.

  🔁 **Corrected the same afternoon: it is a TRANSIENT, and this first recorded it as a standing
  fact.** It failed serially, alone, at 15:20 having passed at 12:41, and passed again unchanged
  by 16:23 on the same machine. Re-measured at 16:40: `pyarrow 25.0.1` imports and all 14 of those
  tests pass. So do not route around it, pin it, or downgrade anything, and do not disable Smart
  App Control, which is a one-way door. Re-measure, one second:

  ```bash
  python -c "import pyarrow, pyarrow.parquet; print(pyarrow.__version__)"
  ```

- 🔁 **Corrected 2026-08-20: the full suite takes 30 to 40 minutes with a database, not 12.**
  Three runs on this machine that day, on an otherwise idle box: **37:13, 39:20 and 29:45**, at
  6,088 tests. The old line said "about 12 minutes" with no date and no way to re-check, which is
  why it was still believed after tripling. It was almost certainly true when written.

  **There is no hotspot to fix, and that is the useful part.** The thirty slowest tests account for
  roughly **511s of 1785s, about 29%**, and the slowest single one is 72s of setup. The rest is 6,000
  tests at an average of **0.29s each**. So the suite is not slow, it is large; anyone hunting for
  the one bad test will not find it, and the lever that exists is parallelism (`pytest -n`), not
  surgery.

  Note the spread: the same suite on the same machine varied by **10 minutes** across three runs.
  Budget for the top of that range rather than the middle.

  ⚠️ **The old line's other half, "about 5 minutes without a database", is NOT re-measured here.**
  Every run above had one. It is dropped rather than carried forward, because a figure that has been
  wrong by 3× in its measured half has earned no trust in its unmeasured one. Treat the no-database
  runtime as unknown until somebody runs it, rather than as five minutes.

  Re-measure, and get the breakdown rather than just the number:

  ```bash
  python -m pytest tests/ -q --durations=30
  ```

- **Read the skip count before calling a run green.** Roughly 22 skips is healthy; **34 is the
  current figure** (measured 2026-08-20, all three runs). Several hundred means the DB tests never
  ran, and the reason is printed in the skip text.
- ⚠️ **A green run needs the network, and one test says so only by failing.**
  `tests/test_entailment.py::test_qnli_judge_separates_answering_from_adjacent_text` downloads
  `cross-encoder/qnli-distilroberta-base` from HuggingFace. Measured 2026-08-20: it passed in two
  runs and failed in a third with `[Errno 11001] getaddrinfo failed`, after retrying five times.
  That is the whole difference between `6087 passed, 1 failed` and `6088 passed` in the same hour
  on the same commit. **Check the failure text before assuming a regression**: a network failure
  here looks exactly like a broken judge.
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
- **`git_clone_race_guard.py`** denies a commit onto the default branch, and a push of a branch
  that is not the one checked out, because a second session sharing a clone can move the working
  tree between your `checkout -b` and your `commit`. **A tag is exempt**: it names an object
  directly and has no working tree to disagree with, and refusing one broke every signed release
  until 2026-08-22. A name that is both a tag and a branch stays guarded, since git itself refuses
  that push as ambiguous. Escape: a trailing `# RACE_GUARD_OK`.

  ⚠️ It runs **before** the command, so creating and pushing a tag in one compound command is
  still refused: at that moment the tag does not exist and the guard cannot tell it from a branch.
  Tag in one command, push in the next.

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
