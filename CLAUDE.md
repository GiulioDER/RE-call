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

It is read-only apart from writing `.mcp.json`, and it prints what it found. Do the same four
things by hand if you prefer:

1. **Know which branch you are on.** A worktree's directory name is not its branch, and the local
   `master` ref here is routinely stale. Ask git, and diff against `origin/master`, never `master`.
2. **Generate this checkout's MCP config** with `scripts/session-mcp.sh`, once per checkout.
3. **Ask the corpus what state it is in** with `scripts/session-corpus.sh status`.
4. **Start a database only if you are going to run the DB-backed suite**, with
   `eval "$(scripts/session-db.sh up)"`.

⚠️ **Step 3 is new as of 2026-08-25 and it replaces a line that was worse than nothing.** The
report used to say `.mcp.json present (2 servers)`, which is a statement about a FILE and was read
as a statement about a CORPUS. It stayed green while both servers pointed at a database that had
been down for days. A cheap check standing in for an expensive one is not a weaker check, it is a
misleading one, because nobody re-checks a green line.

## Close a session

```bash
scripts/session-close.sh
```

It removes this checkout's own database container, closes the MCP transports this session opened,
reports anything left uncommitted, and lists orphaned containers from checkouts that no longer
exist. It never touches another session's container, and never another agent's MCP server: both
are decided by positive identity, a label in one case and a parent chain in the other.

It also reports how far the VPS2 serving checkout is behind master. Two sections below say what
each of those is for: the serving sync, and the MCP close.

## The serving checkout follows master, and a session is what makes it

```bash
scripts/session-serving.sh              # where the VPS2 serving checkout is, and what it lacks
scripts/session-serving.sh sync         # fast-forward it to origin/master, verify, undo on failure
```

The MCP servers do not answer from the checkout you are editing. They answer from
`~/recall-repos/serving` on VPS2, and the venv there holds an **editable** install pointing at that
clone. Measured 2026-08-26:

```text
.venv/lib/python3.12/site-packages/_editable_impl_recall_rag.pth -> ~/recall-repos/serving
python -c "import recall; print(recall.__file__)" -> ~/recall-repos/serving-master/recall/__init__.py
```

So a fast-forward there changes the code every future server session runs, with no reinstall and
no restart, and until somebody does it a merged fix changes nothing anyone can search. **A stale
server is a working server**: it does not error, it answers from older code, and no session
reports the gap. `scripts/session-close.sh` prints the distance, `--sync-serving` closes it, and
step 5 of `/session-close` is where it happens by default.

Three refusals, each of which is a fact rather than caution:

| Condition | Result | Why |
|---|---|---|
| the update touches `recall/migrations/` | refused unless `--with-migrations` | code that knows migration N against a database at N-1 raises `SchemaTooOld` at startup, and a client renders that as a server with NO tools, which is also the symptom of a missing file, an unapproved server and an unreachable host |
| the serving tree is dirty, diverged, or detached | refused | uncommitted changes there mean somebody is hot-patching a live server, and local commits mean a diverged deployment rather than a stale one |
| `embed.lock` is held | refused | swapping modules under a live indexer breaks a run in a way that surfaces hours later as a partial corpus |

⚠️ **Verification is the point, and a green `git merge` is not it.** After the move: `recall schema
status` must report `compatible: yes`, `recall_mcp.server` must import, and a real JSON-RPC
handshake against the server launched exactly as `.mcp.json` launches it must return a non-empty
tool list. Measured 2026-08-26 against the live host, with nothing to move: **18 tools, 58.6s for
the whole `sync`**, of which the handshake is 21.0s. Any failure resets the checkout to where it
was and says so, because the one state nobody may leave behind is a serving checkout that no
longer serves.

Re-measure, read-only, one ssh:

```bash
scripts/session-serving.sh status
```

Tests: `bash scripts/session_serving_tests.sh`. Git is real and python is stubbed, so it needs no
database, no network and no VPS2; mutation-tested per the guard rule above. CI runs it on Linux
rather than beside the Windows session hooks, because the test that proves a live indexer's
`embed.lock` stops a sync skips itself where `flock` does not exist.

## Close the MCP servers this session opened

```bash
scripts/session-mcp-close.sh              # report: this session's transports, and the fleet
scripts/session-mcp-close.sh close        # close them, then count the fleet again
```

⚠️ **An idle MCP server is not free, and nothing reports the ones that leak.** Measured on VPS2 on
2026-08-26: **18 live servers holding 14.67 GB**, plus their 18 ssh transport wrappers at 0.39 GB,
on a 47 GB host that also runs the live trading services. Roughly 815 MB each.

🔁 **Corrected 2026-08-26, hours after it was written: the COUNT was about double, the memory
was right.** This said "89 live `recall_mcp.server` processes, 21.5 GB resident, oldest 69 hours".
Both a server and its ssh transport wrapper carry the string `python -m recall_mcp.server`, one as
the command it runs and one inside `--cmd=...`, so a matching count reports every server twice.
Measured the same day with the two separated: **18 servers holding 14.67 GB, and 18 wrappers
holding 0.39 GB.** The wrappers are ~22 MB each, which is why the memory total barely moved and the
count halved. `scripts/session-mcp-close.sh report` no longer counts wrappers.


Each server lives exactly as long as its stdio transport, which is the whole mechanism: `.mcp.json`
launches it as `ssh <host> '... exec python -m recall_mcp.server'`, and killing the LOCAL ssh
killed the remote server in **under 3 seconds** (measured the same day with a marked probe,
confirmed by `ps -p`). ssh sets no keepalive, so a client that vanishes leaves its servers running
until somebody looks, and a leaked server is indistinguishable from a working one.

⛔ **Ownership is the parent chain, never the command line.** Three live transports with the
IDENTICAL command line were parented to `codex.exe` rather than Claude the day this was written,
so `pkill -f recall_mcp.server`, here or on the host, would have killed another agent's servers
mid-query. Without `CLAUDE_PID` there is no positive identity and the script reports rather than
guessing. Servers that are not this session's are counted and left alone: age does not prove
abandonment, exactly as with somebody else's container.

`scripts/session-close.sh` closes this session's transports by default; `--keep-mcp` leaves them
open. The fleet is counted before and after, because a kill returning 0 says a signal was
delivered, not that memory was freed on another machine.

🔑 **The SessionEnd hook does the same close automatically, and that is the half that actually
stops the leak**, because a session that leaks is by definition one where nobody ran a checklist.
The script is the visible, verifying form; the hook is the one that catches the rest. Both decide
ownership the same way, and the hook's identity is measured rather than assumed: this worktree's
claim file, written at session start by the client-spawned hook, records `pid=9764`, and that pid
is a `claude.exe` whose own parent is the app root. So the client process is per SESSION, and a
parent chain reaching it separates two sessions of the same app, which a chain reaching the app
root would not.

### Sweeping what has ALREADY leaked

```bash
scripts/session-mcp-close.sh sweep              # what on the host has no client left
scripts/session-mcp-close.sh sweep --kill       # close those, by positive identity only
```

Closing this session's transports stops a session leaking; it cannot touch what leaked before,
because a server whose client vanished still has a live ssh wrapper on the host and is
indistinguishable there from one somebody is querying right now.

🔑 **`scripts/session-mcp.sh` now stamps `RECALL_MCP_CLIENT=<host>-<checkout id>` into every server
command it generates**, which is what makes the question answerable: the wrapper on the host and
the `ssh` transport here are two ends of one command line, so a mark with no live local transport
has no client, and nothing can be querying it. The id is `session-db.sh id`, asked for rather than
re-derived. The variable is inert to the server.

Four things it will not do, measured against the real fleet the day it was written:

| Bucket | What happens | Why |
|---|---|---|
| marked ours, no live transport | closed | positive identity: nothing can be querying it |
| marked ours, transport live | left | somebody is using it |
| marked, another machine | left | this workstation cannot speak for that one |
| unmarked, another agent's config | left | **16 of the 18** live servers were launched by `codex.exe` on this same machine with a nearly identical command line |
| unmarked, ours (pre-marker) | `--unmarked`, and only while zero unmarked transports of our shape are live here | until that is true, one of them may still be held by a client running here |

Tests: `python scripts/session_mcp_sweep_tests.py` (16, both process tables are fixtures and the
killer is a log), mutation-tested six ways, and `bash scripts/session_mcp_close_tests.sh`. The
process table is a fixture and the killer is a log, so neither needs Claude, ssh or a host. Mutation-tested five ways, and **one of the five
survived the first version of the tests**: the fixture excluded the script's own ssh for the wrong
reason, so the self-exclusion guard could be deleted with everything still green. The repair and
the reasoning are in the test header, and it is the clearest example in this repository of why a
guard nobody has watched fail has not been tested.

## Docker: one database per checkout

**The rule: a session that runs the test suite starts its own container and removes it when it is
finished. No session ever points the suite at a container it did not start.**

The suite DROPs tables. When two checkouts shared one container they dropped each other's tables
mid-run, and the failures that came back described the other session's timing rather than anything
about the code under test. Hours went into debugging tests that were never broken.

| Container | Port | Who owns it | Test suite may use it |
|---|---|---|---|
| `recall-sess-<hash>` | 5400 to 5919, on 127.0.0.1 | the checkout that started it | yes, this is the one |
| `recall-db-1` | 5432 | shared, from `docker compose up` at the repo root | **no**, demos and manual work only |

🔁 **Corrected 2026-08-25: `recall-dogfood` on 5433 is gone from this table because it is gone.**
It was listed here as the "shared, long-lived corpus for the `recall` MCP server", and on
2026-08-25 it was not running, had not been for some time, and was not where this project's
corpora live. They are on VPS2. Nothing local should be pointed at 5433; see **MCP servers**
below for what replaced it.

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

🔁 **Corrected 2026-08-25: there is no longer a named local exception.** This paragraph used to
carve one out for the `recall-dogfood` container on port 5433, on the reasoning that its database
was on this machine so embedding it on VPS2 would ship the vectors straight back. The reasoning was
sound and the container is gone; every corpus this project serves now lives on VPS2, so the rule
above has no exception to state.

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

### 6. The embedding cache is ON by default, and it is the reason a rebuild is cheap

Every indexing entry point (`recall index`, `generation build`, the MCP write path, the setup
wizard, seeding) consults a content-addressed SQLite cache before embedding. It lives under the
platform cache directory (`$XDG_CACHE_HOME/recall/embeddings.sqlite`, else `~/.cache/recall/`), and
`RECALL_EMBED_CACHE` moves it or switches it off; `RECALL_EMBED_CACHE_MAX_MB` bounds it, default
512 MB, LRU past that.

🔑 **It covers the case `_reuse_source` cannot.** Generation build already carries a source's chunks
forward when nothing changed, but that reuse is keyed on the PIPELINE FINGERPRINT: bump a
derivation rule, a chunker version or a context mode and every source loses reuse and is
re-embedded, including the ones whose chunk text came out byte-identical. The cache is keyed on the
text and the complete embedder identity, so it also covers a first build after `generation gc` has
pruned the generations reuse would have read, and a `--force` rebuild of an unchanged corpus, which
this file describes as spending "a full Voyage pass".

Deleting the file costs one re-embed and nothing else. A corrupt or unwritable cache degrades to no
cache with a warning rather than failing the run, so it cannot be the reason a build dies.

⚠️ **`Indexer(cache=...)` still defaults to None on purpose, and eval harnesses and benchmarks
construct `Indexer` directly.** A cache appearing under a run that is MEASURING embedding cost
would corrupt the measurement silently, so the opt-in lives at the entry points a person invokes,
not in the library. The test suite disables the cache session-wide
(`tests/conftest.py::_disable_shared_embedding_cache`) for the same reason plus one more: it is
process-independent by design, which makes it a hidden channel between tests.

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

- **`recall-memory`** and **`recall-code`** serve this project's own corpora. They are the only
  servers whose corpus is this repository, and they run **on VPS2 over ssh stdio**, because the
  corpus postgres there listens on 127.0.0.1:55432 only and the Voyage API key lives in that
  host's `.env`. `.mcp.json` therefore carries no secret: it sources that `.env` on the far side.

  🔁 **Corrected 2026-08-25, and the correction is the useful part.** This entry used to describe
  two servers against a `recall-dogfood` container on port 5433, running with
  `RECALL_TRUST_MODE=development` because *"both corpora are uncalibrated and bound to no
  generation, which a strict server correctly refuses"*. That sentence was true when it was
  written and every clause of it had since stopped being true: the container was not running, the
  corpora had moved to VPS2, and they are **certified**. Measured that day against the live
  database, all three tenants resolve `certified`, and driving the server end to end returned
  `trust_state=trusted, calibrated=true` with the strict default in force.

  🔑 **The workaround outlived the problem, and that is the general hazard.** A relaxed trust mode
  is invisible when it is unnecessary: it does not error, it just stamps `degraded` on answers
  that had earned `trusted`. Nothing reports a gate that is open wider than it needs to be. When
  you write a mode that says "this is fine for now", write down the condition that retires it.

  | tenant | embedder | threshold | separability | state |
  |---|---|---:|---:|---|
  | `memory` | `voyage:voyage-4` | 0.509 | 0.974 (50/28) | certified, promoted 2026-08-24 |
  | `re-call-code-gen` | `voyage:voyage-code-3` | 0.662 | 0.988 (22/26) | certified |
  | `re-call-docs` | `BAAI/bge-large-en-v1.5` | 0.637 | 0.976 (40/40) | certified, **off by default** |

  ⛔ **All three are 1024 dimensions and all three are different models.** pgvector computes a
  cosine between any two 1024-vectors without complaining, so the wrong embedder does not raise,
  it returns a confidently ranked list that means nothing. `scripts/session-mcp.sh` passes the
  embedder **per tenant** for this reason and must keep doing so.

  `re-call-docs` is off by default because bge-large is a local ONNX model that goes resident in
  every stdio session on the host that also runs the live trading services; the other two are
  hosted APIs and load no model. Turn it on with `RECALL_MCP_INCLUDE_DOCS=1 scripts/session-mcp.sh`.

  Re-measure the whole picture in about four seconds, and do this rather than trusting the table:

  ```bash
  scripts/session-corpus.sh status
  ```

  ⛔ **The serving checkout on VPS2 must sit at the DATABASE's migration level, and the default is
  a moving target.** Measured 2026-08-25: the database is at migration **0016**, while
  `~/recall-repos/engine` is recall **0.9.6** and knows only up to 0014, so the server refuses to
  start against it:

  ```text
  SchemaTooNew: table 'chunks' has unknown schema migration(s) ['0015', '0016']; upgrade RE-call
  ```

  That refusal is correct and it is loud on the server's stderr, but an MCP client shows it as a
  server that simply has no tools, which is also the symptom of a missing file, an unapproved
  server and an unreachable host.

  🔑 **The fix is a symlink on VPS2, not a path in this repository.** `~/recall-repos/serving`
  points at whichever checkout matches the database (`graph-annotations-6d3aeb28`, 0.10.0,
  migration 0016, as of 2026-08-25), and `scripts/session-mcp.sh` names the symlink. A branch
  worktree is a moving target that vanishes when the branch is done, so pinning one here would put
  the repair in a repository that cannot see the breakage. **Whoever migrates the corpus repoints
  `serving` on the host that migrated it, in the same operation**, and every checkout follows
  without being edited. `RECALL_VPS2_CHECKOUT` overrides it for a one-off.

  ```bash
  ssh vps2 'cd ~/recall-repos && ln -sfn <checkout> serving && ls -ld serving'
  ```

  🔑 **Repointing the symlink and keeping it CURRENT are different jobs.** The symlink is moved by
  whoever migrates the corpus, by hand, on the host that migrated it. Keeping whatever it points
  at at `origin/master` is `scripts/session-serving.sh sync`, which runs every session close. As
  of 2026-08-26 it points at `serving-master`, a clone of this repository tracking master, so the
  ordinary case is a fast-forward and no symlink work at all.

  ⚠️ Use `ln -sfn`, not `ln -sf`. Without `-n`, if `serving` is an existing symlink to a directory
  the link is created *inside* the target rather than replacing it, and the result resolves to
  nothing while looking like it worked.

  ⚠️ **Migration numbers have already collided across branches here.** `semantic_graph_foundation`
  is `0016` on master and `0015` on the `engine-heading-contextualization-033a7fbc` checkout, where
  master's `0015` is `learned_sparse_chunk_index`. The ledger records the filename next to the
  version, so check the filename and not just the number before concluding two hosts agree.
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

- ⚠️ **A local run past 40 minutes means the RUN was serial or the BOX was loaded, not that the
  suite regressed, and `make test` now checks which before launching.** Three facts, measured
  2026-08-26, that close the recurring "the suite is slow, probably fastembed" diagnosis:

  1. **CI runs the whole suite in 4:19** (6,623 passed, 151 skipped, `-n auto` with coverage on a
     4-vCPU runner, run 32966542600). The suite is not intrinsically slow. Re-measure:
     `gh run list --workflow ci.yml --branch master --limit 3` and read the `test` job.
  2. **Every real-model test together — all `requires_fastembed`, the QNLI entailment judge, the
     sentence-transformers reranker — is 73 tests and 80.35s serial, slowest single test 5.26s.**
     They run only locally (CI installs without those extras, hence its 151 skips against a
     healthy local 34). Fastembed is ~1.5 minutes of a ~50 minute serial suite. Re-measure:

     ```bash
     python -m pytest tests/test_their_harness_backend.py tests/test_membench_adapters.py \
       tests/test_fallback_profile_id_distinctness.py tests/test_entailment.py \
       tests/test_bench_beam_candidate_k.py -q --durations=10
     ```
  3. **The fastembed/onnxruntime processes that DO starve this machine are other sessions'
     local `recall.cli index` and benchmark runs**, which the embedding-on-VPS2 rule above
     already covers. One was live during this diagnosis (585 MB RSS, 834 CPU-seconds), beside a
     concurrent pytest from a second session, with 1.4 GB of 12 GB available. On a box in that
     state four workers are not slower, they are OOM-killed, and the retries are the 40+ minutes.

  `scripts/suite-preflight.sh` is the mechanism: `make test` asks it for a worker count sized to
  the memory actually available (≥6 GB → 4, ≥3 GB → 2, else serial), it warns about competing
  pytest and indexing processes by command line, and `N=<n>` still overrides it verbatim.
  Tests: `bash scripts/suite_preflight_tests.sh`, mutation-tested per the guard rule.

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
- ⚠️ **Types are a CI gate too, and ruff does not check them.** `python -m mypy` (or `make
  typecheck`) over the whole tree, which is what the `typecheck` job runs. Measured 2026-08-29:
  349 source files, about 60 seconds cold. This is written down because a session ran nine
  auditors, three verification agents, an adversarial panel, two differential reviews and two
  architect gates over a 54-fix change, and still shipped three type errors to CI, for the
  single reason that nothing in this file told it mypy existed. Ruff being green says nothing
  about mypy: the errors were a widened `tuple[float, ...]` and two `getattr` optionals that
  narrow at runtime and not for the checker.

  ```bash
  python -m mypy
  ```

  ⚠️ **When that command dies with `An Application Control policy has blocked this file`, the gate
  is not unrunnable and you must not report it as unrun.** Measured 2026-08-31: mypy 2.3.0 is a
  mypyc-compiled build shipping ~200 unsigned `.pyd` files, and Smart App Control refused them for
  a whole working day. **A force-reinstall did not clear it and neither did a fresh copy in an
  isolated venv**, so the "rewrite the file" advice that worked for pyarrow does not generalise.

  🔑 mypy also publishes a **non-compiled `py3-none-any` wheel** with zero `.pyd` files, which the
  policy has nothing to adjudicate. `--system-site-packages` keeps the project's dependencies
  visible, and `--no-deps` stops the compiled build coming back through a dependency resolution:

  ```bash
  python -m venv --system-site-packages .mypy-venv
  URL=$(python -c "import json,urllib.request;d=json.load(urllib.request.urlopen('https://pypi.org/pypi/mypy/2.3.0/json'));print(next(f['url'] for f in d['urls'] if f['filename'].endswith('py3-none-any.whl')))")
  .mypy-venv/Scripts/python.exe -m pip install --no-deps --force-reinstall "$URL"
  .mypy-venv/Scripts/python.exe -m mypy
  ```

  ⛔ Do not disable Smart App Control to get past this. It cannot be re-enabled without
  reinstalling Windows, and it has no allowlist to add an exclusion to instead. That is the user's
  decision and never a session's.

## Guards that will interrupt you, and why

Sources live in `scripts/`, deployed copies in `~/.claude/hooks/`. Each has a `*_tests.py` beside
it, and each test file has been mutation-tested: the guard was broken on purpose and the named
test watched to go red. A guard nobody has watched fail has not been tested.

⚠️ **The source and the deployed copy drift.** `session_start_hook.py` and its deployed twin
already differ by 57 lines, and nothing reports it. The newer test files assert that the deployed
copy matches the source, so a forgotten redeploy fails a test instead of silently disabling a
guard. Add that assertion to any hook you write.

- **`session_end_hook.py`** (deployed as `~/.claude/hooks/session_end_workspace.py`) closes,
  at session end, the two things that are THIS session's: the container carrying this checkout's
  label, and the MCP transports whose parent chain reaches `CLAUDE_PID`. The MCP close runs
  **before** the cwd and git checks and outside the claim gate, because the transports belong to
  the session rather than to the checkout, and because the sessions that leak are the ones that
  never opened a repository: measured 2026-08-26, the last three real rows in the log were
  `not-a-git-repo` with a home-directory cwd. It costs about 1.3s (one process listing) against a
  15s budget, and it writes `mcp` and `mcp_detail` into the row, including when it declined.
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
