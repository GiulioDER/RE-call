"""Keep a project's memo folder indexed without making the end of a session wait for it (#1454).

Codex runs `SessionEnd` synchronously and kills it at 3 s, whatever its config asks for
(`SESSION_END_MAX_TIMEOUT_SEC` in openai/codex). Measured 2026-11-19 on Windows with a PostgreSQL
store and the hashing embedder: re-indexing an UNCHANGED folder of 1,000 memos cost a median 3.3 s
in process, and a first index of the same folder 37.8 s. Even the check for "unchanged" costs
about 0.8 s once the count refresh's `import psycopg` is included. So the hook only starts
`python -m recall_hooks index-memory <cwd>`, detached, and returns. That child:

1. compares a stat walk of the folder with the walk recorded after the last successful index. The
   comparison is equality, not "newer than": a renamed, deleted or back-dated memo changes the
   walk as surely as an edited one, and no clock is trusted to order anything;
2. indexes only when they differ, or when the tenant has lost its rows;
3. refreshes the cached count either way.

One child works at a time (a lock held for its whole life, which the OS releases if it dies), and
each is killed at `TIME_LIMIT_SECONDS`. Codex keeps a detached child running when the hook it came
from completed in time (`command_runner.rs`); the hook now does too little to come near the cap.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import IO, Any

STATE_NAME = "recall-memory-index.json"
LOCK_NAME = "recall-memory-index.lock"
#: A child still running then is killed. Whatever it indexed stays indexed, and the next child
#: skips those files by content hash, so a folder too large for one run still converges.
TIME_LIMIT_SECONDS = 30 * 60
#: A memo modified this close to the walk may be written again within the same timestamp tick
#: without its size or mtime changing, so a walk that includes one is never recorded.
RACY_WINDOW_NS = 2_000_000_000
#: Index passes per child: one, plus up to two for memos written while it ran.
MAX_PASSES = 3


def _home() -> Path:
    # Beside the hook config, so Codex and Claude Code keep separate state.
    from recall_hooks import config_path

    return config_path().parent


def folder_key(config: dict[str, Any], memory_dir: Path) -> str:
    """Which store and which code a recorded walk belongs to.

    A walk recorded against one tenant, table, embedder or database says nothing about another, and
    an upgrade may change how memos are chunked, so each of them is part of the key.
    """
    from recall import __version__

    material = [
        str(memory_dir.resolve()),
        hashlib.sha256(str(config.get("dsn", "")).encode("utf-8")).hexdigest(),
        str(config.get("tenant", "default")),
        str(config.get("table", "chunks")),
        str(config.get("embedder", "fastembed")),
        __version__,
    ]
    return hashlib.sha256("\x00".join(material).encode("utf-8")).hexdigest()


def walk(memory_dir: Path) -> tuple[str, int, bool]:
    """`(digest, memo count, racy)` over every `.md` file under `memory_dir`.

    A superset of what the indexer reads, so any change it would see changes the digest; a change
    it would not see costs one background pass that skips every file by content hash. Opens no
    file and imports nothing: `import recall.index` alone took 1.4 to 2.7 s here.
    """
    digest = hashlib.sha256()
    count = 0
    racy = False
    now = time.time_ns()
    for directory, subdirectories, names in os.walk(memory_dir):
        subdirectories.sort()
        for name in sorted(names):
            if not name.lower().endswith(".md"):
                continue
            path = Path(directory, name)
            rel = path.relative_to(memory_dir).as_posix()
            try:
                found = path.stat()
            except OSError:
                # Vanished or unreadable mid-walk: part of the digest, and never recorded.
                digest.update(f"{rel}\x00unreadable\x00".encode())
                racy = True
                continue
            digest.update(
                f"{rel}\x00{found.st_size}:{found.st_mtime_ns}:{found.st_ctime_ns}\x00".encode()
            )
            count += 1
            racy = racy or abs(now - found.st_mtime_ns) < RACY_WINDOW_NS
    return digest.hexdigest(), count, racy


def _recorded() -> dict[str, Any]:
    try:
        document = json.loads((_home() / STATE_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    folders = document.get("folders") if isinstance(document, dict) else None
    return folders if isinstance(folders, dict) else {}


def _record(key: str, digest: str) -> None:
    from recall.atomic_write import atomic_write_bytes

    folders = _recorded()
    folders[key] = {"digest": digest, "indexed_at": time.time()}
    document = {"version": 1, "folders": folders}
    atomic_write_bytes(_home() / STATE_NAME, json.dumps(document, indent=2).encode("utf-8"))


def start_background_index(cwd: str) -> None:
    """Start the detached child for the project at `cwd` and return at once."""
    home = _home()
    home.mkdir(parents=True, exist_ok=True)
    flags = 0
    if os.name == "nt":
        flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(
            subprocess, "DETACHED_PROCESS", 0
        )

    def spawn(extra: int) -> None:
        subprocess.Popen(
            [sys.executable, "-P", "-m", "recall_hooks", "index-memory", cwd],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            # As the write-time relay does: without it the child inherits the hook's pipe
            # handles on Windows and the client waits for the child to exit.
            close_fds=True,
            # Not the session's directory, which the child would otherwise keep from being
            # deleted on Windows for as long as it runs.
            cwd=str(home),
            start_new_session=os.name != "nt",
            creationflags=flags | extra,
        )

    breakaway = getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0)
    try:
        # Out of the client's job object, if it has one, so the child outlives the hook.
        spawn(breakaway)
    except OSError:
        # A job that forbids breakaway refuses the flag; the child then lives as long as the job.
        if not breakaway:
            raise
        spawn(0)


def _try_lock(handle: IO[bytes]) -> bool:
    handle.seek(0)
    try:
        if os.name == "nt":
            msvcrt = importlib.import_module("msvcrt")
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            fcntl = importlib.import_module("fcntl")
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    return True


def run(args: list[str]) -> int:
    """`python -m recall_hooks index-memory <cwd>`: the detached child."""
    # The bound, armed first so it also covers the wait for another child's lock.
    watchdog = threading.Timer(TIME_LIMIT_SECONDS, os._exit, args=(3,))
    watchdog.daemon = True
    watchdog.start()
    try:
        return _run(args)
    finally:
        watchdog.cancel()


def _run(args: list[str]) -> int:
    from recall.memory_index import index_memory_directory
    from recall_hooks import _project_memory_dir, load_config, refresh_stats

    if len(args) != 1:
        return 2
    config = load_config()
    dsn = config.get("dsn")
    if not dsn or str(dsn).startswith("sqlite:///"):
        return 0
    # The same project boundary the hook applied, checked again: this is an entry point anyone can
    # run, and the config may have changed since the hook read it.
    memory_dir = _project_memory_dir(config, args[0])
    if memory_dir is None or not memory_dir.is_dir():
        return 0
    home = _home()
    home.mkdir(parents=True, exist_ok=True)
    with (home / LOCK_NAME).open("a+b") as lock:
        while not _try_lock(lock):
            time.sleep(1.0)
        key = folder_key(config, memory_dir)
        # Read only once the lock is held, so a walk another child recorded while this one waited
        # counts as done.
        entry = _recorded().get(key)
        done = entry.get("digest") if isinstance(entry, dict) else None
        # The settings the corpus was built under (an embedding profile, its key), which the
        # client's own environment need not carry; without them another model would embed.
        extra = config.get("server_env")
        env = (
            {**os.environ, **extra}
            if isinstance(extra, dict) and all(isinstance(v, str) for v in extra.values())
            else None
        )
        first = True
        for _ in range(MAX_PASSES):
            digest, count, racy = walk(memory_dir)
            # An unchanged folder beside a tenant with no rows (a recreated database, say) is
            # indexed again rather than trusted, which is what every session end did before #1454.
            if digest == done and not (first and count and refresh_stats(config) == 0):
                break
            first = False
            indexed = index_memory_directory(
                dsn=str(dsn),
                embedder_name=str(config.get("embedder", "fastembed")),
                memory_dir=memory_dir,
                tenant=str(config.get("tenant", "default")),
                table=str(config.get("table", "chunks")),
                env=env,
                print_fn=lambda *args, **kwargs: None,
            )
            if not indexed:
                break
            done = digest
            if not racy:
                _record(key, digest)
        refresh_stats(config)
    return 0
