"""`ReCallMemoryProvider`: Hermes's MemoryProvider over a child RE-call MCP server."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import logging
import os
import queue
import threading
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from agent.memory_provider import MemoryProvider, RecallStatus, is_trivial_prompt, spawn_context_thread

from .config import (
    CONFIG_FILE,
    DATA_DIR,
    DEFAULTS,
    SEARCH_TOOLS,
    STORE_FILE,
    Settings,
    _shared_dir_or_none,
    atomic_json_write,
    load_settings,
    normal_path,
    read_json,
    save_settings,
)
from .mcp_client import McpError, NotSent, StdioMcpClient, ToolError
from .render import render_search, render_unavailable
from .surface import READ_TOOLS, TOOLS

logger = logging.getLogger(__name__)

PROVIDER_NAME = "re-call"
PROVIDER_LABEL = "RE-call"
START_TIMEOUT_S = 120.0  # the first start loads the embedding model and refreshes the store
PREFETCH_DEADLINE_S = 6.0  # Hermes abandons an external prefetch after 8 s
PREFETCH_BUDGET_S = 7.0  # a whole prefetch, cold-start wait included, under Hermes's 8 s
MIN_SEARCH_S = 1.0  # left for the search after waiting on a starting server
TOOL_TIMEOUT_S = 60.0
RESTART_INTERVAL_S = 60.0
MIRROR_FILE = "mirror.json"


def _digest(target: str, content: str) -> str:
    return f"{target}:{hashlib.sha256(content.encode('utf-8')).hexdigest()}"


def _destination(paths: tuple[Path, Path]) -> list[str]:
    """The store and assistant folder a journal is bound to, as mirror.json records them."""
    return [normal_path(p) for p in paths]


def _store_identity(store: Path) -> str | None:
    """The store file's volume and file number. A move within one volume keeps both; a copy, or
    another store created at some path, does not."""
    try:
        status = store.stat()
    except OSError:
        return None
    return f"{status.st_dev}:{status.st_ino}" if status.st_ino else None


def _memo_text(folder: Path, memo_id: str) -> str | None:
    """The text of assistant memo `memo_id` in `folder`, read as RE-call 0.15.0 writes it
    (recall/lite/assistant_memos.py `write`: `<memo_id>.md`, optional frontmatter, the stripped text)."""
    if not memo_id or Path(memo_id).name != memo_id:
        return None
    try:
        raw = (folder / f"{memo_id}.md").read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        # Unread is not proof of anything: the caller keeps such a mapping unconfirmed and reads the
        # memo again before using it.
        return None
    if raw.startswith("---\n"):
        end = raw.find("\n---\n", 3)
        raw = raw[end + 5:] if end >= 0 else ""
    return raw.strip()


# The model-facing tools that write memory, which Hermes's `memory.write_approval` gates.
WRITE_TOOLS = ("recall_apply_fact", "recall_remember", "recall_forget_memo")
APPROVAL_UNAVAILABLE = (
    "memory.write_approval is on and nobody can approve this write now, so it was not saved to RE-call "
    "(Hermes holds pending memory writes for its own memory only). Ask the user to approve it in an "
    "interactive session, or to turn memory.write_approval off.")
APPROVAL_CHECK_FAILED = "the memory write-approval check failed, so this write was not saved to RE-call"
APPROVAL_NO_GATE = ("memory.write_approval is on but this Hermes has no write-approval gate the provider "
                    "can ask, so this write was not saved to RE-call")
# The values Hermes's tools/write_approval.py `_normalize_enabled` reads as on.
APPROVAL_TRUTHY = frozenset({"on", "true", "yes", "1", "approve", "enabled"})


def _approval_configured() -> bool:
    """memory.write_approval as Hermes's config says, read through `hermes_cli.config`, which is in the
    published plugin API: the fallback for a Hermes whose private gate module was moved or removed, so
    the write tools never write unasked while approval is on. Outside Hermes (no `hermes_cli` package)
    it is off. A config API that changed, or a read that raised, counts as on: the setting cannot be
    told, and refusing never writes unasked."""
    try:
        from hermes_cli.config import cfg_get, load_config_readonly
    except ImportError as exc:
        if isinstance(exc, ModuleNotFoundError) and exc.name == "hermes_cli":
            return False  # not running inside Hermes: there is no setting to honour
        logger.warning("re-call: Hermes's config API did not import", exc_info=True)
        return True
    try:
        value = cfg_get(load_config_readonly(), "memory", "write_approval", default=False)
    except Exception:
        logger.warning("re-call: could not read memory.write_approval", exc_info=True)
        return True
    return value is True or (isinstance(value, str) and value.strip().lower() in APPROVAL_TRUTHY)


def _gate_absent() -> bool:
    """True when this Hermes has no `tools.write_approval` at all, as opposed to one that failed while
    importing (both surface as an ImportError named "tools")."""
    try:
        return importlib.util.find_spec("tools.write_approval") is None
    except (ImportError, ValueError):
        return True  # no `tools` package either


def _write_summary(tool_name: str, arguments: dict[str, Any], folder: Path | None) -> tuple[str, str]:
    """What the approval prompt shows: one line, and every argument the write sends, so a correction
    that retires another memo, or the memory a forget deletes, is never hidden."""
    detail = json.dumps(arguments, ensure_ascii=False, indent=2, default=str)
    if tool_name == "recall_remember":
        line = f"RE-call: remember {str(arguments.get('text', ''))[:80]!r}"
        if arguments.get("supersedes"):
            line += f", replacing memo {str(arguments['supersedes'])!r}"
        return line, detail
    if tool_name == "recall_forget_memo":
        memo_id = str(arguments.get("memo_id", ""))
        text = _memo_text(folder, memo_id) if folder is not None else None
        return f"RE-call: forget memo {memo_id!r}", f"{detail}\n\nThat memo holds:\n{text}" if text else detail
    fact = " ".join(str(arguments.get(k, "")) for k in ("subject", "predicate", "object"))
    return f"RE-call: record the fact {fact[:80]!r}", detail


def _approval_refusal(tool_name: str, arguments: dict[str, Any], folder: Path | None) -> str | None:
    """Why Hermes's memory.write_approval keeps this write from RE-call, or None to write it.

    Hermes's own gate decides (an inline prompt where someone can answer). Its pending store replays
    writes into its built-in memory, so a write it would stage is refused here instead, saying why.
    A check that raises refuses while approval is on (Hermes reads an unreadable config as off), or
    when that cannot be asked either."""
    try:
        from tools import write_approval
    except ImportError as exc:
        # An error naming the gate (no `tools` package, one without `write_approval`, or a gate whose own
        # `from tools import ...` failed, all named "tools"): the setting is read through the public
        # config instead, and while it is on the refusal says whether the gate is missing or broken. Any
        # other import error is a check that failed.
        if exc.name in ("tools", "tools.write_approval"):
            if not _approval_configured():
                return None
            return APPROVAL_NO_GATE if _gate_absent() else APPROVAL_CHECK_FAILED
        logger.warning("re-call: the memory write-approval gate did not import", exc_info=True)
        return APPROVAL_CHECK_FAILED
    summary, detail = _write_summary(tool_name, arguments, folder)
    try:
        decision = write_approval.evaluate_gate(write_approval.MEMORY, inline_summary=summary,
                                                inline_detail=detail)
    except Exception:
        logger.warning("re-call: the memory write-approval check failed", exc_info=True)
        try:
            gated = write_approval.write_approval_enabled(write_approval.MEMORY)
        except Exception:
            gated = True
        return APPROVAL_CHECK_FAILED if gated else None
    if decision.allow:
        return None
    if decision.blocked:
        return decision.message or "Memory write denied by user. The change was not saved."
    return APPROVAL_UNAVAILABLE


def _held_memos(memos: dict[str, str], folder: Path) -> tuple[dict[str, str], dict[str, str]] | None:
    """The journal's mappings split into those whose memo in `folder` holds the entry's text and those
    whose memo could not be read there; None when one names a memo holding other text, or when none
    of them is found."""
    held, unread = {}, {}
    for key, memo_id in memos.items():
        text = _memo_text(folder, memo_id)
        if text is None:
            unread[key] = memo_id
            continue
        if _digest(key.rpartition(":")[0], text) != key:
            return None
        held[key] = memo_id
    return (held, unread) if held or not memos else None


def _mirror_state(raw: dict[str, Any]) -> dict[str, Any]:
    """mirror.json: `memos` maps an entry key to its memo id, `pending` the writes in flight, in
    order. The first layout was flat, `{key: {"memo_id", "entries"}}`; it is read as `memos`."""
    memos = raw.get("memos")
    if not isinstance(memos, dict):
        memos = {k: v["memo_id"] for k, v in raw.items() if isinstance(v, dict) and isinstance(v.get("memo_id"), str)}
    pending = raw.get("pending")
    # A list, because order matters and the file is written with sorted keys. Each entry is
    # [request_id, action, target, content, previous]; a malformed one is dropped rather than
    # handed to the writer thread.
    pending = [
        op for op in (pending if isinstance(pending, list) else [])
        if isinstance(op, list) and len(op) == 5 and all(isinstance(x, str) for x in op[:4])
        and (op[4] is None or isinstance(op[4], str))
    ]
    submitted = raw.get("submitted", [])
    if not isinstance(submitted, list) or not all(isinstance(x, str) for x in submitted):
        raise ToolError("invalid mirror submission state; reconciliation required")
    unconfirmed = raw.get("unconfirmed")
    unconfirmed = {k: v for k, v in (unconfirmed if isinstance(unconfirmed, dict) else {}).items()
                   if isinstance(v, str)}
    # One pair per memo removed or replaced while unreadable. A later move can leave another
    # unread memo for the same entry. Older journals recorded just one memo per entry.
    retire = raw.get("retire_when_read")
    retire = list(retire.items()) if isinstance(retire, dict) else retire
    retire = list(dict.fromkeys(
        tuple(pair) for pair in (retire if isinstance(retire, list) else [])
        if isinstance(pair, (list, tuple)) and len(pair) == 2 and all(isinstance(part, str) for part in pair)
    ))
    previous = raw.get("previous")
    previous = [p for p in (previous if isinstance(previous, list) else []) if isinstance(p, dict)]
    return {"memos": {k: v for k, v in memos.items() if isinstance(v, str)}, "pending": pending,
            "submitted": submitted, "destination": raw.get("destination"), "store": raw.get("store"),
            "unconfirmed": unconfirmed, "retire_when_read": retire, "previous": previous}


# The lock covers reading and writing mirror.json only, never a server call, so it is held briefly.
LOCK_WAIT_S = 30.0
# RE-call's refusals of a `supersedes` target (recall/lite/memo_requests.py `_target_conflict`).
TARGET_CONFLICTS = frozenset({"target_missing", "stale_target", "revision_mismatch"})


@contextlib.contextmanager
def _locked(path: Path) -> Iterator[None]:
    """An exclusive OS lock on `<path>.lock`, released by the OS if this process dies."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_name(path.name + ".lock"), "a+b") as handle:
        if os.name == "nt":
            import msvcrt

            deadline = time.monotonic() + LOCK_WAIT_S
            while True:
                try:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    if time.monotonic() > deadline:
                        raise
                    time.sleep(0.1)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class ReCallMemoryProvider(MemoryProvider):
    """Every binding comes from `initialize()`'s `hermes_home`; nothing is kept at module level."""

    def __init__(self) -> None:
        self._settings: Settings | None = None
        self._client: StdioMcpClient | None = None
        self._ready = threading.Event()
        self._stop_writer = threading.Event()
        self._start_error = ""
        self._last_start = 0.0
        self._client_lock = threading.Lock()
        self._starting = False
        self._start_done = threading.Event()  # set when a start ends, whether it succeeded or failed
        self._closed = False
        self._primary = True
        self._status: RecallStatus | None = None
        # Hermes abandons a prefetch after 8 s but its thread runs on; only the newest may set status.
        self._prefetch_token: object | None = None
        self._writes: queue.Queue[tuple[str, str, str, dict[str, Any]] | None] = queue.Queue()
        self._writer: threading.Thread | None = None

    # -- identity and activation ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return PROVIDER_NAME

    def is_available(self) -> bool:
        return not load_settings(self._home_for_probe()).problem()

    def unavailable_reason(self) -> str:
        return load_settings(self._home_for_probe()).problem()

    @staticmethod
    def _home_for_probe() -> str:
        try:
            from hermes_constants import get_hermes_home

            return str(get_hermes_home())
        except ImportError:
            return os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        self._closed = False  # an instance initialized again after shutdown() serves again
        self._stop_writer.clear()
        self._settings = load_settings(kwargs["hermes_home"])
        self._primary = kwargs.get("agent_context", "primary") == "primary"
        problem = self._settings.problem()
        if problem:
            self._start_error = problem
            logger.warning("re-call: inactive: %s", problem)
            return
        started = time.monotonic()
        starter = self._launch("re-call-start")
        if starter is not None:
            # Hermes runs the first turn right after initialize(); waiting here (bounded by
            # START_TIMEOUT_S) is what lets that turn's prefetch find a ready server.
            starter.join()
            logger.info("re-call: server %s after %.2fs", "ready" if self._ready.is_set() else "failed",
                        time.monotonic() - started)
        if self._primary and self._settings.lite and self._settings.mirror_builtin:
            # Under the lock shutdown() takes the writer in: either it is published before shutdown()
            # takes it, or shutdown() came first (during the start above) and no writer is started.
            with self._client_lock:
                if self._closed:
                    return
                writer = spawn_context_thread(self._write_loop, name="re-call-writer")
                writer.start()
                self._writer = writer

    def _launch(self, name: str) -> threading.Thread | None:
        # The check and the claim are one critical section, so two callers cannot both start a child.
        with self._client_lock:
            if self._closed or self._starting:
                return None
            self._start_done.clear()  # before _starting, so a reader that sees a start sees it unfinished
            self._starting = True
            self._last_start = time.monotonic()
        starter = spawn_context_thread(self._start_client, name=name)
        starter.start()
        return starter

    def _start_client(self) -> None:
        try:
            self._start_client_once()
        finally:
            with self._client_lock:
                self._starting = False
                self._start_done.set()  # in the lock, so it cannot mark a restart launched just after

    def _start_client_once(self) -> None:
        settings = self._settings
        assert settings is not None
        argv = settings.resolve_argv()
        if argv is None:
            self._start_error = settings.problem()
            return
        try:
            settings.data_dir.mkdir(parents=True, exist_ok=True)
            if settings.lite:
                settings.assistant_folder.mkdir(parents=True, exist_ok=True)
        except OSError as exc:  # recorded, or the provider would advertise memory it cannot serve
            self._start_error = f"the RE-call server did not start: {exc}"
            logger.warning("re-call: %s", self._start_error)
            return
        client = StdioMcpClient(
            argv,
            env=settings.child_env(os.environ),
            cwd=settings.data_dir,  # no project .env is in reach of the child
            stderr_path=settings.data_dir / "server.log",
            thread_factory=spawn_context_thread,
        )
        try:
            client.start(START_TIMEOUT_S)
        except (McpError, OSError) as exc:
            client.close()
            self._start_error = f"the RE-call server did not start: {exc}"
            logger.warning("re-call: %s (see %s)", self._start_error, settings.data_dir / "server.log")
            return
        with self._client_lock:
            if self._closed:  # shutdown() ran while this child was starting
                client.close()
                return
            self._client = client
        self._start_error = ""
        self._ready.set()

    def _call(self, tool: str, arguments: dict[str, Any], timeout: float) -> Any:
        with self._client_lock:
            client = self._client
        if client is None or not client.alive:
            self._maybe_restart()
            raise NotSent(self._start_error or "the RE-call server is starting")
        return client.call_tool(tool, arguments, timeout)

    def _maybe_restart(self) -> None:
        """Start the server again when it is not running, at most once per RESTART_INTERVAL_S."""
        if self._settings is None or self._settings.problem():
            return
        with self._client_lock:
            if self._closed or self._starting or (self._client is not None and self._client.alive):
                return
            if time.monotonic() - self._last_start < RESTART_INTERVAL_S:
                return
            if self._client is not None:
                self._client.close()
                self._client = None
            self._ready.clear()
        self._launch("re-call-restart")

    # -- per-turn recall --------------------------------------------------------------------------

    def system_prompt_block(self) -> str:
        if self._settings is None or self._start_error:
            return ""
        return (
            "Long-term memory is RE-call. Before each turn it lists memories that passed a "
            "calibrated trust check, flags superseded ones, and says when it abstained."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        token = self._prefetch_token = object()
        self._status = None
        if self._settings is None or is_trivial_prompt(query):
            return ""
        started = time.monotonic()
        if not self._ready.is_set():
            self._maybe_restart()
            # A new session's first turn arrives while the server is still starting (measured
            # 2026-11-08: ready 5.8 s after initialize): wait for it inside the budget rather than
            # give that turn no memory. A start that failed, or none in flight, is not waited on.
            if self._starting:
                self._start_done.wait(PREFETCH_BUDGET_S - MIN_SEARCH_S)
            if not self._ready.is_set():
                return ""
        timeout = min(PREFETCH_DEADLINE_S, PREFETCH_BUDGET_S - (time.monotonic() - started))
        try:
            answer = self._call("recall_search", {"query": query, "k": self._settings.k}, timeout)
        except ToolError as exc:
            return render_unavailable(str(exc)).text
        except McpError as exc:
            logger.info("re-call: prefetch skipped: %s", exc)
            return ""
        if not isinstance(answer, dict):
            logger.info("re-call: prefetch skipped: recall_search answered %s", type(answer).__name__)
            return ""
        rendered = render_search(answer)
        if rendered.citable and self._prefetch_token is token:
            self._status = RecallStatus(PROVIDER_LABEL, rendered.citable)
        return rendered.text

    def recall_status(self) -> RecallStatus | None:
        return self._status

    # -- tools ------------------------------------------------------------------------------------

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        # Hermes indexes a provider's tools in add_provider(), BEFORE initialize(): an uninitialized
        # instance answers from the active profile's settings, as is_available() does.
        settings = self._settings or load_settings(self._home_for_probe())
        if settings.problem() or (self._settings is not None and self._start_error):
            return []
        if not settings.lite:  # only the tools the child's server registers for a non-Lite store
            return [t for t in TOOLS if t["name"] in SEARCH_TOOLS]
        if self._primary:
            return list(TOOLS)
        return [t for t in TOOLS if t["name"] in READ_TOOLS]

    def handle_tool_call(self, tool_name: str, args: dict[str, Any], **kwargs: Any) -> str:
        offered = {s["name"] for s in self.get_tool_schemas()}
        if tool_name not in offered:
            return json.dumps({"error": f"re-call does not offer {tool_name} here"})
        arguments = {k: v for k, v in (args or {}).items() if v not in (None, "")}
        if tool_name in WRITE_TOOLS:
            refusal = _approval_refusal(tool_name, arguments,
                                        self._settings.assistant_folder if self._settings else None)
            if refusal:
                return json.dumps({"error": refusal})
        if tool_name == "recall_search":
            arguments.setdefault("k", self._settings.k if self._settings else 5)
        try:
            return json.dumps(self._call(tool_name, arguments, TOOL_TIMEOUT_S))
        except McpError as exc:
            return json.dumps({"error": str(exc)})

    # -- mirroring Hermes's built-in memory tool -------------------------------------------------

    def on_memory_write(self, action: str, target: str, content: str, metadata: dict[str, Any] | None = None) -> None:
        if self._writer is None:
            return
        # One id per write, minted here, so only this write's own retry can reuse it.
        self._writes.put((action, target, content, {**(metadata or {}), "_request_id": uuid.uuid4().hex}))

    def _wait_for_writer_ready(self) -> bool:
        deadline = time.monotonic() + START_TIMEOUT_S
        while not self._ready.is_set():
            if self._stop_writer.wait(0.1) or time.monotonic() >= deadline:
                return False
        return True

    def _write_loop(self) -> None:
        # Only known, unsubmitted writes can resume automatically.
        if self._wait_for_writer_ready():
            try:
                self._drain_pending()
            except Exception:
                logger.warning("re-call: could not replay journaled built-in memory writes", exc_info=True)
        while True:
            item = self._writes.get()
            if item is None:
                return
            try:
                # Journaled before the wait, so a write taken while the server is not ready (a start
                # that failed, a restart, shutdown) is replayed by the next start rather than lost.
                if not self._journal(*item):
                    continue
                if self._wait_for_writer_ready():
                    self._drain_pending()
                else:
                    logger.warning("re-call: the server is not ready; a built-in memory %s waits in %s",
                                   item[0], MIRROR_FILE)
            except Exception:  # one bad write (a corrupt mirror.json, a refusal) must not end mirroring
                logger.warning("re-call: could not mirror a built-in memory %s", item[0], exc_info=True)

    def _journal(self, action: str, target: str, content: str, metadata: dict[str, Any]) -> bool:
        """Append the write to mirror.json's pending list; False when it is not one to mirror."""
        assert self._settings is not None
        path = self._settings.data_dir / MIRROR_FILE
        previous = metadata.get("previous_content")
        if action in ("replace", "remove") and previous is None:
            return False  # an older Hermes: without the entry's identity, a destructive mirror is unsafe
        if action not in ("add", "replace", "remove"):
            return False
        # Hermes stores each entry stripped, and holds a given text at most once per target, so a key
        # is the stripped text and only says whether a memo mirrors it.
        content = content.strip()
        previous = previous.strip() if previous is not None else None
        old_key = _digest(target, previous) if previous is not None else None
        new_key = _digest(target, content)
        if action == "replace" and old_key == new_key:
            return False
        # Every read and write takes the file lock, never a server call. Unknown outcomes stay
        # pending: the server can evict request IDs, so resubmitting could recreate forgotten text.
        with _locked(path):
            state = self._bound_state(path)
            state["pending"].append([metadata["_request_id"], action, target, content, previous])
            atomic_json_write(path, state)
        return True

    def _switched_to_shared(self, recorded: Any) -> bool:
        """True when the journal names this profile's old default store, still in place, and the
        profile now uses the shared store: the switch the shared store's appearance makes."""
        assert self._settings is not None and self._settings.store_path is not None
        shared = _shared_dir_or_none()
        old = self._settings.data_dir / STORE_FILE
        return (shared is not None and isinstance(recorded, list) and bool(recorded)
                and recorded[0] == normal_path(old) and old.is_file()
                and normal_path(self._settings.store_path) == normal_path(shared / STORE_FILE))

    def _bound_state(self, path: Path) -> dict[str, Any]:
        """mirror.json, bound to this store; the caller holds the lock."""
        assert self._settings is not None and self._settings.store_path is not None
        raw = read_json(path)
        state = _mirror_state(raw)
        destination = _destination((self._settings.store_path, self._settings.assistant_folder))
        identity = _store_identity(self._settings.store_path)
        recorded = state["destination"]
        # A store at a new path (moved as the old-store notice says), or another store file at this
        # path: the journal holds only if its memo ids still name its entries' memos here. RE-call
        # numbers memos in sequence, so another store reuses the ids; and neither a matching file
        # number (reused once freed) nor vanished old paths prove it is the same store.
        changed = recorded != destination or state["store"] not in (None, identity)
        held = None
        if recorded is not None and changed:
            found = _held_memos(state["memos"], self._settings.assistant_folder)
            if found is None and self._switched_to_shared(recorded):
                # The profile's own old store is still in place and the shared one appeared, so the
                # profile switched stores rather than moving its own. Those mappings say nothing about
                # this store: they are set aside, and mirroring maps afresh here. A dsn or memo_folder
                # the user pointed elsewhere is not this case, and stays refused for reconciliation.
                logger.warning("re-call: this profile now uses %s; the mirror journal of %s is kept under "
                               "'previous' in %s and mirroring starts afresh", destination[0], recorded[0],
                               MIRROR_FILE)
                state["previous"] = [*state["previous"], {"destination": recorded, "memos": state["memos"],
                                                          "unconfirmed": state["unconfirmed"]}]
                state["unconfirmed"] = {}
                found = ({}, {})
            if found is None:
                raise ToolError("mirror journal names memos this store does not hold; "
                                "reconcile mirror.json before mirroring")
            # A mapped memo not read here may have an id free in this store, which a later memo would
            # take; it may also be in place later (a move in pieces, a file held open). So its mapping
            # stays unconfirmed: `_apply` uses it only once its memo is read holding the entry's text.
            held, unread = found
            state["memos"] = held  # so only the mappings found here are used
            state["unconfirmed"] = {**state["unconfirmed"], **unread}
        if recorded is None and (state["memos"] or state["pending"]):
            raise ToolError("mirror destination is changed or unknown; reconcile mirror.json before mirroring")
        # A missing identity (a failed stat, a filesystem without file numbers) keeps the recorded one.
        state["destination"] = destination
        state["store"] = identity if identity is not None else state["store"]
        binding = {"destination": destination, "store": state["store"]}
        if held is not None:
            binding["memos"] = held
            binding["unconfirmed"] = state["unconfirmed"]
            binding["previous"] = state["previous"]
        # Only a journal that was read: deciding by a second look at the disk could overwrite one
        # written meanwhile with this empty state. Its other fields go back as read, so an entry the
        # normalized state drops stays on disk for reconciliation.
        if raw and {k: raw.get(k) for k in binding} != binding:
            atomic_json_write(path, {**raw, **binding})  # records the identity a later move is recognised by
        return state

    def _retire_when_read(self, path: Path) -> None:
        """Forget each memo whose entry was removed or replaced before the memo could be read here,
        once it is read holding that entry's text; a memo holding other text is no longer the entry's."""
        assert self._settings is not None
        with _locked(path):
            state = self._bound_state(path)
            # A memo the journal maps now is a live entry (its id given again to a re-added text), never
            # one to retire.
            live = set(state["memos"].values())
            retire = [pair for pair in state["retire_when_read"] if pair[1] not in live]
            if len(retire) != len(state["retire_when_read"]):
                state["retire_when_read"] = retire
                atomic_json_write(path, state)
        for key, memo_id in retire:
            text = _memo_text(self._settings.assistant_folder, memo_id)
            if text is None:
                continue  # not in place yet
            if _digest(key.rpartition(":")[0], text) == key:
                try:
                    self._call("recall_forget_memo", {"memo_id": memo_id}, TOOL_TIMEOUT_S)
                except ToolError:
                    logger.warning("re-call: RE-call refused to forget %s, an entry removed before it "
                                   "was in place", memo_id, exc_info=True)
                except McpError:  # not sent, a timeout or a lost server: kept and tried on the next drain
                    logger.warning("re-call: could not yet forget %s, an entry removed before it was in "
                                   "place", memo_id, exc_info=True)
                    continue
            with _locked(path):
                state = self._bound_state(path)
                pending = state["retire_when_read"]
                if (key, memo_id) in pending:
                    pending.remove((key, memo_id))
                    atomic_json_write(path, state)

    def _drain_pending(self) -> None:
        assert self._settings is not None
        path = self._settings.data_dir / MIRROR_FILE
        self._retire_when_read(path)
        while True:
            with _locked(path):
                pending = self._bound_state(path)["pending"]
            if not pending:
                return
            op = pending[0]
            try:
                self._apply(path, *op)
            except ToolError:
                logger.warning("re-call: RE-call refused a built-in memory %s, not mirrored", op[1], exc_info=True)

    def _apply(self, path: Path, request_id: str, action: str, target: str, content: str, previous: str | None) -> None:
        old_key = _digest(target, previous) if previous is not None else None
        new_key = _digest(target, content)
        with _locked(path):
            state = self._bound_state(path)
            if all(op[0] != request_id for op in state["pending"]):
                return  # another instance on this profile applied it already
            if request_id in state["submitted"]:
                raise McpError("mirror write is in flight or unresolved; replay refused, reconciliation required")
            memos = state["memos"]
            if old_key and old_key not in memos and old_key in state["unconfirmed"]:
                candidate = state["unconfirmed"].pop(old_key)
                text = _memo_text(self._settings.assistant_folder, candidate)
                if text is None:
                    # Still unread, so this remove or replace cannot act on it now; its memo is
                    # forgotten once it is read holding the entry's text, or it would come back current.
                    retirement = (old_key, candidate)
                    if retirement not in state["retire_when_read"]:
                        state["retire_when_read"].append(retirement)
                elif _digest(target, text) == old_key:
                    memos[old_key] = candidate  # confirmed: the entry's own memo
            retired = memos.get(old_key) if old_key else None
            present = new_key in memos
            state["submitted"].append(request_id)
            atomic_json_write(path, state)
        new_memo = None
        try:
            if action == "remove" or present:
                if retired:
                    self._call("recall_forget_memo", {"memo_id": retired}, TOOL_TIMEOUT_S)
            else:
                new_memo = self._remember(content, retired, request_id)
        except ToolError:
            self._settle(path, request_id)  # refused for good: replaying it later would be wrong
            raise
        except NotSent:
            # It never reached the server, so its outcome is known: it stays pending, unclaimed, and
            # replays with its own request id once the server is back.
            with _locked(path):
                state = self._bound_state(path)
                state["submitted"] = [item for item in state["submitted"] if item != request_id]
                atomic_json_write(path, state)
            raise
        self._settle(path, request_id, old_key=old_key, retired=retired, new_key=new_key, new_memo=new_memo)

    def _settle(self, path: Path, request_id: str, *, old_key: str | None = None, retired: str | None = None,
                new_key: str = "", new_memo: str | None = None) -> None:
        with _locked(path):
            state = self._bound_state(path)
            if all(op[0] != request_id for op in state["pending"]):
                return  # settled meanwhile by another instance, whose later writes this must not undo
            if old_key and retired and state["memos"].get(old_key) == retired:
                del state["memos"][old_key]
            if new_memo:
                state["memos"][new_key] = new_memo
            state["pending"] = [op for op in state["pending"] if op[0] != request_id]
            state["submitted"] = [item for item in state["submitted"] if item != request_id]
            atomic_json_write(path, state)

    def _remember(self, text: str, supersedes: str | None, request_id: str) -> str:
        arguments = {"text": text, "request_id": f"hermes-{request_id}"}
        if supersedes:
            arguments["supersedes"] = supersedes
        answer = self._call("recall_remember", arguments, TOOL_TIMEOUT_S)
        if isinstance(answer, dict) and answer.get("conflict") in TARGET_CONFLICTS and supersedes:
            # The memo it would correct is gone or already corrected (the model's own tools act on the
            # same folder): store the new text on its own rather than lose it. Any other refusal, such
            # as `memo_missing` for a memo the user forgot since, is final.
            answer = self._call("recall_remember", {"text": text, "request_id": f"hermes-{request_id}-new"},
                                TOOL_TIMEOUT_S)
        if not isinstance(answer, dict) or not answer.get("memo_id"):
            raise ToolError(f"recall_remember refused: {answer}")  # an answer, so not one to replay
        return str(answer["memo_id"])

    # -- configuration ----------------------------------------------------------------------------

    def get_config_schema(self) -> list[dict[str, Any]]:
        # `hermes memory setup` offers each default and saves what Enter keeps, so the defaults are
        # the profile's saved values: running setup again must not reset them.
        try:
            saved = {**DEFAULTS, **read_json(Path(self._home_for_probe()) / DATA_DIR / CONFIG_FILE)}
        except ValueError:  # an unreadable config.json, which problem() reports; setup offers the defaults
            saved = dict(DEFAULTS)
        return [
            {"key": "recall_command", "description": "Path to RE-call's recall-mcp (0.15.0 or later)",
             "default": saved["recall_command"]},
            {"key": "dsn", "description": "RE-call store DSN (blank: the Lite store your agents share)",
             "default": saved["dsn"]},
            {"key": "memo_folder",
             "description": "Memo folder the store indexes (blank: memos/ beside the default store; under this "
                            "profile when dsn is set)",
             "default": saved["memo_folder"]},
            {"key": "k", "description": "Memories to retrieve per turn", "type": "integer",
             "default": saved["k"], "minimum": 1, "maximum": 20},
            {"key": "mirror_builtin", "description": "Mirror Hermes's built-in memory writes into RE-call",
             "type": "boolean", "default": saved["mirror_builtin"]},
        ]

    def save_config(self, values: dict[str, Any], hermes_home: str) -> None:
        save_settings(values, hermes_home)

    def on_profile_renamed(self, old_name: str, new_name: str, hermes_home: str) -> None:
        """Hermes moved the profile's directory: a journal bound to the store at its old path is bound
        to the same store at its new one, so mirroring goes on."""
        new_home = Path(hermes_home)
        settings = load_settings(new_home)
        path = settings.data_dir / MIRROR_FILE
        if settings.store_path is None or not path.is_file():
            return
        old_home = new_home.with_name(old_name)
        current = (settings.store_path, settings.assistant_folder)
        moved = tuple(old_home / p.relative_to(new_home) if p.is_relative_to(new_home) else p for p in current)
        with _locked(path):
            state = _mirror_state(read_json(path))
            if state["destination"] == _destination(moved):
                state["destination"] = _destination(current)
                atomic_json_write(path, state)

    # -- teardown ---------------------------------------------------------------------------------

    def shutdown(self) -> None:
        self._stop_writer.set()
        with self._client_lock:
            self._closed = True  # a start in flight, or a late thread, must not revive the server
            writer, self._writer = self._writer, None
        if writer is not None:
            self._writes.put(None)
            writer.join(timeout=10)
        with self._client_lock:
            client, self._client = self._client, None
        self._ready.clear()
        if client is not None:
            client.close()
