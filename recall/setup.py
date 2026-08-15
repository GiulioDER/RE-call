from __future__ import annotations

import importlib.util
import json
import os
import shutil
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Sequence

from recall.calibration import Calibration, from_samples, save
from recall.embeddings import resolve_embedder
from recall.eval.calibrate import CalibrationReport

SETUP_BEGIN = "# recall setup begin"
SETUP_END = "# recall setup end"
DEFAULT_ENV_PATH = Path(".env")

#: OpenAI compatible endpoints the reasoning arm can be pointed at. The base URL is the only
#: record of which provider was chosen: `_host_of` in the truth extraction engine already turns
#: it into an audit identity, so a separate provider name would be a second source of truth for
#: the same fact.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENAI_BASE_URL = "https://api.openai.com/v1"

#: Sentinel for "ask me for a base URL", distinct from any real URL.
LOCAL_PROVIDER = "local"

#: Ollama's OpenAI compatible endpoint, offered as the default for a local server.
LOCAL_BASE_URL_DEFAULT = "http://localhost:11434/v1"

#: A local server ignores the key, and the OpenAI client refuses an empty one.
LOCAL_API_KEY = "local"

DEFAULT_CALIBRATION_PATH = Path("calibration.json")
MODEL_DOWNLOAD_FLOOR_BYTES = 1_500_000_000
CLAUDE_MD_BEGIN = "<!-- recall setup begin -->"
CLAUDE_MD_END = "<!-- recall setup end -->"
DEFAULT_CLAUDE_MD_PATH = Path("CLAUDE.md")
DEFAULT_MEMORY_DIR = Path("memory")


@dataclass(frozen=True)
class HardwareProbe:
    cpu_count: int | None
    gpu: str | None
    cuda_available: bool
    free_bytes: int
    internet: bool
    fastembed_available: bool
    sentence_transformers_available: bool


@dataclass(frozen=True)
class Choice:
    label: str
    value: str
    description: str
    #: False when this machine cannot run the option yet. It is still offered, because hiding it
    #: makes the product look like it does not have the feature, and leaves no way to ask for it.
    available: bool = True
    #: What to install to make an unavailable option work. Shown when it is selected.
    unavailable_note: str = ""
    #: Vector width this option produces, where it is fixed and known. Declared rather than
    #: derived, because deriving it means constructing the embedder, and constructing a fastembed
    #: one downloads its weights. Asking "would this fit your table" must not cost 1.2 GB.
    dim: int | None = None


@dataclass(frozen=True)
class CalibrationResult:
    path: Path
    calibration: Calibration
    report: CalibrationReport


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _probe_gpu() -> str | None:
    try:
        import torch
    except Exception:
        torch = None
    if torch is not None:
        try:
            if torch.cuda.is_available():
                return str(torch.cuda.get_device_name(0))
        except Exception:
            pass
        try:
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "apple mps"
        except Exception:
            pass
    if shutil.which("nvidia-smi"):
        return "nvidia-smi"
    return None


def _probe_cuda() -> bool:
    try:
        import torch
    except Exception:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _probe_internet(url: str = "https://pypi.org", timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout):  # noqa: S310 - install-time probe
            return True
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return False


def probe_hardware(path: Path | None = None) -> HardwareProbe:
    root = path or Path.cwd()
    free_bytes = shutil.disk_usage(root).free
    cuda_available = _probe_cuda()
    return HardwareProbe(
        cpu_count=os.cpu_count(),
        gpu=_probe_gpu(),
        cuda_available=cuda_available,
        free_bytes=free_bytes,
        internet=_probe_internet(),
        fastembed_available=_module_available("fastembed"),
        sentence_transformers_available=_module_available("sentence_transformers"),
    )


def _can_download_models(probe: HardwareProbe) -> bool:
    return probe.internet and probe.free_bytes >= MODEL_DOWNLOAD_FLOOR_BYTES


def embedder_choices(
    probe: HardwareProbe,
    *,
    security_required: bool,
    cloud_keys: dict[str, str],
) -> list[Choice]:
    choices = [
        Choice(
            label="hashing",
            value="hashing",
            description="No model download, no network, works anywhere",
            dim=64,
        ),
    ]
    if probe.fastembed_available and _can_download_models(probe):
        choices.append(
            Choice(
                label="fastembed",
                value="fastembed",
                description="Local bge-small embedder, best default when offline matters",
                dim=384,
            )
        )
        # Larger bge models, gated on their own download size rather than the shared floor: the
        # floor is 1.5 GB and bge-large alone is 1.2 GB, so a machine can clear the floor and
        # still have no room left. Each needs its weights plus headroom to unpack them.
        if probe.free_bytes >= 2 * 1024**3:
            choices.append(
                Choice(
                    label="fastembed base",
                    value="fastembed:BAAI/bge-base-en-v1.5",
                    description="Local bge-base, 768 dims, ~210 MB, the best size for quality",
                    dim=768,
                )
            )
        if probe.free_bytes >= 4 * 1024**3:
            choices.append(
                Choice(
                    label="fastembed large",
                    value="fastembed:BAAI/bge-large-en-v1.5",
                    description="Local bge-large, 1024 dims, ~1.2 GB, best quality and slowest",
                    dim=1024,
                )
            )
    if probe.sentence_transformers_available and _can_download_models(probe):
        choices.append(
            Choice(
                label="sentence transformers",
                value="st:sentence-transformers/all-MiniLM-L6-v2",
                description="Local transformer embedder, good if you already have the extra",
                dim=384,
            )
        )
    voyage_key = cloud_keys.get("VOYAGE_API_KEY", "")
    openai_key = cloud_keys.get("OPENAI_API_KEY", "") or cloud_keys.get(
        "OPENROUTER_API_KEY", ""
    )
    if (
        not security_required
        and probe.internet
        and voyage_key
        and _module_available("voyageai")
    ):
        choices.append(
            Choice(
                label="voyage cloud",
                value="voyage:voyage-3",
                description="Cloud embedder, only when data may leave the machine",
                dim=1024,
            )
        )
    if (
        not security_required
        and probe.internet
        and openai_key
        and _module_available("openai")
    ):
        choices.append(
            Choice(
                label="openai compatible cloud",
                value="openai:text-embedding-3-small",
                description="OpenAI compatible cloud path through OpenRouter or OpenAI",
                dim=1536,
            )
        )
    return choices


def _schema_dim_conflict(dsn: str, expected_dim: int, table: str | None = None) -> str | None:
    """The library's own incompatibility message if `table` disagrees on width, else None.

    Asks `check_schema` rather than reading `pg_attribute` here, so there is one authority on
    what "compatible" means and this cannot drift from it.

    `table` matters: the wizard must check the table the caller actually uses. Checking a
    hard-coded `chunks` would report a conflict against a leftover table a `--table` user does
    not care about, and stay silent about the one they do.

    Only `SchemaIncompatible` is a conflict. Every other failure returns None on purpose: no
    database yet, no table yet, no psycopg, no permission, and migrations pending, none of which
    mean the width is wrong, and the wizard has always been runnable before a schema exists.
    A table that is behind on migrations has an unknown width rather than a wrong one, so it is
    not grounds for refusing a choice.
    """
    try:
        from recall.schema import SchemaIncompatible
        from recall.store import DEFAULT_TABLE, PgVectorStore
    except Exception:
        return None
    try:
        with PgVectorStore(dsn, dim=expected_dim, table=table or DEFAULT_TABLE) as store:
            store.check_schema()
    except SchemaIncompatible as exc:
        return str(exc)
    except Exception:
        return None
    return None


def _table_row_count(dsn: str, table: str | None = None) -> int | None:
    """Return the number of rows in `table`, or None when it cannot be read safely here."""
    try:
        from psycopg import sql
        from recall.schema import _connect
        from recall.store import DEFAULT_TABLE
    except Exception:
        return None
    try:
        with _connect(dsn) as conn:
            ident = sql.Identifier(table or DEFAULT_TABLE)
            query = sql.SQL("SELECT count(*) FROM {}").format(ident)
            row = conn.execute(query).fetchone()
    except Exception:
        return None
    return int(row[0]) if row else 0


def _schema_prepare_state(
    dsn: str,
    expected_dim: int,
    table: str | None = None,
) -> tuple[Literal["compatible", "needs_apply", "conflict", "unknown"], str | None]:
    """Classify what setup needs to do before this embedder can be used."""
    try:
        from recall.schema import SchemaIncompatible, SchemaTooOld, _connect, check_schema
        from recall.store import DEFAULT_TABLE
    except Exception:
        return "unknown", None
    try:
        with _connect(dsn) as conn:
            check_schema(conn, table=table or DEFAULT_TABLE, dim=expected_dim)
    except SchemaTooOld as exc:
        return "needs_apply", str(exc)
    except SchemaIncompatible as exc:
        return "conflict", str(exc)
    except Exception:
        return "unknown", None
    return "compatible", None


def _table_row_counts(dsn: str, tables: Sequence[str]) -> dict[str, int] | None:
    """Return row counts for the named tables, treating absent tables as zero rows."""
    try:
        from psycopg import sql
        from recall.schema import _connect
    except Exception:
        return None
    counts: dict[str, int] = {}
    try:
        with _connect(dsn) as conn:
            for table in tables:
                exists = conn.execute("SELECT to_regclass(%s)", (table,)).fetchone()
                if not exists or exists[0] is None:
                    counts[table] = 0
                    continue
                query = sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(table))
                row = conn.execute(query).fetchone()
                counts[table] = int(row[0]) if row else 0
    except Exception:
        return None
    return counts


def _drop_default_schema_family(migration_dsn: str) -> None:
    """Drop the default serving table plus the global generation tables and migration ledger."""
    from psycopg import sql
    from recall.schema import GENERATION_TABLES, GLOBAL_MIGRATION_TARGET, LEDGER_TABLE, _connect
    from recall.store import DEFAULT_TABLE, PgVectorStore

    with PgVectorStore(migration_dsn, dim=1, table=DEFAULT_TABLE) as store:
        store.drop_table()
    with _connect(migration_dsn) as conn:
        for table in GENERATION_TABLES:
            conn.execute(sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(sql.Identifier(table)))
        ledger = conn.execute("SELECT to_regclass(%s)", (LEDGER_TABLE,)).fetchone()
        if ledger and ledger[0]:
            conn.execute(
                sql.SQL("DELETE FROM {} WHERE target_table = %s").format(sql.Identifier(LEDGER_TABLE)),
                (GLOBAL_MIGRATION_TARGET,),
            )


def _prepare_schema_for_embedder(
    *,
    dsn: str,
    migration_dsn: str | None,
    table: str | None,
    embedder: Choice,
    print_fn: Callable[..., None],
) -> None:
    """Make the selected embedder usable now, or stop with a message the user can act on."""
    if embedder.dim is None:
        return
    from recall.schema import GENERATION_TABLES, apply_migrations
    from recall.store import DEFAULT_TABLE, PgVectorStore

    ddl_dsn = migration_dsn or dsn
    state, detail = _schema_prepare_state(dsn, embedder.dim, table)
    target_table = table or DEFAULT_TABLE

    if state == "compatible":
        return
    if state == "needs_apply":
        if detail:
            print_fn(detail)
        try:
            applied = apply_migrations(ddl_dsn, table=target_table, dim=embedder.dim)
        except Exception as exc:
            raise SystemExit(
                "The wizard could not prepare the schema automatically. Pass --migration-dsn if "
                "the serving DSN is read only, and verify that the chosen role can create tables "
                f"and indexes. Original error: {type(exc).__name__}: {exc}"
            ) from exc
        if applied:
            print_fn(f"Prepared {target_table!r} for {embedder.dim} dimensions.")
        else:
            print_fn(f"Verified {target_table!r} for {embedder.dim} dimensions.")
        return
    if state == "unknown":
        return

    if detail:
        print_fn(detail)
    if target_table == DEFAULT_TABLE:
        counts = _table_row_counts(ddl_dsn, (DEFAULT_TABLE, *GENERATION_TABLES))
        if counts is None:
            raise SystemExit(
                "The wizard could see that the default schema does not match the selected "
                "embedder, but it could not safely inspect whether the default tables are empty. "
                "Check the database manually, then rerun setup."
            )
        non_empty = {name: count for name, count in counts.items() if count > 0}
        if non_empty:
            details = ", ".join(f"{name}={count}" for name, count in sorted(non_empty.items()))
            raise SystemExit(
                f"The selected embedder needs vector({embedder.dim}), but the default schema "
                f"already contains data: {details}. I will not rebuild those tables "
                "automatically. Use an embedder matching the existing schema, or point setup at a "
                "fresh table name or database."
            )
        print_fn(
            f"The selected embedder needs vector({embedder.dim}). The default schema is empty, so "
            "the wizard will rebuild it now."
        )
        try:
            _drop_default_schema_family(ddl_dsn)
            applied = apply_migrations(ddl_dsn, table=target_table, dim=embedder.dim)
        except Exception as exc:
            raise SystemExit(
                "The wizard could not rebuild the default schema automatically. Verify that the "
                "chosen role can drop and create the RE-call tables, or pass --migration-dsn with "
                f"the owner role. Original error: {type(exc).__name__}: {exc}"
            ) from exc
        if applied:
            print_fn(f"Prepared {target_table!r} for {embedder.dim} dimensions.")
        else:
            print_fn(f"Verified {target_table!r} for {embedder.dim} dimensions.")
        return

    rows = _table_row_count(ddl_dsn, target_table)
    if rows is None:
        raise SystemExit(
            "The wizard could see that this table does not match the selected embedder, but it "
            "could not safely inspect whether the table is empty. Check the table manually, then "
            "rerun setup."
        )
    if rows > 0:
        raise SystemExit(
            f"The selected embedder needs vector({embedder.dim}), but this table already holds "
            f"{rows} row(s). I will not rebuild a populated table automatically. Use an embedder "
            "matching the existing table, or point setup at a fresh table name."
        )
    print_fn(
        f"The selected embedder needs vector({embedder.dim}). The current table is empty, so the "
        "wizard will rebuild it now."
    )
    try:
        with PgVectorStore(ddl_dsn, dim=embedder.dim, table=target_table) as store:
            store.drop_table()
        applied = apply_migrations(ddl_dsn, table=target_table, dim=embedder.dim)
    except Exception as exc:
        raise SystemExit(
            "The wizard could not rebuild the selected table automatically. Verify that the "
            "chosen role can drop and create the RE-call table, or pass --migration-dsn with the "
            f"owner role. Original error: {type(exc).__name__}: {exc}"
        ) from exc
    if applied:
        print_fn(f"Prepared {target_table!r} for {embedder.dim} dimensions.")
    else:
        print_fn(f"Verified {target_table!r} for {embedder.dim} dimensions.")


def _why_unavailable(probe: HardwareProbe, *, needs_cuda: bool = False) -> str:
    """The conditions that actually failed, not the one that usually fails.

    These mirror the guards the callers use to compute `runnable`, and must stay in step with
    them: a note naming sentence-transformers to somebody who already has it, when the real
    blocker is free disk, sends them to fix the wrong thing.
    """
    missing: list[str] = []
    if needs_cuda and not probe.cuda_available:
        missing.append("no CUDA device was detected")
    if not probe.sentence_transformers_available:
        missing.append("sentence-transformers is not installed")
    if not _can_download_models(probe):
        missing.append("there is not enough internet or free disk to download the model")
    return "; ".join(missing) if missing else "this machine cannot run it"


def reranker_choices(
    probe: HardwareProbe, *, security_required: bool
) -> list[Choice]:
    runnable = probe.sentence_transformers_available and _can_download_models(probe)
    note = (
        f"Reranking is unavailable here: {_why_unavailable(probe)}. The extra is "
        'pip install "recall-rag[rerank]". Rerun setup and choose it again once it can run.'
    )
    choices = [Choice(label="none", value="", description="Skip reranking for speed")]
    choices.append(
        Choice(
            label="ms marco reranker",
            value="RECALL_RERANK=1",
            description="Default local cross encoder, the measured rerank win",
            available=runnable,
            unavailable_note=note,
        )
    )
    # `security_required` hides bge as a policy decision rather than a capability one, so it stays
    # hidden rather than being listed as unavailable: nothing the reader installs would reveal it.
    if not security_required:
        choices.append(
            Choice(
                label="bge reranker",
                value="RECALL_RERANK=1;RECALL_RERANK_MODEL=BAAI/bge-reranker-base",
                description="Heavier local reranker, available if you want to try it",
                available=runnable,
                unavailable_note=note,
            )
        )
    return choices


def sparse_choices(probe: HardwareProbe) -> list[Choice]:
    choices = [
        Choice(
            label="postgres fts",
            value="RECALL_SPARSE=fts",
            description="Current sparse retrieval path, no extra GPU requirement",
        )
    ]
    runnable = (
        probe.cuda_available
        and probe.sentence_transformers_available
        and _can_download_models(probe)
    )
    choices.append(
        Choice(
            label="splade",
            value="RECALL_SPARSE=splade",
            description="CUDA sparse encoder for higher coverage when the machine can run it",
            available=runnable,
            unavailable_note=(
                f"SPLADE is unavailable here: {_why_unavailable(probe, needs_cuda=True)}. The "
                'extra is pip install "recall-rag[sparse]". Rerun setup and choose it again once '
                "it can run."
            ),
        )
    )
    return choices


def entailment_choices(probe: HardwareProbe) -> list[Choice]:
    choices = []
    if probe.sentence_transformers_available and _can_download_models(probe):
        choices.append(
            Choice(
                label="qnli judge",
                value="RECALL_ENTAILMENT=1;RECALL_ENTAILMENT_MODEL=cross-encoder/qnli-distilroberta-base;RECALL_ENTAILMENT_REVISION=7dd04ee0a6040c06fb381ad7edcb8585f4d937fd",
                description="Default optional entailment judge, pinned and local",
            )
        )
        choices.append(
            Choice(
                label="nli judge",
                value="RECALL_ENTAILMENT=1;RECALL_ENTAILMENT_MODEL=cross-encoder/nli-deberta-v3-large",
                description="Stronger local judge, larger and unpinned by default",
            )
        )
    return choices


def reasoning_provider_choices(
    probe: HardwareProbe,
    *,
    security_required: bool,
) -> list[Choice]:
    """Providers for the optional reasoning arm, local first.

    Local leads because `_choose` refuses a menu whose first entry is unavailable, and a local
    endpoint is the only option that needs neither a key nor internet. It is therefore the only
    entry that can be offered unconditionally.

    Cloud entries are withheld entirely under `security_required`, rather than being offered and
    marked. That differs from how an uninstalled package is handled, and deliberately: a missing
    package is a thing the user can go and fix, whereas the security answer is a decision they
    already made, and re-offering it invites them to undo it by accident.
    """
    choices = [
        Choice(
            label="local endpoint",
            value=LOCAL_PROVIDER,
            description="An OpenAI compatible server you run yourself, such as Ollama or vLLM",
        )
    ]
    if security_required:
        return choices

    openai_installed = _module_available("openai")
    blockers: list[str] = []
    if not openai_installed:
        blockers.append('the openai package is not installed, pip install "recall-rag[extract]"')
    if not probe.internet:
        blockers.append("no internet connection was detected")
    note = "; ".join(blockers)
    runnable = openai_installed and probe.internet

    choices.append(
        Choice(
            label="openrouter",
            value=OPENROUTER_BASE_URL,
            description="Many providers behind one key, including cheap models",
            available=runnable,
            unavailable_note=note,
        )
    )
    choices.append(
        Choice(
            label="openai",
            value=OPENAI_BASE_URL,
            description="OpenAI directly, for an existing OpenAI key",
            available=runnable,
            unavailable_note=note,
        )
    )
    return choices


#: Sentinel value for the "type an id yourself" entry. The empty string cannot collide with a
#: real model id, so it needs no separate flag on `Choice`.
MANUAL_MODEL = ""


def reasoning_model_choices(base_url: str) -> list[Choice]:
    """Models offered for a cloud provider, default first, always ending in manual entry.

    The OpenRouter ids were verified against its live catalogue on 2026-08-14, which is how a
    sixth candidate that no longer exists was caught before shipping. They will still go stale:
    this is a static list inside a released artifact, and the provider's roster is not ours. The
    manual entry is the mitigation and must stay last on every provider.

    Descriptions carry no prices. Prices were read during design and left out on purpose, because
    a number in a shipped menu is a measurement nothing re-checks.
    """
    manual = Choice(
        label="enter a model id",
        value=MANUAL_MODEL,
        description="Type an id yourself, for anything not listed",
    )
    if base_url == OPENAI_BASE_URL:
        return [
            Choice(
                label="gpt-4o mini",
                value="gpt-4o-mini",
                description="Cheap and fast, a good default",
            ),
            Choice(
                label="gpt-4o",
                value="gpt-4o",
                description="Higher quality and several times the price",
            ),
            manual,
        ]
    return [
        Choice(
            label="gpt-4o mini",
            value="openai/gpt-4o-mini",
            description="Small, fast and inexpensive, the default",
        ),
        Choice(
            label="deepseek chat",
            value="deepseek/deepseek-chat",
            description="Low cost general model, strong for the price",
        ),
        Choice(
            label="deepseek r1",
            value="deepseek/deepseek-r1",
            description="Reasoning tuned, slower and dearer than chat",
        ),
        Choice(
            label="llama 3.3 70b",
            value="meta-llama/llama-3.3-70b-instruct",
            description="Open weights, the cheapest option here",
        ),
        Choice(
            label="claude sonnet 4.5",
            value="anthropic/claude-sonnet-4.5",
            description="Best quality and the dearest, matching the truth extraction default",
        ),
        manual,
    ]


def probe_reasoning_model(
    *,
    base_url: str,
    api_key: str,
    model: str,
    timeout: float = 20.0,
) -> str | None:
    """Send one minimal completion. `None` when it worked, otherwise what went wrong.

    A wrong key, a retired model id and a local server that is not running are the three likely
    failures, and each is far cheaper to find here than on the reader's first reasoning query.

    This never raises. It runs against three different providers and a base URL the user typed,
    so the set of reachable exception types is not knowable from here, and letting one escape
    would turn an optional step into a failed install. The caller prints the string and writes
    the configuration regardless.

    The call runs on a daemon thread joined with `timeout`, rather than relying on the client's
    own `timeout` alone. httpx treats a bare float as separate connect, read, write and pool
    timeouts, and its read timeout only bounds the gap between chunks of a streamed response, not
    the total call. A slow endpoint that trickles bytes could otherwise keep resetting that timer
    forever. The thread is a daemon so a call that never returns cannot block interpreter exit.
    """
    if not _module_available("openai"):
        return 'the openai package is not installed, install "recall-rag[extract]"'

    outcome: list[str | None] = [f"no response within {timeout:.0f}s"]

    def call() -> None:
        try:
            from openai import OpenAI

            client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
                max_retries=0,
            )
            # No token cap is sent. `max_tokens` is rejected outright by OpenAI's o series and by
            # GPT-5, which want `max_completion_tokens`, so capping here would report a working
            # model as unreachable. Those models are not in the menu but manual entry can name
            # one, and a probe that cries failure over a working configuration is worse than no
            # probe at all. The reply to "ping" is a few tokens, which is not worth a
            # compatibility branch.
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "ping"}],
            )
            outcome[0] = None
        except Exception as exc:
            outcome[0] = f"{type(exc).__name__}: {exc}"

    try:
        thread = threading.Thread(target=call, daemon=True)
        thread.start()
        thread.join(timeout)
    except Exception as exc:
        # Starting or joining the thread is outside `call`'s own try/except, so a failure here
        # (an OS refusing a new thread is the realistic case) needs its own net. The never raise
        # rule covers this path too, not just what happens inside the call itself.
        return f"{type(exc).__name__}: {exc}"
    return outcome[0]


def _prompt_twice(
    input_fn: Callable[[str], str],
    print_fn: Callable[..., None],
    text: str,
) -> str:
    """Ask, and on a blank answer ask once more. Empty when both answers are blank.

    A stray Enter should not cost the reader the whole wizard, and giving up after the second
    blank stops a scripted or piped stdin from looping forever. Both places that need a model id
    share this so the retry rule lives in exactly one place.
    """
    answer = _prompt(input_fn, print_fn, text)
    if answer:
        return answer
    print_fn("That was blank. One more try:")
    return _prompt(input_fn, print_fn, text)


def _reasoning_interview(
    input_fn: Callable[[str], str],
    print_fn: Callable[..., None],
    probe: HardwareProbe,
    *,
    security_required: bool,
    cloud_keys: dict[str, str],
) -> dict[str, str]:
    """Ask yes or no, then provider, then model. Returns the keys to write.

    Always returns `RECALL_REASONING`, so that switched off and never configured stay
    distinguishable in the file. `cloud_keys` may gain a provider key the user supplies here,
    which is why it is taken as a mutable dict rather than a mapping.
    """
    off = {"RECALL_REASONING": "0"}
    if not _ask_yes_no(
        input_fn, print_fn, "Enable the optional reasoning arm?", default=False
    ):
        return off

    provider = _choose(
        input_fn,
        print_fn,
        "Choose the provider for the reasoning arm:",
        reasoning_provider_choices(probe, security_required=security_required),
        sole_note=(
            "Reasoning provider: local endpoint, the only option while data security is "
            "required. Cloud providers stay withheld until that answer changes."
        ),
    )

    if provider.value == LOCAL_PROVIDER:
        base_url = (
            _prompt(
                input_fn,
                print_fn,
                f"Base URL for the local endpoint [{LOCAL_BASE_URL_DEFAULT}]: ",
            )
            or LOCAL_BASE_URL_DEFAULT
        )
        api_key = LOCAL_API_KEY
        model = _prompt_twice(input_fn, print_fn, "Model id: ")
    else:
        base_url = provider.value
        key_name = (
            "OPENROUTER_API_KEY" if base_url == OPENROUTER_BASE_URL else "OPENAI_API_KEY"
        )
        api_key = cloud_keys.get(key_name, "")
        if not api_key:
            api_key = _prompt_twice(input_fn, print_fn, f"{key_name}: ")
            if not api_key:
                print_fn("Reasoning arm skipped: no API key was given.")
                return off
            cloud_keys[key_name] = api_key
        model_choice = _choose(
            input_fn,
            print_fn,
            "Choose the model for the reasoning arm:",
            reasoning_model_choices(base_url),
        )
        model = model_choice.value
        if model == MANUAL_MODEL:
            model = _prompt_twice(input_fn, print_fn, "Model id: ")

    if not model:
        print_fn("Reasoning arm skipped: no model id was given.")
        return off

    failure = probe_reasoning_model(base_url=base_url, api_key=api_key, model=model)
    if failure is None:
        print_fn(f"Reasoning arm verified against {model}.")
    else:
        print_fn(f"Could not reach {model}: {failure}")
        print_fn("Writing the settings anyway. Correct them in .env and try again.")

    return {
        "RECALL_REASONING": "1",
        "RECALL_REASONING_MODEL": model,
        "RECALL_REASONING_BASE_URL": base_url,
        "RECALL_REASONING_API_KEY": api_key,
    }


def _prompt(
    input_fn: Callable[[str], str],
    print_fn: Callable[..., None],
    text: str,
) -> str:
    print_fn(text, end="")
    return input_fn("").strip()


def _ask_yes_no(
    input_fn: Callable[[str], str],
    print_fn: Callable[..., None],
    text: str,
    *,
    default: bool | None = None,
) -> bool:
    suffix = " [y/n] "
    if default is True:
        suffix = " [Y/n] "
    elif default is False:
        suffix = " [y/N] "
    while True:
        raw = _prompt(input_fn, print_fn, f"{text}{suffix}").lower()
        if not raw and default is not None:
            return default
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print_fn("Please answer yes or no.")


def _choose(
    input_fn: Callable[[str], str],
    print_fn: Callable[..., None],
    title: str,
    choices: Sequence[Choice],
    *,
    sole_note: str | None = None,
) -> Choice:
    if not choices:
        raise ValueError(f"no choices available for {title}")
    if not choices[0].available:
        # The fallback below returns choices[0] when someone picks an option this machine cannot
        # run, so the first entry has to be the runnable baseline. Every builder puts it there.
        # Failing here is far better than the alternative, which is writing an unrunnable value
        # into .env under a message announcing it was kept for safety.
        raise ValueError(f"the first choice for {title} must be runnable")
    if len(choices) == 1 and sole_note is not None:
        # A menu of one is not a choice. The hardware probe already decided this, so asking the
        # reader to type `1` costs a keystroke and tells them nothing. Say what was selected and
        # why nothing else was offered, which is how the entailment judge already reports its
        # own absence a few lines below. Only call sites that pass a note opt into this, so
        # nothing else in the wizard changes shape without saying so.
        print_fn(sole_note)
        return choices[0]
    print_fn(title)
    for i, choice in enumerate(choices, 1):
        suffix = "" if choice.available else "  (not installed yet)"
        print_fn(f"  {i}. {choice.label}: {choice.description}{suffix}")
    while True:
        raw = _prompt(input_fn, print_fn, "Select a number: ")
        try:
            idx = int(raw)
        except ValueError:
            idx = 0
        if 1 <= idx <= len(choices):
            picked = choices[idx - 1]
            if picked.available:
                return picked
            # Unavailable options are offered so the reader can see the feature exists and ask
            # for it. The choice still cannot be written to .env: the module is absent, so it
            # would fail at query time, long after the person who could fix it walked away.
            # Falling back to choices[0] relies on the first entry being the always-runnable
            # baseline, which `reranker_choices` and `sparse_choices` both guarantee.
            print_fn(picked.unavailable_note)
            print_fn(f"Keeping {choices[0].label} for now.")
            return choices[0]
        print_fn(f"Choose one of 1..{len(choices)}.")


def _update_env_block(path: Path, values: dict[str, str]) -> None:
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    start = next((i for i, line in enumerate(lines) if line.strip() == SETUP_BEGIN), None)
    end = next((i for i, line in enumerate(lines) if line.strip() == SETUP_END), None)
    block = [SETUP_BEGIN]
    for key, value in values.items():
        block.append(f"{key}={_quote_env(value)}")
    block.append(SETUP_END)
    if start is not None and end is not None and end >= start:
        lines = [*lines[:start], *block, *lines[end + 1 :]]
    else:
        if lines and lines[-1].strip():
            lines.append("")
        lines.extend(block)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _update_markdown_block(path: Path, begin: str, end: str, content: str) -> None:
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    start = next((i for i, line in enumerate(lines) if line.strip() == begin), None)
    stop = next((i for i, line in enumerate(lines) if line.strip() == end), None)
    block = [begin, *content.splitlines(), end]
    if start is not None and stop is not None and stop >= start:
        lines = [*lines[:start], *block, *lines[stop + 1 :]]
    else:
        if lines and lines[-1].strip():
            lines.append("")
        lines.extend(block)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _claude_md_block() -> str:
    return (
        "## Using recall\n"
        "\n"
        "This project is indexed by recall. Call `recall_search` before proposing an idea, "
        "forming a hypothesis, or repeating past work. If a closed decision or falsified "
        "hypothesis surfaces, do not re-litigate it.\n"
        "\n"
        "- When `abstained` is true, no hit survived the trust gate (or `decision: abstain` from "
        "`recall_evidence`) — say you do not know instead of answering from degraded hits.\n"
        "- Use `recall_evidence` instead of `recall_search` when about to answer from memory "
        "rather than just consult it; cite only `chunk_id` values from its `items`.\n"
        "- Write new durable facts to `memory/`, one file per fact, indexed by "
        "`memory/MEMORY.md` (see that file for the format), then call `recall_index` on "
        "`memory/` so the new file and the updated index both become searchable.\n"
    )


def scaffold_claude_md(path: Path = DEFAULT_CLAUDE_MD_PATH) -> None:
    _update_markdown_block(path, CLAUDE_MD_BEGIN, CLAUDE_MD_END, _claude_md_block())


def _memory_md_starter() -> str:
    return (
        "# Memory index\n"
        "\n"
        "This file is the always-loaded index. Each fact below lives in its own file under "
        "`memory/`, with frontmatter:\n"
        "\n"
        "```markdown\n"
        "---\n"
        "name: <short-kebab-case-slug>\n"
        "description: <one-line summary, used to judge relevance>\n"
        "metadata:\n"
        "  type: user | feedback | project | reference\n"
        "---\n"
        "\n"
        "<the fact>\n"
        "```\n"
        "\n"
        "Add one line per fact here as you create them:\n"
        "\n"
        "`- [Title](file.md) — one-line hook`\n"
    )


def scaffold_memory_index(memory_dir: Path = DEFAULT_MEMORY_DIR) -> bool:
    memory_dir.mkdir(parents=True, exist_ok=True)
    index_path = memory_dir / "MEMORY.md"
    if index_path.exists():
        return False
    index_path.write_text(_memory_md_starter(), encoding="utf-8")
    return True


def index_memory_directory(
    *,
    dsn: str,
    embedder_name: str,
    memory_dir: Path = DEFAULT_MEMORY_DIR,
    env: dict[str, str] | None = None,
    print_fn: Callable[..., None] = print,
) -> None:
    if os.environ.get("RECALL_ENV", "development").lower() == "production":
        print_fn(
            f"Skipping auto-index: RECALL_ENV is production. Index {memory_dir} via your "
            "production build pipeline instead."
        )
        return
    try:
        from recall.index import Indexer, chunk_text
        from recall.store import DEFAULT_TABLE, DEFAULT_TENANT, PgVectorStore

        embedder = resolve_embedder(embedder_name, env=env)
        with PgVectorStore(
            dsn, dim=embedder.dim, table=DEFAULT_TABLE, tenant=DEFAULT_TENANT
        ) as store:
            store.check_schema()
            indexer = Indexer(store, embedder, chunker=chunk_text)
            stats = indexer.index_path(memory_dir, glob="**/*.md")
    except Exception as exc:  # best effort: scaffolded files must survive even if this fails
        print_fn(
            f"Could not auto-index {memory_dir}: {exc} — run "
            f"'python -m recall.cli index {memory_dir}' once the schema is applied for this "
            "embedder's dimension."
        )
        return
    print_fn(f"Indexed {stats.chunks} chunks from {stats.files} files in {memory_dir}")


def _quote_env(value: str) -> str:
    if value == "":
        return '""'
    if any(ch.isspace() for ch in value) or any(ch in value for ch in '#"'):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return value


def _resolve_output_path(raw: str | None, *, default: Path) -> Path:
    path = Path(raw).expanduser() if raw else default
    if path.exists() and path.is_dir():
        path = path / default.name
    return path.resolve()


def _looks_like_windows_path(raw: str) -> bool:
    return len(raw) >= 2 and raw[1] == ":" and raw[0].isalpha()


def _require_local_path(raw: str, *, label: str) -> Path:
    if _looks_like_windows_path(raw) and os.name != "nt":
        raise ValueError(
            f"{label} looks like a Windows host path ({raw!r}). Inside Docker, use the "
            "container path instead, for example /app/recall/eval/queries.json."
        )
    return Path(raw).expanduser().resolve()


def _require_local_output_path(raw: str | None, *, label: str, default: Path) -> Path:
    if raw and _looks_like_windows_path(raw) and os.name != "nt":
        raise ValueError(
            f"{label} looks like a Windows host path ({raw!r}). Inside Docker, use the "
            "container path instead, for example /app/recall/calibration.json."
        )
    return _resolve_output_path(raw, default=default)


def calibrate_from_files(
    *,
    dsn: str,
    embedder_name: str,
    queries_path: Path,
    corpus_dir: Path | None = None,
    out: Path | None = None,
    env: dict[str, str] | None = None,
) -> CalibrationResult:
    queries_path = _require_local_path(str(queries_path), label="Path to labeled queries JSON")
    corpus_dir = (
        _require_local_path(str(corpus_dir), label="Path to your corpus") if corpus_dir else None
    )
    out = _require_local_output_path(
        str(out) if out else None,
        label="Calibration output path",
        default=DEFAULT_CALIBRATION_PATH,
    )
    queries = json.loads(queries_path.read_text(encoding="utf-8"))
    labeled = [q for q in queries if isinstance(q, dict) and not q.get("trust")]
    if not labeled or not all("query" in q and "answerable" in q for q in labeled):
        raise ValueError(
            "queries file entries need 'query' and 'answerable' keys "
            "(see recall/eval/queries.json for the format)"
        )
    if not any(q["answerable"] for q in labeled) or not any(not q["answerable"] for q in labeled):
        raise ValueError(
            "queries file needs at least one answerable AND one unanswerable entry"
        )
    embedder = resolve_embedder(embedder_name, env=env)
    from recall.eval.calibrate import calibrate as run_calibration

    measured = run_calibration(
        dsn,
        embedder,
        corpus_dir=corpus_dir,
        queries_path=queries_path,
    )
    cal = from_samples(
        embedder.name,
        measured.answerable_max_cos,
        measured.unanswerable_max_cos,
    )
    path = save(cal, out)
    return CalibrationResult(path=path, calibration=cal, report=measured)


def run_setup_wizard(
    *,
    dsn: str,
    migration_dsn: str | None = None,
    table: str | None = None,
    env_path: Path = DEFAULT_ENV_PATH,
    claude_md_path: Path = DEFAULT_CLAUDE_MD_PATH,
    memory_dir: Path = DEFAULT_MEMORY_DIR,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[..., None] = print,
) -> dict[str, str]:
    probe = probe_hardware()
    print_fn(
        "Hardware check: "
        f"cpu={probe.cpu_count or 'unknown'}, "
        f"gpu={probe.gpu or 'none'}, "
        f"free_disk={probe.free_bytes // (1024 * 1024 * 1024)} GB, "
        f"internet={'yes' if probe.internet else 'no'}"
    )
    security_required = _ask_yes_no(
        input_fn, print_fn, "Is data security necessary for this installation?", default=True
    )
    cloud_keys: dict[str, str] = {}
    if not security_required:
        print_fn("Step 1b, optional API keys. Leave blank to skip.")
        for key in ("VOYAGE_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY"):
            value = _prompt(input_fn, print_fn, f"{key}: ")
            if value:
                cloud_keys[key] = value
    embedders = embedder_choices(
        probe, security_required=security_required, cloud_keys=cloud_keys
    )
    while True:
        embedder = _choose(
            input_fn,
            print_fn,
            "Choose the embedder you want to use:",
            embedders,
            sole_note=(
                "Embedder: hashing, the only one this machine can use. It needs no download and "
                "no network, and it retrieves noticeably worse than a real model. Install one "
                'with pip install "recall-rag[fastembed]", or set an API key above, then rerun '
                "setup."
            ),
        )
        _prepare_schema_for_embedder(
            dsn=dsn,
            migration_dsn=migration_dsn,
            table=table,
            embedder=embedder,
            print_fn=print_fn,
        )
        break
    rerankers = reranker_choices(probe, security_required=security_required)
    reranker = _choose(
        input_fn,
        print_fn,
        "Choose whether to enable reranking:",
        rerankers,
    )
    sparse_backend = _choose(
        input_fn,
        print_fn,
        "Choose the sparse retrieval backend:",
        sparse_choices(probe),
    )
    entailment = None
    entailment_options = entailment_choices(probe)
    if entailment_options:
        if _ask_yes_no(
            input_fn,
            print_fn,
            "Enable the optional entailment judge before the final answer?",
            default=False,
        ):
            entailment = _choose(
                input_fn,
                print_fn,
                "Choose the entailment judge model:",
                entailment_options,
            )
    else:
        print_fn(
            "Entailment judge is unavailable on this machine. It needs sentence-transformers "
            "and enough internet and disk to download a model. The extra is "
            'pip install "recall-rag[entail]". Rerun setup and choose it again once it can run.'
        )

    reasoning_values = _reasoning_interview(
        input_fn,
        print_fn,
        probe,
        security_required=security_required,
        cloud_keys=cloud_keys,
    )

    scaffold_requested = _ask_yes_no(
        input_fn,
        print_fn,
        "Scaffold CLAUDE.md and a memory/ directory for this project?",
        default=True,
    )

    # Deferred to right before each return, after `.env` is written: `scaffold_claude_md` and
    # `index_memory_directory` can resolve/download an embedder model and open a DB connection,
    # and running that BEFORE the answers just gathered are persisted means an interruption
    # during a long model download loses the whole completed interview, not just the scaffold.
    def _run_scaffold() -> None:
        try:
            scaffold_claude_md(claude_md_path)
            print_fn(f"Updated {claude_md_path}")
            if scaffold_memory_index(memory_dir):
                print_fn(f"Wrote {memory_dir / 'MEMORY.md'}")
            else:
                print_fn(f"{memory_dir / 'MEMORY.md'} already exists, left unchanged.")
            index_memory_directory(
                dsn=dsn,
                embedder_name=embedder.value,
                memory_dir=memory_dir,
                env=cloud_keys,
                print_fn=print_fn,
            )
        except Exception as exc:
            print_fn(f"Could not scaffold CLAUDE.md/memory: {exc}")

    values: dict[str, str] = {
        "RECALL_DSN": dsn,
        "RECALL_SECURITY_REQUIRED": "1" if security_required else "0",
        "RECALL_EMBEDDER": embedder.value,
        "RECALL_SPARSE": "fts",
        "RECALL_ENTAILMENT": "0",
    }
    if reranker.value == "RECALL_RERANK=1":
        values["RECALL_RERANK"] = "1"
    elif reranker.value.startswith("RECALL_RERANK=1;"):
        values["RECALL_RERANK"] = "1"
        for chunk in reranker.value.split(";")[1:]:
            key, _, value = chunk.partition("=")
            if key and value:
                values[key] = value
    else:
        values["RECALL_RERANK"] = "0"

    if sparse_backend.value:
        key, _, value = sparse_backend.value.partition("=")
        if key and value:
            values[key] = value

    if entailment is not None and entailment.value:
        parts = entailment.value.split(";")
        for chunk in parts:
            key, _, value = chunk.partition("=")
            if key and value:
                values[key] = value

    values.update(reasoning_values)
    values.update(cloud_keys)

    if _ask_yes_no(
        input_fn,
        print_fn,
        "Do you want to calibrate the threshold now against a labeled query file?",
        default=False,
    ):
        queries_raw = _prompt(
            input_fn, print_fn, "Path to labeled queries JSON [leave blank to skip]: "
        )
        corpus_raw = _prompt(
            input_fn, print_fn, "Path to your corpus [leave blank to skip]: "
        )
        if not queries_raw or not corpus_raw:
            print_fn(
                "Calibration skipped. Run recall calibrate later with a labeled query file "
                "when you have one."
            )
            _update_env_block(env_path, values)
            print_fn(f"Wrote {env_path}")
            if scaffold_requested:
                _run_scaffold()
            return values
        queries = _require_local_path(queries_raw, label="Path to labeled queries JSON")
        corpus = _require_local_path(corpus_raw, label="Path to your corpus")
        out_raw = _prompt(input_fn, print_fn, "Calibration output path [calibration.json]: ")
        try:
            result = calibrate_from_files(
                dsn=dsn,
                embedder_name=embedder.value,
                queries_path=queries,
                corpus_dir=corpus,
                out=_require_local_output_path(
                    out_raw or None,
                    label="Calibration output path",
                    default=DEFAULT_CALIBRATION_PATH,
                ),
                env=cloud_keys,
            )
        except ValueError as exc:
            print_fn(str(exc))
            # This path never wrote `.env` even before this function grew a scaffold step — a
            # bad calibration path aborts the whole wizard. But scaffolding was requested
            # independently of calibration succeeding, so still attempt it best-effort on the
            # way out rather than silently dropping a completed part of the interview.
            if scaffold_requested:
                _run_scaffold()
            raise SystemExit(2) from exc
        values["RECALL_CALIBRATION"] = str(result.path)
        print_fn(
            f"Calibration saved to {result.path}. Threshold={result.calibration.threshold:.3f}"
        )
    else:
        print_fn(
            "Calibration skipped. Run recall calibrate later with a labeled query file "
            "when you have one."
        )

    _update_env_block(env_path, values)
    print_fn(f"Wrote {env_path}")
    if scaffold_requested:
        _run_scaffold()
    return values
