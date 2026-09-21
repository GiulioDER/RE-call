"""Run the frozen C6 versus C7 live qualification without launching official AML.

This script is deliberately limited to the two preregistered Hosted variants. It ingests the
frozen coding corpus into isolated users, compares the live top 100 rankings, executes the three
isolated routing probes, and reads the C7 table only to audit specialist identity and vector
distinctness. It never invokes the Agent Memory Leaderboard harness.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import statistics
import unicodedata
from typing import Any, Callable, Mapping, Protocol
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from uuid import uuid4

from recall_aml.identity import specialist_tenant, tenant_for
from recall_aml.specialists import route_query


QUALIFICATION = "2026-09-20-aml-routed-specialist-corpus"
C6_VARIANT = "C6_code4_exact_bm25"
C7_VARIANT = "C7_routed_specialists"
MANIFEST_SHA256 = "58055df1828b2c1e51bc3c7f9f82e916145c67aa58332f22ce1b86b2d849b814"
EXPECTED_SESSIONS = 196
EXPECTED_WINDOWS = 1_220
EXPECTED_TASKS = 34
EXPECTED_TABLE = "recall_aml_routed_specialists_chunks"
CONTEXT_PROFILE = "voyage-context-4-v1"
PNG_DATA_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)
_TEXT_FIELDS = ("content", "tool_result", "tool_input")
_NOISE = re.compile(r"[*_`~]+")
_JOINERS = re.compile(r"[-‐-―/\\]+")
_SPACE = re.compile(r"\s+")
_FORBIDDEN_RESPONSE_KEY = re.compile(
    r"(?:embedding|vector|cosine|cross.?model|model.?scores?|averag(?:e|ed|ing))",
    re.IGNORECASE,
)


class QualificationRefusal(RuntimeError):
    """A safe, machine readable refusal raised before an invalid live run."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class Call:
    status: int
    payload: dict[str, Any]
    headers: dict[str, str]


class HttpClient:
    def __init__(self, base_url: str, api_key: str, *, timeout: float = 300.0) -> None:
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout

    def call(self, path: str, payload: dict[str, Any] | None = None) -> Call:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            self.base_url + path,
            data=body,
            method="GET" if payload is None else "POST",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urlopen(request, timeout=self._timeout) as response:  # noqa: S310
                raw = response.read().decode("utf-8")
                return Call(
                    response.status,
                    json.loads(raw),
                    {key.lower(): value for key, value in response.headers.items()},
                )
        except HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            try:
                payload_out = json.loads(raw)
            except json.JSONDecodeError:
                payload_out = {"error": "non_json_response"}
            return Call(
                exc.code,
                payload_out,
                {key.lower(): value for key, value in exc.headers.items()},
            )


class ClientProtocol(Protocol):
    def call(self, path: str, payload: dict[str, Any] | None = None) -> Call: ...


@dataclass(frozen=True)
class FrozenCorpus:
    corpus_root: Path
    sessions: dict[str, str]
    rendered: dict[str, str]
    tasks: tuple[tuple[str, str, frozenset[str]], ...]


def _normalise(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.lower())
    folded = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return _SPACE.sub(" ", _JOINERS.sub(" ", _NOISE.sub("", folded)))


def _safe_manifest_path(root: Path, relative: str) -> Path:
    if (
        not relative.strip()
        or PurePosixPath(relative).is_absolute()
        or PureWindowsPath(relative).is_absolute()
        or PureWindowsPath(relative).drive
        or "\\" in relative
    ):
        raise QualificationRefusal("corpus_path_escape")
    resolved_root = root.resolve()
    candidate = (resolved_root / relative).resolve()
    if candidate != resolved_root and resolved_root not in candidate.parents:
        raise QualificationRefusal("corpus_path_escape")
    return candidate


def _render_transcript(path: Path) -> str:
    parts: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            parts.append(line)
            continue
        for field in _TEXT_FIELDS:
            value = event.get(field)
            if isinstance(value, str):
                parts.append(value)
            elif value is not None:
                parts.append(json.dumps(value, ensure_ascii=False))
    rendered = _normalise(" ".join(parts)).strip()
    if not rendered:
        raise QualificationRefusal("empty_rendered_session")
    return rendered


def _window_count(text: str) -> int:
    words = text.split()
    count = 0
    for start in range(0, max(1, len(words)), 120):
        count += 1
        if start + 160 >= len(words):
            break
    return count


def load_frozen_corpus(amb_root: Path) -> FrozenCorpus:
    corpus_root = amb_root / "corpus"
    manifest_path = corpus_root / "manifest.json"
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != MANIFEST_SHA256:
        raise QualificationRefusal("wrong_manifest_sha256")
    raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sessions = raw_manifest.get("sessions")
    if not isinstance(sessions, dict) or len(sessions) != EXPECTED_SESSIONS:
        raise QualificationRefusal("wrong_session_roster")
    verified: dict[str, str] = {}
    rendered: dict[str, str] = {}
    for relative, expected_hash in sorted(sessions.items()):
        if not isinstance(relative, str) or not isinstance(expected_hash, str):
            raise QualificationRefusal("invalid_manifest_entry")
        path = _safe_manifest_path(corpus_root, relative)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            raise QualificationRefusal("corpus_hash_mismatch")
        verified[relative] = expected_hash
        rendered[relative] = _render_transcript(path)
    if sum(_window_count(text) for text in rendered.values()) != EXPECTED_WINDOWS:
        raise QualificationRefusal("wrong_window_count")

    tasks: list[tuple[str, str, frozenset[str]]] = []
    for task_path in sorted((amb_root / "tasks").glob("*/task.json")):
        task = json.loads(task_path.read_text(encoding="utf-8"))
        task_id = str(task.get("task_id", ""))
        prompt = str(task.get("prompt", "")).strip()
        gold = frozenset(
            relative
            for relative in verified
            if relative.startswith(f"sessions/{task_id}/")
        )
        if gold:
            if not prompt:
                raise QualificationRefusal("empty_task_prompt")
            tasks.append((task_id, prompt, gold))
    if len(tasks) != EXPECTED_TASKS:
        raise QualificationRefusal("wrong_task_roster")
    return FrozenCorpus(corpus_root, verified, rendered, tuple(tasks))


def resolve_credentials(
    environment: Mapping[str, str],
    *,
    c6_name: str,
    c7_name: str,
    shared_name: str,
) -> tuple[str, str, dict[str, object]]:
    c6_specific = environment.get(c6_name, "")
    c7_specific = environment.get(c7_name, "")
    shared = environment.get(shared_name, "")
    c6 = c6_specific or shared
    c7 = c7_specific or shared
    if not c6 or not c7:
        raise QualificationRefusal("missing_endpoint_credentials")
    if c6_specific and c7_specific and c6 == c7:
        raise QualificationRefusal("dedicated_keys_not_distinct")
    return c6, c7, {
        "c6_credential": "dedicated" if c6_specific else "shared_fallback",
        "c7_credential": "dedicated" if c7_specific else "shared_fallback",
        "keys_distinct": c6 != c7,
    }


def validate_endpoint_versions(c6: Call, c7: Call) -> dict[str, bool]:
    if c6.status != 200 or c6.payload.get("variant") != C6_VARIANT:
        raise QualificationRefusal("c6_endpoint_is_not_exact_variant")
    if c7.status != 200 or c7.payload.get("variant") != C7_VARIANT:
        raise QualificationRefusal("c7_endpoint_is_not_exact_variant")
    c6_shape = {
        "embedding_profile": "voyage-code-4-v1",
        "exact_dense": True,
        "ordering_profile": "source-session-c-collation-segment-v1",
        "window_renderer_profile": "message-content-only-v1",
        "embedding_call_lock": True,
        "embedding_cache": True,
    }
    c7_shape = {
        **c6_shape,
        "context_embedding_profile": CONTEXT_PROFILE,
        "multimodal_embedding_profile": "voyage-multimodal-3.5-v2",
        "multimodal_embedding_model": "voyage-multimodal-3.5",
        "specialist_router_profile": "conservative-specialist-router-v1",
        "specialist_fusion_profile": "routed-rank-fusion-v1",
        "rrf_constant": 60,
    }
    return {
        "c6_frozen_shape": all(c6.payload.get(key) == value for key, value in c6_shape.items()),
        "c7_frozen_shape": all(c7.payload.get(key) == value for key, value in c7_shape.items()),
    }


def _payload_digest(source: str, text: str, metadata: object) -> str:
    encoded = json.dumps(
        {"source": source, "text": text, "metadata": metadata},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def audit_specialist_namespaces(dsn: str, user_id: str) -> dict[str, object]:
    """Compare C7 logical payloads and vectors without returning any vector value."""
    try:
        import psycopg
    except ImportError as exc:  # pragma: no cover, deployment dependency
        raise QualificationRefusal("psycopg_unavailable") from exc

    primary_tenant = tenant_for(user_id)
    context_tenant = specialist_tenant(primary_tenant, CONTEXT_PROFILE)

    def read_rows(tenant: str) -> dict[str, tuple[str, str]]:
        with psycopg.connect(dsn, autocommit=True, connect_timeout=10) as connection:
            connection.execute(
                "SELECT set_config('recall.tenant_id', %s, false)", (tenant,)
            )
            rows = connection.execute(
                f"SELECT id, source, text, metadata, embedding::text "  # noqa: S608
                f"FROM {EXPECTED_TABLE} WHERE tenant_id = %s ORDER BY id",
                (tenant,),
            ).fetchall()
        return {
            str(chunk_id): (
                _payload_digest(str(source), str(text), metadata),
                hashlib.sha256(str(vector).encode("ascii")).hexdigest(),
            )
            for chunk_id, source, text, metadata, vector in rows
        }

    primary = read_rows(primary_tenant)
    context = read_rows(context_tenant)
    shared = sorted(set(primary) & set(context))
    payload_matches = sum(primary[item][0] == context[item][0] for item in shared)
    vector_differences = sum(primary[item][1] != context[item][1] for item in shared)
    return {
        "primary_count": len(primary),
        "context_count": len(context),
        "shared_count": len(shared),
        "same_ids": set(primary) == set(context),
        "payload_match_count": payload_matches,
        "different_vector_count": vector_differences,
        "different_vector_fraction": vector_differences / len(shared) if shared else 0.0,
    }


def _require_ok(call: Call, code: str, *, variant: str | None = None) -> dict[str, Any]:
    if call.status != 200:
        raise QualificationRefusal(code)
    if variant is not None and call.headers.get("x-recall-variant") != variant:
        raise QualificationRefusal(code + "_variant_header")
    return call.payload


def _add_corpus(client: ClientProtocol, corpus: FrozenCorpus, user_id: str) -> int:
    def add_one(item: tuple[str, str]) -> int:
        relative, text = item
        request_id = "qualification-" + hashlib.sha256(relative.encode()).hexdigest()
        payload = _require_ok(
            client.call(
                "/v1/add",
                {
                    "request_id": request_id,
                    "messages": [{"role": "user", "content": text}],
                    "user_id": user_id,
                    "session_id": relative,
                },
            ),
            "corpus_add_failed",
        )
        return int(payload.get("raw_count", -1))

    with ThreadPoolExecutor(max_workers=3) as pool:
        counts = list(pool.map(add_one, corpus.rendered.items()))
    return sum(counts)


def _search(client: ClientProtocol, user_id: str, query: Any, top_k: int, variant: str) -> Call:
    call = client.call(
        "/v1/search", {"query": query, "user_id": user_id, "top_k": top_k}
    )
    _require_ok(call, "search_failed", variant=variant)
    return call


def forbidden_response_keys(payload: object) -> list[str]:
    found: set[str] = set()

    def visit(value: object) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if _FORBIDDEN_RESPONSE_KEY.search(str(key)):
                    found.add(str(key))
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(payload)
    return sorted(found)


def compare_coding_rankings(
    corpus: FrozenCorpus,
    c6: ClientProtocol,
    c7: ClientProtocol,
    c6_user: str,
    c7_user: str,
) -> tuple[dict[str, object], list[dict[str, Any]]]:
    exact = 0
    recall_at_10 = 0
    reciprocal_ranks: list[float] = []
    mismatches: list[str] = []
    responses: list[dict[str, Any]] = []
    for task_id, prompt, gold in corpus.tasks:
        c6_call = _search(c6, c6_user, prompt, 100, C6_VARIANT)
        c7_call = _search(c7, c7_user, prompt, 100, C7_VARIANT)
        responses.extend((c6_call.payload, c7_call.payload))
        c6_data = c6_call.payload.get("data", [])
        c7_data = c7_call.payload.get("data", [])
        c6_ids = [str(item.get("id", "")) for item in c6_data]
        c7_ids = [str(item.get("id", "")) for item in c7_data]
        is_exact = len(c6_ids) == len(c7_ids) == 100 and c6_ids == c7_ids
        exact += int(is_exact)
        if not is_exact:
            mismatches.append(task_id)
        first = next(
            (
                rank
                for rank, item in enumerate(c7_data, start=1)
                if str(item.get("session_id", "")) in gold
            ),
            None,
        )
        recall_at_10 += int(first is not None and first <= 10)
        reciprocal_ranks.append(0.0 if first is None else 1.0 / first)
    return (
        {
            "queries": len(corpus.tasks),
            "exact_top100": exact,
            "mismatch_task_ids": mismatches,
            "source_recall_at_10_count": recall_at_10,
            "mean_reciprocal_rank": statistics.fmean(reciprocal_ranks),
        },
        responses,
    )


def _probe_add_search(
    client: ClientProtocol,
    *,
    user_id: str,
    session_id: str,
    content: Any,
    query: Any,
) -> tuple[Call, Call]:
    add = client.call(
        "/v1/add",
        {
            "request_id": "qualification-" + uuid4().hex,
            "messages": [{"role": "user", "content": content}],
            "user_id": user_id,
            "session_id": session_id,
        },
    )
    _require_ok(add, "probe_add_failed")
    search = _search(client, user_id, query, 3, C7_VARIANT)
    return add, search


def _probe_rank(call: Call, session_id: str) -> int | None:
    return next(
        (
            rank
            for rank, item in enumerate(call.payload.get("data", []), start=1)
            if item.get("session_id") == session_id
        ),
        None,
    )


def _gate(number: int, passed: bool, **detail: object) -> dict[str, object]:
    return {"gate": number, "passed": bool(passed), "detail": detail}


def _as_float(value: object) -> float:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0


def run_qualification(
    *,
    amb_root: Path,
    c6_url: str,
    c7_url: str,
    c6_key: str,
    c7_key: str,
    database_url: str,
    client_factory: Callable[[str, str], ClientProtocol] = HttpClient,
    namespace_auditor: Callable[[str, str], dict[str, object]] = audit_specialist_namespaces,
) -> dict[str, object]:
    if c6_url.rstrip("/") == c7_url.rstrip("/"):
        raise QualificationRefusal("c6_and_c7_endpoints_must_differ")
    corpus = load_frozen_corpus(amb_root)
    if any(route_query(prompt) != "code" for _, prompt, _ in corpus.tasks):
        raise QualificationRefusal("task_router_preflight_failed")

    c6 = client_factory(c6_url, c6_key)
    c7 = client_factory(c7_url, c7_key)
    c6_version = c6.call("/version")
    c7_version = c7.call("/version")
    version_checks = validate_endpoint_versions(c6_version, c7_version)
    c6_health = c6.call("/health")
    c7_health = c7.call("/health")

    run_id = uuid4().hex
    users = {
        "c6_corpus": f"c7-qualification-c6-{run_id}",
        "c7_corpus": f"c7-qualification-c7-{run_id}",
        "code": f"c7-qualification-code-{run_id}",
        "context": f"c7-qualification-context-{run_id}",
        "visual": f"c7-qualification-visual-{run_id}",
    }
    cleanup: dict[str, int] = {}
    responses: list[dict[str, Any]] = []
    try:
        c6_written = _add_corpus(c6, corpus, users["c6_corpus"])
        c7_written = _add_corpus(c7, corpus, users["c7_corpus"])
        status_call = c7.call("/v1/corpus/status", {"user_id": users["c7_corpus"]})
        status = _require_ok(status_call, "c7_corpus_status_failed")
        namespace = namespace_auditor(database_url, users["c7_corpus"])
        coding, coding_responses = compare_coding_rankings(
            corpus, c6, c7, users["c6_corpus"], users["c7_corpus"]
        )
        responses.extend(coding_responses)

        code_session = "qualification/probes/code"
        code_query = "Fix parser.py because pytest reports the C7ProbeError traceback"
        _, code_search = _probe_add_search(
            c7,
            user_id=users["code"],
            session_id=code_session,
            content="parser.py requires stable ordering to fix pytest C7ProbeError.",
            query=code_query,
        )
        context_session = "qualification/probes/context"
        context_query = "What was decided in yesterday's meeting about the launch codename?"
        _, context_search = _probe_add_search(
            c7,
            user_id=users["context"],
            session_id=context_session,
            content=(
                "During yesterday's meeting we decided the launch codename is "
                "CONTEXT-C7-CEDAR-531."
            ),
            query=context_query,
        )
        visual_session = "qualification/probes/visual"
        visual_content = [
            {"type": "text", "text": "The screenshot marker is VISUAL-C7-ORANGE-742."},
            {"type": "image_url", "image_url": {"url": PNG_DATA_URL}},
        ]
        visual_query = "What does this screenshot show about the orange qualification marker?"
        _, visual_search = _probe_add_search(
            c7,
            user_id=users["visual"],
            session_id=visual_session,
            content=visual_content,
            query=visual_query,
        )
        responses.extend((code_search.payload, context_search.payload, visual_search.payload))

        code_rank = _probe_rank(code_search, code_session)
        context_rank = _probe_rank(context_search, context_session)
        visual_rank = _probe_rank(visual_search, visual_session)
        visual_match = next(
            (
                item.get("content") == visual_content
                for item in visual_search.payload.get("data", [])
                if item.get("session_id") == visual_session
            ),
            False,
        )
        forbidden = sorted(
            {key for response in responses for key in forbidden_response_keys(response)}
        )
        c7_version_ok = version_checks["c7_frozen_shape"]
        gates = [
            _gate(
                1,
                c6_health.status == 200
                and c6_health.payload.get("status") == "ready"
                and c7_health.status == 200
                and c7_health.payload.get("status") == "ready"
                and c7_version_ok,
                c6_health=c6_health.payload.get("status"),
                c7_health=c7_health.payload.get("status"),
            ),
            _gate(2, version_checks["c6_frozen_shape"] and c7_version_ok, **version_checks),
            _gate(
                3,
                c7_written == EXPECTED_WINDOWS
                and int(status.get("raw_chunk_count", -1)) == EXPECTED_WINDOWS
                and int(status.get("source_session_count", -1)) == EXPECTED_SESSIONS
                and namespace.get("primary_count") == EXPECTED_WINDOWS
                and namespace.get("context_count") == EXPECTED_WINDOWS
                and namespace.get("shared_count") == EXPECTED_WINDOWS
                and namespace.get("same_ids") is True
                and namespace.get("payload_match_count") == EXPECTED_WINDOWS,
                c6_written=c6_written,
                c7_written=c7_written,
                primary_count=namespace.get("primary_count"),
                context_count=namespace.get("context_count"),
                shared_count=namespace.get("shared_count"),
                payload_match_count=namespace.get("payload_match_count"),
            ),
            _gate(
                4,
                _as_float(namespace.get("different_vector_fraction", 0.0)) >= 0.99,
                different_vector_count=namespace.get("different_vector_count"),
                shared_count=namespace.get("shared_count"),
                different_vector_fraction=namespace.get("different_vector_fraction"),
            ),
            _gate(
                5,
                all(route_query(prompt) == "code" for _, prompt, _ in corpus.tasks)
                and route_query(code_query) == "code"
                and code_rank is not None
                and code_rank <= 3,
                coding_task_routes=EXPECTED_TASKS,
                code_probe_rank=code_rank,
            ),
            _gate(
                6,
                coding["queries"] == EXPECTED_TASKS
                and coding["exact_top100"] == EXPECTED_TASKS
                and coding["source_recall_at_10_count"] == EXPECTED_TASKS
                and _as_float(coding["mean_reciprocal_rank"]) >= 0.8464,
                **coding,
            ),
            _gate(
                7,
                route_query(context_query) == "context"
                and context_rank is not None
                and context_rank <= 3,
                route=route_query(context_query),
                planted_rank=context_rank,
            ),
            _gate(
                8,
                route_query(visual_query) == "multimodal"
                and visual_rank is not None
                and visual_rank <= 3
                and visual_match,
                route=route_query(visual_query),
                planted_rank=visual_rank,
                ordered_parts_byte_exact=visual_match,
            ),
            _gate(9, not forbidden, forbidden_response_keys=forbidden),
        ]
        return {
            "schema_version": 1,
            "qualification": QUALIFICATION,
            "measured_at": datetime.now(UTC).isoformat(),
            "passed": all(bool(gate["passed"]) for gate in gates),
            "official_aml_launched": False,
            "frozen_inputs": {
                "manifest_sha256": MANIFEST_SHA256,
                "sessions": EXPECTED_SESSIONS,
                "windows": EXPECTED_WINDOWS,
                "tasks": EXPECTED_TASKS,
                "add_workers": 3,
            },
            "endpoint_variants": {"c6": C6_VARIANT, "c7": C7_VARIANT},
            "gates": gates,
            "cleanup_statuses": cleanup,
        }
    finally:
        for label, user in users.items():
            client = c6 if label == "c6_corpus" else c7
            try:
                cleanup[label] = client.call("/v1/delete", {"user_id": user}).status
            except Exception:  # noqa: BLE001, cleanup must not mask qualification evidence
                cleanup[label] = 0


def _emit(result: dict[str, object], out: Path | None) -> None:
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--amb-root", type=Path, required=True)
    parser.add_argument("--c6-url", required=True)
    parser.add_argument("--c7-url", required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--c6-api-key-env", default="RECALL_AML_C6_API_KEY")
    parser.add_argument("--c7-api-key-env", default="RECALL_AML_C7_API_KEY")
    parser.add_argument("--shared-api-key-env", default="RECALL_AML_API_KEY")
    parser.add_argument("--c7-database-url-env", default="RECALL_AML_DATABASE_URL")
    parser.add_argument("--execute-live-qualification", action="store_true")
    args = parser.parse_args()

    if not args.execute_live_qualification:
        raise SystemExit("refused: pass --execute-live-qualification for the preregistered live run")
    try:
        c6_key, c7_key, credential_summary = resolve_credentials(
            os.environ,
            c6_name=args.c6_api_key_env,
            c7_name=args.c7_api_key_env,
            shared_name=args.shared_api_key_env,
        )
        database_url = os.environ.get(args.c7_database_url_env, "")
        if not database_url:
            raise QualificationRefusal("missing_c7_database_url")
        result = run_qualification(
            amb_root=args.amb_root,
            c6_url=args.c6_url,
            c7_url=args.c7_url,
            c6_key=c6_key,
            c7_key=c7_key,
            database_url=database_url,
        )
        result["credentials"] = credential_summary
    except QualificationRefusal as exc:
        result = {
            "schema_version": 1,
            "qualification": QUALIFICATION,
            "passed": False,
            "official_aml_launched": False,
            "refused": True,
            "reason": exc.code,
        }
    except Exception as exc:  # noqa: BLE001, secret safe terminal artifact
        result = {
            "schema_version": 1,
            "qualification": QUALIFICATION,
            "passed": False,
            "official_aml_launched": False,
            "refused": False,
            "error_class": type(exc).__name__,
        }
    _emit(result, args.out)
    if not result.get("passed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
