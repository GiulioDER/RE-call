"""Stage 1 of the MM-1/MM-3 pre-registration: MemEye retrieval and delivery under four Search arms.

Pre-registration: ``docs/preregistrations/2026-09-25-aml-c9-multimodal-scope-and-dates.md``.

Ingests the eight public MemEye MCQ scenarios once through arm B (served C9 behaviour), then
Searches every question and option rotation under B, B' (B again), P (``preserve``) and D
(``dual``), each served by its own process over the same table
(``scripts/aml_mm_scope_vps3.sh``). Every response is stored compactly: image parts are replaced
by the SHA-256 of their decoded bytes, which maps back to the cached source image, so Stage 2 can
rebuild the exact reader input without keeping megabytes of base64 per row. Each scenario's
tenant is deleted and verified empty before the next begins.

VPS3 only. No answer model runs here. The one paid call is C9's Add-time compile on text-only
rounds (DeepSeek V4.1 Flash on this host); the run reads the OpenRouter balance first and between
scenarios, and stops below the pre-registered floor.

Usage::

    python scripts/aml_mm_scope_stage1.py fetch --cache-dir DIR
    python scripts/aml_mm_scope_stage1.py run --cache-dir DIR --out FILE.jsonl --run-id ID
"""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import threading
import time
from typing import Any, Callable
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

MEMEYE_REVISION = "f139f89d08bf0f66986153b5ff010cac24ba383d"
HF_BASE = "https://huggingface.co/datasets/MemEyeBench/MemEye/resolve"
SCENARIOS = (
    "Brand_Memory_Test",
    "Card_Playlog_Test",
    "Cartoon_Entertainment_Companion",
    "Home_Renovation_Interior_Design",
    "Multi-Scene_Visual_Case_Archive_Assistant",
    "Outdoor_Navigation_Route_Memory_Assistant",
    "Personal_Health_Dashboard_Assistant",
    "Social_Chat_Memory_Test",
)
ARM_PORTS = {"B": 18031, "B2": 18031, "P": 18032, "D": 18033}
ARM_ORDER = ("B", "P", "D", "B2")


@dataclass(frozen=True)
class Arms:
    """Which services a run Searches: MM-1's four arms by default, MM-4's two by ``--arm-config``.

    ``expect`` maps each arm to ``/version`` fields it must report before anything is sent; a dotted
    key reads a nested field (``image_text.leg``).
    """

    ports: dict[str, int]
    order: tuple[str, ...]
    ingest: str
    expect: dict[str, dict[str, Any]]


MM1_ARMS = Arms(ARM_PORTS, ARM_ORDER, "B", {"B": {"multimodal_scope": "route"},
                                            "P": {"multimodal_scope": "preserve"},
                                            "D": {"multimodal_scope": "dual"}})


def load_arms(path: Path | None) -> Arms:
    if path is None:
        return MM1_ARMS
    raw = json.loads(path.read_text(encoding="utf-8"))
    arms = Arms({str(k): int(v) for k, v in raw["ports"].items()}, tuple(raw["order"]), str(raw["ingest"]),
                {str(k): dict(v) for k, v in raw["expect"].items()})
    if set(arms.order) != set(arms.ports) or arms.ingest not in arms.ports:
        raise Stage1Error("arm config: order must name exactly the arms with ports, ingest among them")
    return arms


def _field(version: dict[str, Any], dotted: str) -> Any:
    value: Any = version
    for part in dotted.split("."):
        value = value.get(part) if isinstance(value, dict) else None
    return value


def check_versions(versions: dict[str, dict[str, Any]], arms: Arms) -> None:
    """Refuse a run unless every arm serves one commit and reports exactly its expected settings."""
    commits = {version["git_commit"] for version in versions.values()}
    if len(commits) != 1:
        raise Stage1Error("arms serve different commits")
    for arm, fields in arms.expect.items():
        for key, expected in fields.items():
            if _field(versions[arm], key) != expected:
                raise Stage1Error(f"arm {arm} serves {key}={_field(versions[arm], key)!r}, expected {expected!r}")
MAX_IMAGE_BYTES = 10 * 1024 * 1024
RETRYABLE = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
#: Stop below this OpenRouter balance: the official C9 shares the key. 40 by default; the user
#: lowered it to 5 on 2026-09-25 for the DeepSeek runs (RECALL_EXPERIMENT_CREDIT_FLOOR_USD).
CREDIT_FLOOR_USD = float(os.environ.get("RECALL_EXPERIMENT_CREDIT_FLOOR_USD", "40"))


class Stage1Error(RuntimeError):
    """A failure that invalidates the run rather than one row."""


@dataclass(frozen=True)
class Reply:
    status: int
    payload: dict[str, Any]
    headers: dict[str, str]
    latency_ms: float
    attempts: int


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _get(url: str, *, timeout: float = 120, headers: dict[str, str] | None = None) -> bytes:
    request = Request(url, headers={"User-Agent": "RE-call-mm-scope/1", **(headers or {})})
    with urlopen(request, timeout=timeout) as response:  # noqa: S310, pinned hosts only
        return bytes(response.read())


def safe_image_path(scenario: str, value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != scenario:
        raise Stage1Error(f"image path outside its scenario: {value}")
    if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
        raise Stage1Error(f"image format not allowed by the AML contract: {value}")
    return path


def fetch(cache_dir: Path) -> dict[str, Any]:
    """Download the eight scenario files and every referenced image at the pinned revision."""
    identity: dict[str, Any] = {"revision": MEMEYE_REVISION, "scenarios": {}}
    for scenario in SCENARIOS:
        raw = _get(f"{HF_BASE}/{MEMEYE_REVISION}/data/dialog/{scenario}.json")
        target = cache_dir / f"{scenario}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        dataset = json.loads(raw)
        images = sorted(
            {
                str(safe_image_path(scenario, str(image)))
                for session in dataset["multi_session_dialogues"]
                for dialogue in session["dialogues"]
                for image in dialogue.get("input_image", [])
            }
        )
        for relative in images:
            path = cache_dir / "image" / relative
            if not path.exists():
                encoded = "/".join(quote(part) for part in PurePosixPath(relative).parts)
                payload = _get(f"{HF_BASE}/{MEMEYE_REVISION}/data/image/{encoded}")
                if len(payload) > MAX_IMAGE_BYTES:
                    raise Stage1Error(f"image over 10 MiB: {relative}")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
        identity["scenarios"][scenario] = {
            "sha256": sha256_bytes(raw),
            "questions": len(dataset["human-annotated QAs"]),
            "images": len(images),
        }
    (cache_dir / "identity.json").write_text(json.dumps(identity, indent=2), encoding="utf-8")
    return identity


def _data_uri(path: Path) -> str:
    mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
    raw = path.read_bytes()
    return f"data:{mime[path.suffix.lower()]};base64,{base64.b64encode(raw).decode('ascii')}"


def add_requests(scenario: str, dataset: dict[str, Any], cache_dir: Path, user_id: str) -> list[dict[str, Any]]:
    """One Add per dialogue round, exactly as the frozen MemEye harness translates them."""
    requests: list[dict[str, Any]] = []
    for session in dataset["multi_session_dialogues"]:
        timestamp = int(
            datetime.fromisoformat(str(session["date"]))
            .replace(hour=12, tzinfo=timezone.utc)
            .timestamp()
            * 1_000
        )
        for dialogue in session["dialogues"]:
            images = [
                _data_uri(cache_dir / "image" / safe_image_path(scenario, str(value)))
                for value in dialogue.get("input_image", [])
            ]
            user_content: Any = dialogue["user"]
            if images:
                user_content = [{"type": "text", "text": dialogue["user"]}]
                user_content.extend({"type": "image_url", "image_url": {"url": url}} for url in images)
            round_id = str(dialogue["round"])
            requests.append(
                {
                    "request_id": "mms:" + sha256_bytes(f"{user_id}|{round_id}".encode()),
                    "user_id": user_id,
                    "session_id": round_id,
                    "messages": [
                        {"role": "user", "content": user_content, "timestamp": timestamp},
                        {"role": "assistant", "content": dialogue["assistant"], "timestamp": timestamp},
                    ],
                }
            )
    return requests


def call(port: int, path: str, payload: dict[str, Any], token: str, *, attempts: int = 6) -> Reply:
    body = json.dumps(payload, separators=(",", ":")).encode()
    last: Reply | None = None
    for attempt in range(1, attempts + 1):
        request = Request(
            f"http://127.0.0.1:{port}{path}",
            data=body,
            method="POST",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
        )
        started = time.perf_counter()
        try:
            with urlopen(request, timeout=900) as response:  # noqa: S310, localhost only
                raw = response.read()
                last = Reply(
                    response.status,
                    json.loads(raw),
                    {key.lower(): value for key, value in response.headers.items()},
                    (time.perf_counter() - started) * 1_000,
                    attempt,
                )
        except HTTPError as exc:
            raw = exc.read()
            try:
                error_payload = json.loads(raw)
            except json.JSONDecodeError:
                error_payload = {"error": raw[:200].decode("utf-8", "replace")}
            last = Reply(exc.code, error_payload, {}, (time.perf_counter() - started) * 1_000, attempt)
        if last.status not in RETRYABLE:
            return last
        time.sleep(min(30.0, 2.0**attempt))
    assert last is not None
    return last


def compact_item(item: dict[str, Any]) -> dict[str, Any]:
    """Keep everything the reader would see except the base64, which becomes its SHA-256."""
    content = item.get("content")
    if isinstance(content, list):
        parts: list[dict[str, Any]] = []
        for part in content:
            if part.get("type") == "image_url":
                value = part["image_url"]
                url = value["url"] if isinstance(value, dict) else value
                payload = base64.b64decode(url.split(",", 1)[1])
                parts.append({"type": "image", "sha256": sha256_bytes(payload), "bytes": len(payload)})
            else:
                parts.append({"type": "text", "text": part.get("text", "")})
        content = parts
    return {
        "id": item.get("id"),
        "session_id": item.get("session_id"),
        "kind": item.get("kind"),
        "created_at": item.get("created_at"),
        "score": item.get("score"),
        "content": content,
    }


def credit_balance(key: str) -> float:
    payload = json.loads(
        _get("https://openrouter.ai/api/v1/credits", headers={"Authorization": f"Bearer {key}"})
    )["data"]
    return float(payload["total_credits"]) - float(payload["total_usage"])


def require_credit(key: str) -> float:
    balance = credit_balance(key)
    if balance < CREDIT_FLOOR_USD:
        raise Stage1Error(
            f"OpenRouter balance {balance:.2f} is below the {CREDIT_FLOOR_USD:.0f} USD floor; "
            "stopping so this experiment cannot drain the official run"
        )
    return balance


def run_scenario(
    index: int,
    scenario: str,
    cache_dir: Path,
    run_id: str,
    token: str,
    openrouter_key: str,
    write: Callable[[str], None],
    arms: Arms = MM1_ARMS,
    keep_tenant: bool = False,
) -> dict[str, Any]:
    """Ingest one scenario through the ingest arm, Search it under every arm, then delete and verify it."""
    require_credit(openrouter_key)
    dataset = json.loads((cache_dir / f"{scenario}.json").read_text(encoding="utf-8"))
    user_id = f"mms-{run_id}-{index}"
    add_rows = []
    for request in add_requests(scenario, dataset, cache_dir, user_id):
        # request_id is fixed per (user, round), so a resumed run replays stored Adds for free.
        reply = call(arms.ports[arms.ingest], "/v1/add", request, token)
        if reply.status != 200:
            raise Stage1Error(f"{scenario} Add {request['session_id']} HTTP {reply.status}: {reply.payload}")
        has_image = isinstance(request["messages"][0]["content"], list)
        add_rows.append({**reply.payload, "has_image": has_image, "latency_ms": reply.latency_ms})
    searches = 0
    for q_index, qa in enumerate(dataset["human-annotated QAs"]):
        shift = q_index % len(arms.order)
        order = arms.order[shift:] + arms.order[:shift]
        lines: list[str] = []
        for rotation_index, rotation in enumerate(qa["options"]):
            options = {key: str(value) for key, value in rotation.items() if key != "answer"}
            request = {
                "query": qa["question"],
                "options": [options[key] for key in sorted(options)],
                "user_id": user_id,
                "top_k": 100,
            }
            for arm in order:
                reply = call(arms.ports[arm], "/v1/search", request, token)
                if reply.status != 200:
                    raise Stage1Error(f"{scenario} Search {qa['question_id']} {arm} HTTP {reply.status}")
                lines.append(
                    json.dumps(
                        {
                            "scenario": scenario,
                            "question_id": qa["question_id"],
                            "rotation": rotation_index,
                            "arm": arm,
                            "route": reply.headers.get("x-recall-specialist-route"),
                            "visual_leg": reply.headers.get("x-recall-visual-leg"),
                            "latency_ms": round(reply.latency_ms, 1),
                            "items": [compact_item(item) for item in reply.payload["data"]],
                        },
                        separators=(",", ":"),
                    )
                )
                searches += 1
        write("".join(line + "\n" for line in lines))
    if keep_tenant:
        # MM-4 counts the sidecars in the database before anything is deleted; ``cleanup`` then
        # deletes and verifies every tenant of the run.
        result = {"user_id": user_id, "adds": len(add_rows), "image_adds": sum(r["has_image"] for r in add_rows),
                  "text_adds": sum(not r["has_image"] for r in add_rows), "searches": searches,
                  "cleanup_passed": None}
        print(json.dumps({scenario: result}), flush=True)
        return result
    deleted = call(arms.ports[arms.ingest], "/v1/delete", {"user_id": user_id}, token)
    after = call(arms.ports[arms.ingest], "/v1/search",
                 {"query": "cleanup verification", "user_id": user_id, "top_k": 1}, token)
    cleanup_passed = deleted.status == 200 and after.status == 200 and not after.payload.get("data")
    result = {
        "user_id": user_id,
        "adds": len(add_rows),
        "image_adds": sum(row["has_image"] for row in add_rows),
        "text_adds": sum(not row["has_image"] for row in add_rows),
        "compiled_records": sum(int(row.get("compiled_count") or 0) for row in add_rows),
        "compiler_fallbacks": sum(bool(row.get("compiler_fallback")) for row in add_rows),
        "searches": searches,
        "deleted": deleted.payload,
        "cleanup_passed": cleanup_passed,
    }
    print(json.dumps({scenario: result}), flush=True)
    if not cleanup_passed:
        raise Stage1Error(f"{scenario} cleanup failed")
    return result


def completed_scenarios(out: Path, cache_dir: Path, arms: Arms = MM1_ARMS) -> set[str]:
    """Scenarios whose every question, rotation and arm is already in ``out``."""
    if not out.exists():
        return set()
    seen: dict[str, set[tuple[str, int, str]]] = {}
    with out.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            seen.setdefault(row["scenario"], set()).add((row["question_id"], row["rotation"], row["arm"]))
    done = set()
    for scenario, keys in seen.items():
        dataset = json.loads((cache_dir / f"{scenario}.json").read_text(encoding="utf-8"))
        expected = sum(len(qa["options"]) for qa in dataset["human-annotated QAs"]) * len(arms.order)
        if len(keys) == expected:
            done.add(scenario)
    return done


def cleanup(run_id: str, token: str, arms: Arms = MM1_ARMS) -> dict[str, Any]:
    """Delete and verify every scenario tenant a ``--keep-tenants`` run left, by its fixed user id."""
    port = arms.ports[arms.ingest]
    result = {}
    for index, scenario in enumerate(SCENARIOS):
        user_id = f"mms-{run_id}-{index}"
        deleted = call(port, "/v1/delete", {"user_id": user_id}, token)
        after = call(port, "/v1/search", {"query": "cleanup verification", "user_id": user_id, "top_k": 1}, token)
        result[scenario] = {"user_id": user_id, "deleted": deleted.payload,
                            "cleanup_passed": deleted.status == 200 and after.status == 200
                            and not after.payload.get("data")}
    if not all(entry["cleanup_passed"] for entry in result.values()):
        raise Stage1Error(f"cleanup failed: {result}")
    return result


def run(
    cache_dir: Path, out: Path, run_id: str, token: str, openrouter_key: str, *, workers: int = 1,
    arms: Arms = MM1_ARMS, keep_tenants: bool = False,
) -> dict[str, Any]:
    identity = json.loads((cache_dir / "identity.json").read_text(encoding="utf-8"))
    if identity["revision"] != MEMEYE_REVISION:
        raise Stage1Error("cache is not the pinned MemEye revision")
    versions = {}
    for arm in sorted(set(arms.expect) | {arms.ingest}):
        request = Request(f"http://127.0.0.1:{arms.ports[arm]}/version")
        with urlopen(request, timeout=30) as response:  # noqa: S310, localhost only
            versions[arm] = json.loads(response.read())
    check_versions(versions, arms)
    done = completed_scenarios(out, cache_dir, arms)
    summary: dict[str, Any] = {
        "run_id": run_id,
        "commit": versions[arms.ingest]["git_commit"],
        "generation_model": versions[arms.ingest]["generation_model"],
        "arms": {"order": list(arms.order), "ports": arms.ports, "ingest": arms.ingest},
        "versions": {arm: {"multimodal_scope": v.get("multimodal_scope"), "image_text": v.get("image_text")}
                     for arm, v in versions.items()},
        "credit_start_usd": require_credit(openrouter_key),
        "resumed_complete": sorted(done),
        "workers": workers,
        "scenarios": {},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()

    def write(text: str) -> None:
        with lock, out.open("a", encoding="utf-8") as sink:
            sink.write(text)

    pending = [(index, scenario) for index, scenario in enumerate(SCENARIOS) if scenario not in done]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            scenario: pool.submit(run_scenario, index, scenario, cache_dir, run_id, token, openrouter_key, write,
                                  arms, keep_tenants)
            for index, scenario in pending
        }
        for scenario, future in futures.items():
            summary["scenarios"][scenario] = future.result()
    summary["credit_end_usd"] = credit_balance(openrouter_key)
    (out.parent / (out.stem + ".summary.json")).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    fetch_parser = sub.add_parser("fetch")
    fetch_parser.add_argument("--cache-dir", type=Path, required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--cache-dir", type=Path, required=True)
    run_parser.add_argument("--out", type=Path, required=True)
    run_parser.add_argument("--run-id", required=True)
    run_parser.add_argument("--workers", type=int, default=1, help="scenarios run at once")
    run_parser.add_argument("--arm-config", type=Path, default=None,
                            help="JSON {order, ports, ingest, expect}; MM-1's four arms when omitted")
    run_parser.add_argument("--keep-tenants", action="store_true",
                            help="leave every scenario tenant for a census; `cleanup` deletes them")
    cleanup_parser = sub.add_parser("cleanup")
    cleanup_parser.add_argument("--run-id", required=True)
    cleanup_parser.add_argument("--arm-config", type=Path, default=None)
    args = parser.parse_args()
    if args.command == "fetch":
        print(json.dumps(fetch(args.cache_dir), indent=2))
        return
    token = os.environ.get("RECALL_AML_TOKEN", "").strip()
    if args.command == "cleanup":
        if not token:
            raise SystemExit("RECALL_AML_TOKEN must be set")
        print(json.dumps(cleanup(args.run_id, token, load_arms(args.arm_config)), indent=2))
        return
    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not token or not openrouter_key:
        raise SystemExit("RECALL_AML_TOKEN and OPENROUTER_API_KEY must be set")
    print(json.dumps(run(args.cache_dir, args.out, args.run_id, token, openrouter_key, workers=args.workers,
                         arms=load_arms(args.arm_config), keep_tenants=args.keep_tenants), indent=2))


if __name__ == "__main__":
    main()
