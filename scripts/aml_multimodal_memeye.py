"""Run the frozen public MemEye Brand pilot against one hosted AML arm."""

from __future__ import annotations

import argparse
import base64
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import mimetypes
import os
from pathlib import Path, PurePosixPath
import re
import time
from typing import Any, Callable, Iterable
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen


DATASET_REVISION = "f139f89d08bf0f66986153b5ff010cac24ba383d"
DATASET_JSON = "data/dialog/Brand_Memory_Test.json"
DATASET_SHA256 = "4aed4963277f742fe06f68b0910d7c1cfec5d0ed240904d291e02887712fbf78"
MEMEYE_COMMIT = "0358e70d714980980dfd8c87903384db474f0b16"
PROMPT_PATH = "benchmark/prompt/sys_prompt_mcq.txt"
HF_BASE = "https://huggingface.co/datasets/MemEyeBench/MemEye/resolve"
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/MinghoKwok/MemEye"
ANSWER_MODEL = "openai/gpt-4o-mini"
ANSWER_TOKEN_BUDGET = 117_760
IMAGE_TOKEN_ESTIMATE = 1_000
ANSWER_INPUT_USD_PER_MILLION = 1.0
ANSWER_OUTPUT_USD_PER_MILLION = 4.0
ANSWER_MAX_REQUEST_RESERVATION_USD = (
    ANSWER_TOKEN_BUDGET * ANSWER_INPUT_USD_PER_MILLION
    + 16 * ANSWER_OUTPUT_USD_PER_MILLION
) / 1_000_000
EXPERIMENT_COST_CEILING_USD = 25.0
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_RESPONSE_MEDIA_BYTES = 30 * 1024 * 1024
RETRYABLE_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 524})
ARMS = ("MM0_caption", "MM1_preserve", "MM2_dual")


class PilotError(RuntimeError):
    """A failure that invalidates this arm rather than one question."""


@dataclass(frozen=True)
class HttpResult:
    status: int
    payload: dict[str, Any]
    latency_ms: float
    response_bytes: int
    attempts: int


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _get_bytes(url: str, *, timeout: float = 120) -> bytes:
    request = Request(url, headers={"User-Agent": "RE-call-MemEye-pilot/1"})
    with urlopen(request, timeout=timeout) as response:  # noqa: S310, trusted pinned hosts
        return bytes(response.read())


def _safe_image_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise PilotError("dataset image path escapes the pinned image root")
    if not path.parts or path.parts[0] != "Brand_Memory_Test":
        raise PilotError("dataset image path is outside Brand_Memory_Test")
    if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
        raise PilotError("dataset image format is not supported by the AML contract")
    return path


def materialize_dataset(cache_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Download only the pinned Brand JSON and referenced images into a bounded cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset_url = f"{HF_BASE}/{DATASET_REVISION}/{DATASET_JSON}"
    raw = _get_bytes(dataset_url)
    if sha256_bytes(raw) != DATASET_SHA256:
        raise PilotError("pinned MemEye dataset JSON hash mismatch")
    payload = json.loads(raw)
    image_paths = sorted(
        {
            str(_safe_image_path(str(image)))
            for session in payload["multi_session_dialogues"]
            for dialogue in session["dialogues"]
            for image in dialogue.get("input_image", [])
        }
    )
    manifest: list[dict[str, Any]] = []
    for relative in image_paths:
        path = _safe_image_path(relative)
        target = cache_dir.joinpath(*path.parts)
        encoded = "/".join(quote(part) for part in path.parts)
        image_url = f"{HF_BASE}/{DATASET_REVISION}/data/image/{encoded}"
        image = _get_bytes(image_url)
        if len(image) > MAX_IMAGE_BYTES:
            raise PilotError(f"decoded image exceeds 10 MiB: {relative}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(image)
        manifest.append(
            {"path": relative, "bytes": len(image), "sha256": sha256_bytes(image)}
        )
    prompt = _get_bytes(f"{GITHUB_RAW_BASE}/{MEMEYE_COMMIT}/{PROMPT_PATH}")
    (cache_dir / "sys_prompt_mcq.txt").write_bytes(prompt)
    (cache_dir / "Brand_Memory_Test.json").write_bytes(raw)
    identity = {
        "dataset_revision": DATASET_REVISION,
        "dataset_json": DATASET_JSON,
        "dataset_sha256": DATASET_SHA256,
        "memeye_commit": MEMEYE_COMMIT,
        "prompt_path": PROMPT_PATH,
        "prompt_sha256": sha256_bytes(prompt),
        "image_count": len(manifest),
        "images": manifest,
    }
    (cache_dir / "identity.json").write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload, identity


def load_materialized_dataset(cache_dir: Path) -> tuple[dict[str, Any], dict[str, Any], str]:
    raw = (cache_dir / "Brand_Memory_Test.json").read_bytes()
    if sha256_bytes(raw) != DATASET_SHA256:
        raise PilotError("materialized MemEye dataset JSON hash mismatch")
    identity = json.loads((cache_dir / "identity.json").read_text(encoding="utf-8"))
    if identity.get("dataset_revision") != DATASET_REVISION:
        raise PilotError("materialized MemEye revision mismatch")
    if identity.get("dataset_sha256") != DATASET_SHA256:
        raise PilotError("materialized MemEye identity hash mismatch")
    prompt_bytes = (cache_dir / "sys_prompt_mcq.txt").read_bytes()
    if sha256_bytes(prompt_bytes) != identity.get("prompt_sha256"):
        raise PilotError("materialized MemEye prompt hash mismatch")
    for item in identity.get("images", []):
        path = _safe_image_path(str(item["path"]))
        image = cache_dir.joinpath(*path.parts).read_bytes()
        if len(image) != item.get("bytes") or sha256_bytes(image) != item.get("sha256"):
            raise PilotError(f"materialized image identity mismatch: {path}")
    return json.loads(raw), identity, prompt_bytes.decode("utf-8")


def _data_uri(path: Path) -> str:
    raw = path.read_bytes()
    if len(raw) > MAX_IMAGE_BYTES:
        raise PilotError(f"decoded image exceeds 10 MiB: {path.name}")
    mime = mimetypes.guess_type(path.name)[0]
    if mime not in {"image/jpeg", "image/png", "image/webp"}:
        raise PilotError(f"unsupported image MIME type: {mime}")
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def build_add_requests(
    dataset: dict[str, Any], cache_dir: Path, *, arm: str, user_id: str
) -> list[dict[str, Any]]:
    """Translate each public dialogue round into one source-grounded AML Add request."""
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
                _data_uri(cache_dir.joinpath(*_safe_image_path(str(value)).parts))
                for value in dialogue.get("input_image", [])
            ]
            if images:
                user_content: Any = [{"type": "text", "text": dialogue["user"]}]
                user_content.extend(
                    {"type": "image_url", "image_url": {"url": image}} for image in images
                )
            else:
                user_content = dialogue["user"]
            round_id = str(dialogue["round"])
            request_id = "memeye:" + sha256_bytes(
                canonical_json(
                    {
                        "revision": DATASET_REVISION,
                        "arm": arm,
                        "user_id": user_id,
                        "round_id": round_id,
                    }
                )
            )
            requests.append(
                {
                    "request_id": request_id,
                    "user_id": user_id,
                    "session_id": round_id,
                    "messages": [
                        {"role": "user", "content": user_content, "timestamp": timestamp},
                        {
                            "role": "assistant",
                            "content": dialogue["assistant"],
                            "timestamp": timestamp,
                        },
                    ],
                }
            )
    return requests


class JsonClient:
    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        timeout: float = 600,
        max_attempts: int = 3,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.sleep = sleep

    def call(self, path: str, payload: dict[str, Any] | None = None) -> HttpResult:
        body = None if payload is None else canonical_json(payload)
        headers = {"Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if body is not None:
            headers["Content-Type"] = "application/json"
        last: HttpResult | None = None
        for attempt in range(1, self.max_attempts + 1):
            request = Request(
                self.base_url + path,
                data=body,
                method="GET" if body is None else "POST",
                headers=headers,
            )
            started = time.perf_counter()
            try:
                with urlopen(request, timeout=self.timeout) as response:  # noqa: S310
                    raw = response.read()
                    result = HttpResult(
                        response.status,
                        json.loads(raw),
                        (time.perf_counter() - started) * 1_000,
                        len(raw),
                        attempt,
                    )
            except HTTPError as exc:
                raw = exc.read()
                try:
                    error_payload = json.loads(raw)
                except json.JSONDecodeError:
                    error_payload = {"error": "non-json response"}
                result = HttpResult(
                    exc.code,
                    error_payload,
                    (time.perf_counter() - started) * 1_000,
                    len(raw),
                    attempt,
                )
            last = result
            if result.status not in RETRYABLE_STATUSES or attempt == self.max_attempts:
                return result
            self.sleep(float(2 ** (attempt - 1)))
        assert last is not None
        return last


def decoded_media_bytes(content: Any) -> int:
    if not isinstance(content, list):
        return 0
    total = 0
    for part in content:
        if not isinstance(part, dict) or part.get("type") != "image_url":
            continue
        value = part.get("image_url")
        if isinstance(value, dict):
            value = value.get("url")
        if not isinstance(value, str) or ";base64," not in value:
            raise PilotError("Search returned a non-inline image")
        encoded = value.split(";base64,", 1)[1]
        total += len(base64.b64decode(encoded, validate=True))
    return total


def retrieval_metrics(items: list[dict[str, Any]], clues: Iterable[str]) -> dict[str, Any]:
    clue_set = set(clues)
    ranks = [
        rank
        for rank, item in enumerate(items, start=1)
        if str(item.get("session_id", "")) in clue_set
    ]
    metrics: dict[str, Any] = {
        "clue_count": len(clue_set),
        "first_clue_rank": min(ranks) if ranks else None,
        "mrr": 0.0 if not ranks else 1.0 / min(ranks),
    }
    returned_sessions = [str(item.get("session_id", "")) for item in items]
    for cutoff in (5, 10, 100):
        found = clue_set.intersection(returned_sessions[:cutoff])
        metrics[f"any_recall_at_{cutoff}"] = float(bool(found))
        metrics[f"complete_recall_at_{cutoff}"] = float(found == clue_set)
        metrics[f"clue_fraction_at_{cutoff}"] = (
            len(found) / len(clue_set) if clue_set else 1.0
        )
    return metrics


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return sum(materialized) / len(materialized) if materialized else 0.0


def aggregate_questions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    retrieval_keys = [
        *(f"any_recall_at_{k}" for k in (5, 10, 100)),
        *(f"complete_recall_at_{k}" for k in (5, 10, 100)),
        *(f"clue_fraction_at_{k}" for k in (5, 10, 100)),
        "mrr",
    ]
    overall: dict[str, Any] = {
        "question_count": len(rows),
        "rotation_count": sum(len(row["rotations"]) for row in rows),
        "mean_debiased_em": _mean(float(row["debiased_em"]) for row in rows),
        "strict_question_accuracy": _mean(
            float(bool(row["strict_exact_match"])) for row in rows
        ),
        "valid_choice_rate": _mean(
            float(rotation["valid_choice"])
            for row in rows
            for rotation in row["rotations"]
        ),
    }
    for key in retrieval_keys:
        overall[key] = _mean(
            float(rotation["retrieval"][key])
            for row in rows
            for rotation in row["rotations"]
        )
    positions = Counter(
        rotation["selected_position"]
        for row in rows
        for rotation in row["rotations"]
        if rotation["selected_position"] != "INVALID"
    )
    overall["selected_position_counts"] = dict(sorted(positions.items()))
    by_axis: dict[str, dict[str, float]] = {}
    axes = sorted({axis for row in rows for axis in row["axes"]})
    for axis in axes:
        axis_rows = [row for row in rows if axis in row["axes"]]
        by_axis[axis] = {
            "question_count": float(len(axis_rows)),
            "mean_debiased_em": _mean(float(row["debiased_em"]) for row in axis_rows),
            "any_recall_at_10": _mean(
                float(rotation["retrieval"]["any_recall_at_10"])
                for row in axis_rows
                for rotation in row["rotations"]
            ),
            "complete_recall_at_10": _mean(
                float(rotation["retrieval"]["complete_recall_at_10"])
                for row in axis_rows
                for rotation in row["rotations"]
            ),
        }
    overall["by_axis"] = by_axis
    return overall


def _content_parts(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if not isinstance(content, list):
        raise PilotError("Search content has an invalid shape")
    rendered: list[dict[str, Any]] = []
    for part in content:
        if part.get("type") == "text":
            rendered.append({"type": "text", "text": str(part["text"])})
        elif part.get("type") == "image_url":
            value = part["image_url"]
            if isinstance(value, dict):
                value = value["url"]
            rendered.append(
                {"type": "image_url", "image_url": {"url": value, "detail": "low"}}
            )
        else:
            raise PilotError("Search returned an unknown content part")
    return rendered


def pack_answer_content(
    items: list[dict[str, Any]], question: str, options: dict[str, str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Pack only a ranked prefix under the registered conservative token estimate."""
    parts: list[dict[str, Any]] = [
        {"type": "text", "text": "Retrieved memory evidence follows in rank order.\n"}
    ]
    estimated_tokens = math.ceil(len(question) / 4) + 128
    admitted = 0
    for rank, item in enumerate(items, start=1):
        candidate = [
            {
                "type": "text",
                "text": f"\nMemory {rank}, session {item.get('session_id', '')}:\n",
            },
            *_content_parts(item["content"]),
        ]
        candidate_tokens = 0
        for part in candidate:
            if part["type"] == "text":
                candidate_tokens += math.ceil(len(part["text"]) / 4)
            else:
                candidate_tokens += IMAGE_TOKEN_ESTIMATE
        if estimated_tokens + candidate_tokens > ANSWER_TOKEN_BUDGET:
            break
        parts.extend(candidate)
        estimated_tokens += candidate_tokens
        admitted += 1
    option_lines = "\n".join(f"{key}. {options[key]}" for key in sorted(options))
    parts.append(
        {
            "type": "text",
            "text": (
                "\nQuestion:\n"
                + question
                + "\nOptions:\n"
                + option_lines
                + "\nAnswer with ONLY the option letter. Do not explain."
            ),
        }
    )
    return parts, {
        "returned_items": len(items),
        "admitted_items": admitted,
        "estimated_input_tokens": estimated_tokens,
        "truncated": admitted < len(items),
    }


def extract_choice(text: str, valid_keys: set[str]) -> str:
    keys = "".join(sorted(re.escape(key.upper()) for key in valid_keys))
    stripped = text.strip().upper()
    if stripped in valid_keys:
        return stripped
    match = re.search(rf"(?:ANSWER|CHOICE)\s*(?:IS|:)\s*([{keys}])\b", stripped)
    if match:
        return match.group(1)
    match = re.search(rf"\b([{keys}])\b", stripped)
    return match.group(1) if match else "INVALID"


def answer_rotation(
    client: JsonClient,
    *,
    api_key: str,
    system_prompt: str,
    parts: list[dict[str, Any]],
    reserve_timeout_cost: Callable[[float], None] | None = None,
) -> tuple[str, dict[str, Any]]:
    answer_client = JsonClient(
        "https://openrouter.ai/api/v1",
        api_key,
        timeout=600,
        max_attempts=1,
        sleep=client.sleep,
    )
    payload = {
        "model": ANSWER_MODEL,
        "temperature": 0,
        "max_tokens": 16,
        "usage": {"include": True},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": parts},
        ],
    }
    result: HttpResult | None = None
    timeout_retry_count = 0
    timeout_reserved_cost = 0.0
    started = time.perf_counter()
    for attempt in range(1, 4):
        try:
            result = answer_client.call("/chat/completions", payload)
        except TimeoutError as exc:
            timeout_retry_count += 1
            timeout_reserved_cost += ANSWER_MAX_REQUEST_RESERVATION_USD
            if reserve_timeout_cost is not None:
                reserve_timeout_cost(ANSWER_MAX_REQUEST_RESERVATION_USD)
            if attempt == 3:
                raise PilotError("Answer provider timed out after three attempts") from exc
            client.sleep(float(2 ** (attempt - 1)))
            continue
        if result.status not in RETRYABLE_STATUSES or attempt == 3:
            break
        client.sleep(float(2 ** (attempt - 1)))
    if result is None:
        raise PilotError("Answer provider returned no result")
    if result.status != 200:
        raise PilotError(f"Answer provider returned HTTP {result.status}")
    choices = result.payload.get("choices") or []
    if not choices:
        raise PilotError("Answer provider returned no choices")
    content = choices[0].get("message", {}).get("content")
    if not isinstance(content, str) or not content.strip():
        raise PilotError("Answer provider returned no text")
    usage = result.payload.get("usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    reported_cost = float(usage.get("cost") or 0.0)
    conservative_cost = (
        prompt_tokens * ANSWER_INPUT_USD_PER_MILLION
        + completion_tokens * ANSWER_OUTPUT_USD_PER_MILLION
    ) / 1_000_000
    return content, {
        "latency_ms": (time.perf_counter() - started) * 1_000,
        "attempts": attempt,
        "timeout_retry_count": timeout_retry_count,
        "timeout_reserved_cost_usd": timeout_reserved_cost,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": int(usage.get("total_tokens") or 0),
        "reported_cost_usd": reported_cost,
        "conservative_cost_usd": conservative_cost,
        "cost_usd": max(reported_cost, conservative_cost),
    }


def _axes(qa: dict[str, Any]) -> list[str]:
    return sorted(
        {
            str(value)
            for group in qa.get("point", [])
            for value in (group if isinstance(group, list) else [group])
        }
    )


def run_arm(
    *,
    arm: str,
    base_url: str,
    token: str,
    answer_key: str,
    cache_dir: Path,
    output: Path,
    run_id: str,
    spend_ledger: Path,
) -> dict[str, Any]:
    if arm not in ARMS:
        raise PilotError(f"unknown arm: {arm}")
    dataset, identity, system_prompt = load_materialized_dataset(cache_dir)
    client = JsonClient(base_url, token)
    user_id = f"memeye-brand:{run_id}:{arm}"
    started = time.monotonic()
    spend_usd = 0.0
    prior_spend_usd = 0.0
    if spend_ledger.exists():
        prior_spend_usd = float(
            json.loads(spend_ledger.read_text(encoding="utf-8"))["total_spend_usd"]
        )

    def record_spend(amount: float) -> None:
        nonlocal spend_usd
        spend_usd += amount
        cumulative_spend = prior_spend_usd + spend_usd
        spend_ledger.parent.mkdir(parents=True, exist_ok=True)
        ledger_tmp = spend_ledger.with_suffix(spend_ledger.suffix + ".tmp")
        ledger_tmp.write_text(
            json.dumps({"total_spend_usd": cumulative_spend}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        ledger_tmp.replace(spend_ledger)
        if cumulative_spend > EXPERIMENT_COST_CEILING_USD:
            raise PilotError("registered provider cost ceiling exceeded")

    add_rows: list[dict[str, Any]] = []
    question_rows: list[dict[str, Any]] = []
    cleanup: dict[str, Any] = {"attempted": False, "passed": False}
    error: str | None = None
    try:
        health = client.call("/health")
        if health.status != 200 or health.payload.get("status") != "ready":
            raise PilotError("hosted arm is not ready")
        version = client.call("/version")
        if version.status != 200 or version.payload.get("variant") != arm:
            raise PilotError("hosted arm identity mismatch")
        requests = build_add_requests(dataset, cache_dir, arm=arm, user_id=user_id)
        for request in requests:
            added = client.call("/v1/add", request)
            if added.status != 200:
                raise PilotError(f"Add returned HTTP {added.status}")
            expected = {
                "request_id": request["request_id"],
                "user_id": user_id,
                "session_id": request["session_id"],
            }
            if any(added.payload.get(key) != value for key, value in expected.items()):
                raise PilotError("Add did not echo the frozen identifiers")
            add_rows.append(
                {
                    "session_id": request["session_id"],
                    "latency_ms": added.latency_ms,
                    "attempts": added.attempts,
                }
            )
        replay = client.call("/v1/add", requests[0])
        if replay.status != 200:
            raise PilotError("idempotent Add replay failed")

        for qa in dataset["human-annotated QAs"]:
            rotations: list[dict[str, Any]] = []
            for rotation in qa["options"]:
                if prior_spend_usd + spend_usd >= EXPERIMENT_COST_CEILING_USD:
                    raise PilotError("registered experiment provider cost ceiling reached")
                options = {key: str(value) for key, value in rotation.items() if key != "answer"}
                search = client.call(
                    "/v1/search",
                    {
                        "query": qa["question"],
                        "options": [options[key] for key in sorted(options)],
                        "user_id": user_id,
                        "top_k": 100,
                    },
                )
                if search.status != 200:
                    raise PilotError(f"Search returned HTTP {search.status}")
                items = search.payload.get("data")
                if not isinstance(items, list) or len(items) > 100:
                    raise PilotError("Search returned an invalid result list")
                media_bytes = sum(decoded_media_bytes(item.get("content")) for item in items)
                if media_bytes > MAX_RESPONSE_MEDIA_BYTES:
                    raise PilotError("Search exceeded the decoded-media response budget")
                content, packing = pack_answer_content(items, qa["question"], options)
                answer, usage = answer_rotation(
                    client,
                    api_key=answer_key,
                    system_prompt=system_prompt,
                    parts=content,
                    reserve_timeout_cost=record_spend,
                )
                record_spend(usage["cost_usd"])
                selected = extract_choice(answer, set(options))
                rotations.append(
                    {
                        "selected_position": selected,
                        "valid_choice": selected != "INVALID",
                        "em": float(selected == str(rotation["answer"]).upper()),
                        "retrieval": retrieval_metrics(items, qa["clue"]),
                        "search_latency_ms": search.latency_ms,
                        "search_attempts": search.attempts,
                        "search_response_bytes": search.response_bytes,
                        "search_decoded_media_bytes": media_bytes,
                        "packing": packing,
                        "answer_usage": usage,
                    }
                )
            debiased = _mean(float(row["em"]) for row in rotations)
            question_rows.append(
                {
                    "question_id": qa["question_id"],
                    "axes": _axes(qa),
                    "clue_count": len(set(qa["clue"])),
                    "rotations": rotations,
                    "debiased_em": debiased,
                    "strict_exact_match": debiased == 1.0,
                }
            )
            if time.monotonic() - started > 4 * 60 * 60:
                raise PilotError("registered wall-clock ceiling exceeded")
    except Exception as exc:  # noqa: BLE001, the invalid artifact must survive every arm failure
        error = f"{type(exc).__name__}: {exc}"
    finally:
        cleanup["attempted"] = True
        deleted = client.call("/v1/delete", {"user_id": user_id})
        after = client.call(
            "/v1/search", {"query": "cleanup verification", "user_id": user_id, "top_k": 1}
        )
        cleanup.update(
            {
                "delete_status": deleted.status,
                "after_search_status": after.status,
                "after_items": len(after.payload.get("data", [])),
                "passed": deleted.status == 200
                and after.status == 200
                and not after.payload.get("data"),
            }
        )
        if not cleanup["passed"] and error is None:
            error = "PilotError: cleanup verification failed"
    final_result: dict[str, Any] = {
        "schema_version": 1,
        "arm": arm,
        "run_id": run_id,
        "complete": error is None and len(question_rows) == 29 and cleanup["passed"],
        "error": error,
        "identity": {
            **{key: value for key, value in identity.items() if key != "images"},
            "answer_model": ANSWER_MODEL,
            "answer_temperature": 0,
            "answer_token_budget": ANSWER_TOKEN_BUDGET,
            "top_k": 100,
        },
        "add": {
            "count": len(add_rows),
            "replay_passed": len(add_rows) == 72 and replay.status == 200
            if "replay" in locals()
            else False,
            "latencies_ms": [row["latency_ms"] for row in add_rows],
            "retry_count": sum(max(0, row["attempts"] - 1) for row in add_rows),
        },
        "questions": question_rows,
        "aggregate": aggregate_questions(question_rows),
        "provider_spend_usd": spend_usd,
        "cleanup": cleanup,
        "elapsed_seconds": time.monotonic() - started,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(final_result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return final_result


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch")
    fetch.add_argument("--cache-dir", type=Path, required=True)
    run = sub.add_parser("run-arm")
    run.add_argument("--arm", choices=ARMS, required=True)
    run.add_argument("--base-url", required=True)
    run.add_argument("--token-env", default="RECALL_AML_TOKEN")
    run.add_argument("--answer-key-env", default="OPENROUTER_API_KEY")
    run.add_argument("--cache-dir", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--run-id", required=True)
    run.add_argument("--spend-ledger", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "fetch":
        _, identity = materialize_dataset(args.cache_dir)
        print(json.dumps({key: value for key, value in identity.items() if key != "images"}, indent=2))
        return
    token = os.environ.get(args.token_env, "").strip()
    answer_key = os.environ.get(args.answer_key_env, "").strip()
    if not token or not answer_key:
        raise SystemExit("required API credentials are not set")
    result = run_arm(
        arm=args.arm,
        base_url=args.base_url,
        token=token,
        answer_key=answer_key,
        cache_dir=args.cache_dir,
        output=args.output,
        run_id=args.run_id,
        spend_ledger=args.spend_ledger,
    )
    print(json.dumps({"arm": args.arm, "complete": result["complete"], "error": result["error"]}))
    if not result["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
