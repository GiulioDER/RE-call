"""Fixes from the independent review of the C9 speed branch (2026-09-26), one test each.

Red proof, 2026-09-26, each by mutating the named production line with this file unchanged,
watching the named assertion fail, then restoring it:

* ``test_a_retry_after_above_the_shared_ceiling_waits_the_clamp``: calling
  ``_retry_after_seconds(exc)`` without ``cap=None`` in ``_retry_delay``
  (``recall_aml/compiler.py``) failed ``assert sleeps == [10.0, 10.0]`` with ``[0.25, 0.5]``.
* ``test_a_413_over_the_budget_is_fitted_once``: ``_REFIT_STATUSES = frozenset({400})`` failed
  ``assert records`` (the 413 was raised on the first attempt, one call).
* ``test_a_failed_supersedable_lookup_drops_the_reference_not_the_compile``: replacing the
  ``try``/``except`` around ``loader()`` in ``_supersedable`` with ``cache.append(set(loader()))``
  failed with ``RuntimeError: database went away`` out of ``compile_anchored_v3``.
* ``test_build_compiler_refuses_a_short_answer_on_the_v2_compiler``: deleting the version guard in
  ``build_compiler`` (``recall_aml/__main__.py``) failed with ``DID NOT RAISE ValueError``.
* ``test_a_cancelled_prefetch_is_treated_as_absent``: replacing ``return None`` in the
  ``except asyncio.CancelledError`` branch of ``HostedService._prefetched`` with ``pass`` (so a
  cancelled prefetch re-raises into the Add) failed with ``CancelledError``.
* ``test_a_failed_add_does_not_wait_for_its_prefetch``: making the ``finally`` of
  ``HostedService._add_once`` wait whenever ``prefetch_task is not None`` (dropping ``and
  stored``) failed ``assert elapsed < 1.5`` (the Add waited the full 3 s for the blocked prefetch).
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest

import recall_aml.compiler as compiler_module
from recall_aml.compiler import COMPILER_MAX_RETRY_AFTER_SECONDS, OpenAICompiler, PriorRecords
from recall_aml.models import Message
from recall_aml.service import HostedService
from tests.test_aml_add_concurrency import (
    GroundedCompiler,
    RecordingRepository,
    add_request,
    c9_service,
)
from tests.test_aml_compiler_input_cap import compiler_with_window, session

openai = pytest.importorskip("openai")
httpx = pytest.importorskip("httpx")


def _status_error(status: int, headers: dict[str, str] | None = None) -> Exception:
    response = httpx.Response(
        status,
        headers=headers or {},
        request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"),
    )
    client = openai.OpenAI(api_key="test", base_url="https://openrouter.ai/api/v1", max_retries=0)
    error: Exception = client._make_status_error_from_response(response)
    return error


def _client(create: Any) -> Any:
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def test_a_retry_after_above_the_shared_ceiling_waits_the_clamp() -> None:
    sleeps: list[float] = []

    def create(**_: Any) -> Any:
        raise _status_error(429, {"retry-after": "120"})

    compiler = OpenAICompiler(_client(create), sleep=sleeps.append)
    with pytest.raises(openai.RateLimitError):
        compiler.compile_anchored_v3([Message(role="user", content="pytest passed")], "s", [])

    assert sleeps == [COMPILER_MAX_RETRY_AFTER_SECONDS, COMPILER_MAX_RETRY_AFTER_SECONDS]


def test_the_shared_retry_after_ceiling_is_unchanged_for_embedders() -> None:
    """Control, not a behaviour change: the embedders' own reading still ignores a 120 s ask."""
    from recall.embeddings import _retry_after_seconds

    error = _status_error(429, {"retry-after": "120"})
    assert _retry_after_seconds(error) is None
    assert _retry_after_seconds(error, cap=None) == 120.0


class PayloadTooLarge(Exception):
    status_code = 413


def test_a_413_over_the_budget_is_fitted_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(compiler_module, "ANCHOR_PAYLOAD_BUDGET_CHARS", 20_000)
    calls: list[dict[str, Any]] = []
    compiler = compiler_with_window(25_000, calls)
    inner = compiler._client.chat.completions.create

    def create(**kwargs: Any) -> Any:
        try:
            return inner(**kwargs)
        except Exception as exc:  # the window helper raises its own 400 class
            raise PayloadTooLarge("request entity too large") from exc

    compiler._client = _client(create)
    try:
        records = compiler.compile_anchored_v3(session(40), "session", [])
    except PayloadTooLarge:
        records = []

    assert records
    assert len(calls) == 2


def test_a_failed_supersedable_lookup_drops_the_reference_not_the_compile() -> None:
    text = "Replaced mem_0a1b2c3d with the queue fix and pytest passed."
    anchor = compiler_module.build_evidence_anchors(
        [Message(role="user", content=text)], "s", identifier_version=3
    )[0]
    answer = {
        "records": [
            {
                "kind": "successful repair",
                "action": "the queue fix",
                "evidence_anchor_ids": [anchor.id],
                "source_session_id": "s",
                "supersedes": ["mem_0a1b2c3d"],
            }
        ]
    }

    def create(**_: Any) -> Any:
        return SimpleNamespace(
            choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=json.dumps(answer)))]
        )

    def unavailable() -> set[str]:
        raise RuntimeError("database went away")

    compiler = OpenAICompiler(_client(create), sleep=lambda _: None)
    records = compiler.compile_anchored_v3(
        [Message(role="user", content=text)], "s", PriorRecords([], unavailable)
    )

    assert [record.action for record in records] == ["the queue fix"]
    assert records[0].supersedes == []


def test_build_compiler_refuses_a_short_answer_on_the_v2_compiler() -> None:
    from recall_aml.__main__ import build_compiler
    from recall_aml.config import HostedSettings
    from recall_aml.variants import VARIANTS

    v2 = next(item for item in VARIANTS if item.anchor_compiler and item.anchor_compiler_version == 2)
    settings = HostedSettings(
        database_url="postgresql://unused", api_key="k", git_commit="c", openrouter_api_key="or"
    )

    with pytest.raises(ValueError, match="v3 anchored compiler only"):
        build_compiler(
            settings,
            dataclasses.replace(v2, anchor_compile_output="select"),
            client_factory=lambda **_: object(),
        )


def test_a_cancelled_prefetch_is_treated_as_absent() -> None:
    async def scenario() -> Any:
        task: asyncio.Task[Any] = asyncio.create_task(asyncio.sleep(3600))
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        return await HostedService._prefetched(task, [])

    assert asyncio.run(scenario()) is None


def test_a_failed_add_does_not_wait_for_its_prefetch() -> None:
    release = threading.Event()
    repository = RecordingRepository(prefetch=True)
    original = repository.embed_texts

    def blocked(*args: Any) -> Any:
        release.wait(3.0)
        return original(*args)

    def broken(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("prior records unavailable")

    repository.embed_texts = blocked  # type: ignore[method-assign]
    repository.prior_records = broken  # type: ignore[method-assign]
    service = c9_service(repository, GroundedCompiler())

    async def scenario() -> tuple[float, BaseException | None]:
        # Timed inside the loop: ``asyncio.run`` joins executor threads when it closes, which
        # would add the blocked prefetch's wait to any time measured around it.
        started = time.perf_counter()
        error: BaseException | None = None
        try:
            await service.add(add_request())
        except RuntimeError as exc:
            error = exc
        elapsed = time.perf_counter() - started
        release.set()
        return elapsed, error

    try:
        elapsed, error = asyncio.run(scenario())
    finally:
        release.set()

    assert isinstance(error, RuntimeError) and "prior records unavailable" in str(error)
    assert elapsed < 1.5
