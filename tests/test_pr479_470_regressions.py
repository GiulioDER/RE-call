"""Regressions for the surviving defects in PRs 479 through 470.

Red proof: run these unchanged nodes against the baseline implementation.
The upload, erasure receipt, MCP refusal and concurrent admission assertions must
fail there, then pass with the corresponding production repairs.
"""

from __future__ import annotations

import asyncio
import base64
import json
import tracemalloc
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.conftest import requires_db


@pytest.mark.parametrize("parts", [(b"abcde",), (b"abc", b"de")])
def test_upload_accepts_the_decoded_byte_boundary(monkeypatch, tmp_path, parts):
    """FIX-001: stage_uploads must subtract valid padding from its size estimate.

    Baseline uploads.py precheck refuses five decoded bytes, estimated as six.
    The oversized control must still refuse before the boundary is accepted.
    """
    from recall.desktop import uploads

    monkeypatch.setenv("RECALL_INDEX_ROOT", str(tmp_path))
    monkeypatch.setattr(uploads, "_MAX_TOTAL_BYTES", 5)
    with pytest.raises(uploads.UploadError):
        uploads.stage_uploads("acme", [{"name": "large", "content_b64": "YWJjZGVm"}])
    files = [
        {"name": f"memo{i}.md", "content_b64": base64.b64encode(data).decode("ascii")}
        for i, data in enumerate(parts)
    ]
    error = None
    try:
        _job, root, total = uploads.stage_uploads("acme", files)
    except uploads.UploadError as exc:
        error = exc
    assert error is None, f"valid decoded-byte boundary was refused: {error}"
    assert total == 5
    assert [p.read_bytes() for p in sorted(root.iterdir())] == list(parts)


def test_padded_oversize_is_refused_without_copying_the_payload(monkeypatch, tmp_path):
    """FIX-001 revision: bound guard allocations independently of encoded input size.

    Red against the first repair's str.rstrip: a padded 1 MiB payload allocates
    another roughly 1.3 MiB string before refusal. The constant-size suffix guard
    stays below 128 KiB, allowing ordinary staging/error bookkeeping overhead.
    """
    from recall.desktop import uploads

    monkeypatch.setenv("RECALL_INDEX_ROOT", str(tmp_path))
    monkeypatch.setattr(uploads, "_MAX_TOTAL_BYTES", 5)
    encoded = base64.b64encode(b"x" * (1024 * 1024)).decode("ascii")
    tracemalloc.start()
    try:
        with pytest.raises(uploads.UploadError):
            uploads.stage_uploads("acme", [{"name": "large", "content_b64": encoded}])
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 128 * 1024, "the refusal guard copied the oversized padded payload"


@requires_db
def test_forget_reports_incomplete_staged_erasure(make_store, monkeypatch, tmp_path):
    """FIX-002: public forget must report an in-scope unlink failure.

    Baseline recall.uploads.delete_staged_sources swallows PermissionError, so
    the receipt assertion fails although rows are erased and one file survives.
    Successful, absent and outside-staging files remain controls in this node.
    """
    from recall.types import Chunk
    from recall_mcp.service import forget_memory

    monkeypatch.setenv("RECALL_INDEX_ROOT", str(tmp_path))
    store = make_store(3)
    root = tmp_path / "uploads" / str(store.tenant) / "job"
    root.mkdir(parents=True)
    blocked, good, missing = [root / name for name in ("blocked.md", "good.md", "missing.md")]
    outside = tmp_path / "outside.md"
    for path in (blocked, good, outside):
        path.write_text("private memo", encoding="utf-8")
    paths = (blocked, good, missing, outside)
    store.upsert(
        [Chunk(str(i), str(path), "private memo", {"file": path.name})
         for i, path in enumerate(paths)],
        [[1.0, 0.0, 0.0]] * len(paths),
    )
    unlink = Path.unlink

    def refuse_one(path, *args, **kwargs):
        if path == blocked:
            raise PermissionError("simulated staged-file refusal")
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", refuse_one)
    result = forget_memory(store, [p.name for p in paths], erase_memo_requests=False)
    assert store.count() == 0
    assert blocked.exists() and outside.exists()
    assert not good.exists() and not missing.exists()
    assert result.staged_files_removed == -1, "retained staged bytes must be a failed cleanup receipt"
    assert "WARNING" in result.message and "staged upload cleanup failed" in result.message
    assert "before the next index" in result.message


@pytest.mark.parametrize(
    ("payload", "tool_error"),
    [
        ({"error": "trust_refusal", "failure_code": "INDEX_NOT_READY"}, False),
        ({"hits": []}, True),
        ({}, False),
        ({"hits": "invalid"}, False),
        ([], False),
        ({"hits": ["invalid"]}, False),
    ],
)
def test_expansion_probe_refuses_failed_or_malformed_search(monkeypatch, payload, tool_error):
    """FIX-003: retrieve must distinguish refused service from an empty working gate.

    Baseline retrieve either reports an empty miss or leaks an unrelated shape
    exception. The intended assertion requires a deliberate SystemExit refusal.
    A successful empty result is checked through the same transport first.
    """
    import mcp
    import mcp.client.stdio

    from scripts import agent_ab_probe_expansion as probe

    response = SimpleNamespace(content=[SimpleNamespace(text='{"hits": []}')], isError=False)

    @asynccontextmanager
    async def transport(_params):
        yield None, None

    class Session:
        def __init__(self, *_args):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def initialize(self):
            return None

        async def call_tool(self, *_args):
            return response

    monkeypatch.setattr(mcp, "ClientSession", Session)
    monkeypatch.setattr(mcp.client.stdio, "stdio_client", transport)
    spec = SimpleNamespace(python="python", cwd=Path.cwd(), env=lambda: {})
    assert asyncio.run(probe.retrieve(spec, ["query"])) == {"query": []}
    response.content = [SimpleNamespace(text=json.dumps(payload))]
    response.isError = tool_error
    error = None
    try:
        asyncio.run(probe.retrieve(spec, ["query"]))
    except (SystemExit, Exception) as exc:
        error = exc
    assert isinstance(error, SystemExit), f"search failure was not deliberately refused: {error!r}"


@pytest.mark.parametrize("initial", [None, "success", "outage"])
def test_oidc_reserves_failure_budget_before_yielding(monkeypatch, initial):
    """FIX-004: one fixed-clock token admits only one concurrent validation.

    Baseline server_auth.py checks allow(), yields, and only debits after all
    callers have entered. It fails the calls==1 assertion. Initial success and
    provider-outage calls also require the reservation to be returned.
    """
    from recall_mcp import server_auth
    from recall_mcp.limits import FailedAuthThrottle, Rate
    from recall_mcp.oidc import IdentityProviderUnavailable, TokenRejected

    async def scenario():
        gate = asyncio.Event()
        calls = []

        async def offload(fn):
            calls.append(fn)
            await gate.wait()
            return fn()

        class Validator:
            config = SimpleNamespace(issuer="https://issuer.example")

            def validate(self, token):
                if token == "success":
                    return SimpleNamespace(
                        name="agent", tenant="acme", scopes=frozenset(), expires_at=None,
                    )
                if token == "outage":
                    raise IdentityProviderUnavailable("unavailable")
                raise TokenRejected("invalid")

        monkeypatch.setattr(server_auth, "_to_thread", offload)
        throttle = FailedAuthThrottle(Rate(capacity=1.0, per_second=1.0), clock=lambda: 0.0)
        verifier = server_auth.OidcTokenVerifier(Validator(), throttle)
        if initial:
            gate.set()
            first = await verifier.verify_token(initial)
            assert (first is not None) == (initial == "success")
            calls.clear()
            gate.clear()
        tasks = [asyncio.create_task(verifier.verify_token("invalid")) for _ in range(4)]
        await asyncio.sleep(0)
        gate.set()
        results = await asyncio.gather(*tasks)
        assert results == [None] * 4
        assert len(calls) == 1, "concurrent callers bypassed the reserved failure-work budget"
        assert not throttle.allow(), "the admitted invalid token must consume the budget"

    asyncio.run(scenario())
