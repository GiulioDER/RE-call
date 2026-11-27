"""A provider key must never follow a redirect to the host the Location names.

Python's default opener follows a POST 301/302/303 as a GET and copies every header except
Content-Length and Content-Type onto it (`urllib.request.HTTPRedirectHandler.redirect_request`),
so a bearer key sent as an ordinary header reaches whatever host and scheme a 302 points at.
The loopback tests run two real servers: the first answers 302 to the second, and the second
records the Authorization header of everything it receives. The last test is a static guard for
the callers pinned to a fixed provider host, which cannot be pointed at 127.0.0.1.

Red proof, 2026-11-26, against the pre-fix callers on master `e372aeb6e`:

- `test_the_answer_provider_key_does_not_follow_a_redirect`: with
  `recall.answer_provider._OpenRouterClient.chat` calling `request.urlopen`, the redirect was
  followed and failed at `assert target.authorizations == []` with
  `['Bearer secret-answer-key'] == []`.
- `test_the_wizard_catalogue_key_does_not_follow_a_redirect`: with `recall.wizard.llm._fetch`
  building `Request(url, headers=headers)`, the redirect carried the key and it failed at
  `assert target.authorizations == [None]` with `['Bearer secret-catalogue-key'] == [None]`.
- `test_no_configurable_caller_forwards_its_key_on_a_redirect`: with all 13 callers at their
  master versions, it failed at `assert leaked == []` with all 13 names, first
  `'recall.ops.restore_smoke._http_json'`.
- `test_no_urllib_caller_sends_a_key_as_an_ordinary_header`: on the master tree it failed at
  `assert offenders == []` with 23 sites, first `.github/scripts/openrouter_security_scan.py:131`.
  The guard does not trace a header dict passed in by a caller, which is why the three generic
  helpers it cannot see (`restore_smoke._http_json`, `aml_c9_probe.http_json`,
  `aml_mm_scope_stage1._get`) are in the loopback list.
"""

from __future__ import annotations

import ast
import json
import subprocess
import threading
import urllib.error
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from recall import answer_provider
from recall.wizard import llm

_BODY = json.dumps({"choices": [{"message": {"content": "{}"}}], "data": []}).encode()


class _Server:
    def __init__(self, redirect_to: str | None) -> None:
        self.redirect_to = redirect_to
        self.authorizations: list[str | None] = []
        server = self

        class _Handler(BaseHTTPRequestHandler):
            def _answer(self) -> None:
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                server.authorizations.append(self.headers.get("Authorization"))
                if server.redirect_to is not None:
                    self.send_response(302)
                    self.send_header("Location", server.redirect_to)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(_BODY)))
                self.end_headers()
                self.wfile.write(_BODY)

            do_GET = _answer  # noqa: N815 - name fixed by BaseHTTPRequestHandler
            do_POST = _answer  # noqa: N815 - name fixed by BaseHTTPRequestHandler

            def log_message(self, *args: object) -> None:
                """Keep the access log out of the test output."""

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.port = self.httpd.server_port
        self.url = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(
            target=self.httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
        )

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)


@contextmanager
def _redirect_pair(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[_Server, _Server]]:
    # urllib routes 127.0.0.1 through an exported proxy too, which would put the proxy, not
    # these servers, on the other end.
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    target = _Server(redirect_to=None)
    origin = _Server(redirect_to=f"{target.url}/landed")
    target.thread.start()
    origin.thread.start()
    try:
        yield origin, target
    finally:
        origin.close()
        target.close()


def test_the_answer_provider_key_does_not_follow_a_redirect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invariant: the OpenRouter answer client sends its key to the configured endpoint only."""
    with _redirect_pair(monkeypatch) as (origin, target):
        client = answer_provider._OpenRouterClient(
            f"{origin.url}/v1", api_key="secret-answer-key", timeout=10
        )
        status = None
        try:
            client.chat(model="m", messages=[], max_tokens=1, reasoning_effort=None)
        except urllib.error.HTTPError as refused:
            status = refused.code

    assert origin.authorizations == ["Bearer secret-answer-key"]
    assert target.authorizations == []
    assert status == 302


def test_the_wizard_catalogue_key_does_not_follow_a_redirect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invariant: the catalogue fetch may follow a redirect, but its key stays on the first hop."""
    with _redirect_pair(monkeypatch) as (origin, target):
        body = llm._fetch(
            f"{origin.url}/models", headers={"Authorization": "Bearer secret-catalogue-key"}
        )

    assert body == _BODY
    assert origin.authorizations == ["Bearer secret-catalogue-key"]
    assert target.authorizations == [None]


_KEY = "secret-script-key"
_BEARER = f"Bearer {_KEY}"


def _configurable_callers() -> list[tuple[str, Callable[[_Server], object]]]:
    """Every credentialed urllib caller whose URL can be pointed at the loopback origin.

    Imported here rather than at module level, so collecting this file imports no script.
    """
    from benchmarks.beam import aml_c9_probe
    from recall.ops import restore_smoke
    from scripts import (
        aml_c7_qualification,
        aml_c8_graph_openrouter_smoke,
        aml_c9_mechanism_bench,
        aml_hosted_verify,
        aml_mm_scope_stage1,
        aml_multimodal_memeye,
        aml_text_route_smoke,
        aml_x1_stageb,
        run_openrouter_answer_batch,
        run_query_construction_batch,
    )

    auth = {"Authorization": _BEARER}
    return [
        (
            "recall.ops.restore_smoke._http_json",
            lambda o: restore_smoke._http_json(f"{o.url}/mcp", {}, headers=auth),
        ),
        (
            "benchmarks.beam.aml_c9_probe.http_json",
            lambda o: aml_c9_probe.http_json(f"{o.url}/v1", {}, auth, 10),
        ),
        (
            "scripts.aml_mm_scope_stage1._get",
            lambda o: aml_mm_scope_stage1._get(f"{o.url}/credits", headers=auth),
        ),
        (
            "scripts.aml_mm_scope_stage1.call",
            lambda o: aml_mm_scope_stage1.call(o.port, "/v1", {}, _KEY, attempts=1),
        ),
        (
            "scripts.aml_x1_stageb.call",
            lambda o: aml_x1_stageb.call(o.port, "/v1", {}, _KEY, attempts=1),
        ),
        (
            "scripts.aml_multimodal_memeye.JsonClient.call",
            lambda o: aml_multimodal_memeye.JsonClient(
                o.url, _KEY, timeout=10, max_attempts=1
            ).call("/v1", {}),
        ),
        (
            "scripts.aml_c7_qualification.HttpClient.call",
            lambda o: aml_c7_qualification.HttpClient(o.url, _KEY, timeout=10).call("/v1", {}),
        ),
        (
            "scripts.aml_c8_graph_openrouter_smoke.Client.call",
            lambda o: aml_c8_graph_openrouter_smoke.Client(o.url, _KEY).call("/v1", {}),
        ),
        (
            "scripts.aml_c9_mechanism_bench.Client.call",
            lambda o: aml_c9_mechanism_bench.Client(o.url, _KEY).call("/v1", {}, timeout=10),
        ),
        (
            "scripts.aml_text_route_smoke.Client.call",
            lambda o: aml_text_route_smoke.Client(o.url, _KEY).call("/v1", {}),
        ),
        (
            "scripts.aml_hosted_verify.Client.call",
            lambda o: aml_hosted_verify.Client(o.url, _KEY).call("/v1", {}),
        ),
        (
            "scripts.run_openrouter_answer_batch._post_json",
            lambda o: run_openrouter_answer_batch._post_json(
                url=f"{o.url}/v1", api_key=_KEY, payload={}, timeout=10
            ),
        ),
        (
            "scripts.run_query_construction_batch._post_json",
            lambda o: run_query_construction_batch._post_json(
                url=f"{o.url}/v1", api_key=_KEY, payload={}, timeout=10
            ),
        ),
    ]


def test_no_configurable_caller_forwards_its_key_on_a_redirect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invariant: each caller sends its key to the origin and never to the redirect target.

    The callers differ in what they do with the 302 (follow it, retry, return the status, raise on
    the unexpected body), so only what the two servers saw is asserted.
    """
    leaked: list[str] = []
    for name, invoke in _configurable_callers():
        with _redirect_pair(monkeypatch) as (origin, target):
            try:
                invoke(origin)
            except Exception:  # noqa: BLE001 - a caller raising on the 302 has not leaked
                pass
        assert origin.authorizations[:1] == [_BEARER], name
        if any(target.authorizations):
            leaked.append(name)
    assert leaked == []


def _requests_with_a_plain_key(tree: ast.Module) -> list[int]:
    """Line numbers of `Request(...)` calls whose ordinary headers carry Authorization.

    Covers a dict literal passed as headers, and a name bound in the same function to such a
    literal or given the key by subscript. A header dict passed in by a caller is not traced;
    the loopback test covers those helpers.
    """

    def has_key(node: ast.AST) -> bool:
        return isinstance(node, ast.Dict) and any(
            isinstance(k, ast.Constant) and k.value == "Authorization" for k in node.keys
        )

    lines: set[int] = set()
    for scope in ast.walk(tree):
        if not isinstance(scope, ast.Module | ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        keyed: set[str] = set()
        for node in ast.walk(scope):
            if isinstance(node, ast.Assign) and has_key(node.value):
                keyed |= {t.id for t in node.targets if isinstance(t, ast.Name)}
            if (
                isinstance(node, ast.Subscript)
                and isinstance(node.ctx, ast.Store)
                and isinstance(node.value, ast.Name)
                and isinstance(node.slice, ast.Constant)
                and node.slice.value == "Authorization"
            ):
                keyed.add(node.value.id)
        for node in ast.walk(scope):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            called = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if called != "Request":
                continue
            headers = next((kw.value for kw in node.keywords if kw.arg == "headers"), None)
            if headers is None and len(node.args) > 2:
                headers = node.args[2]
            if headers is not None and (
                has_key(headers) or (isinstance(headers, ast.Name) and headers.id in keyed)
            ):
                lines.add(node.lineno)
    return sorted(lines)


def _uses_urllib_request(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(a.name == "urllib.request" for a in node.names):
            return True
        if isinstance(node, ast.ImportFrom) and (
            node.module == "urllib.request"
            or (node.module == "urllib" and any(a.name == "request" for a in node.names))
        ):
            return True
    return False


def _refuses_redirects(tree: ast.Module) -> bool:
    return any(
        isinstance(node, ast.ClassDef)
        and any(ast.unparse(base).endswith("HTTPRedirectHandler") for base in node.bases)
        for node in ast.walk(tree)
    )


def test_no_urllib_caller_sends_a_key_as_an_ordinary_header() -> None:
    """Invariant: in every tracked module that uses urllib.request, a Request never carries
    Authorization in its ordinary headers, unless the module refuses redirects outright.

    This guards the callers pinned to a fixed provider host, which the loopback test cannot point
    at 127.0.0.1, and the next script written by copying one of them.
    """
    root = Path(__file__).resolve().parents[1]
    listed = subprocess.run(
        ["git", "ls-files", "*.py"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.split()
    assert listed, "git ls-files listed nothing, so nothing was checked"
    offenders: list[str] = []
    for relative in listed:
        if relative.startswith("tests/"):
            continue
        tree = ast.parse((root / relative).read_text(encoding="utf-8"))
        if _uses_urllib_request(tree) and not _refuses_redirects(tree):
            offenders += [f"{relative}:{line}" for line in _requests_with_a_plain_key(tree)]
    assert offenders == []
