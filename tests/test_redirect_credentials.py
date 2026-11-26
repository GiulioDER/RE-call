"""A provider key must never follow a redirect to the host the Location names.

Python's default opener follows a POST 301/302/303 as a GET and copies every header except
Content-Length and Content-Type onto it (`urllib.request.HTTPRedirectHandler.redirect_request`),
so a bearer key sent as an ordinary header reaches whatever host and scheme a 302 points at.
Each test runs two real loopback servers: the first answers 302 to the second, and the second
records the Authorization header of everything it receives.

Red proof, 2026-11-26, against the pre-fix callers on master `e372aeb6e`:

- `test_the_answer_provider_key_does_not_follow_a_redirect`: with
  `recall.answer_provider._OpenRouterClient.chat` calling `request.urlopen`, the redirect was
  followed and failed at `assert target.authorizations == []` with
  `['Bearer secret-answer-key'] == []`.
- `test_the_wizard_catalogue_key_does_not_follow_a_redirect`: with `recall.wizard.llm._fetch`
  building `Request(url, headers=headers)`, the redirect carried the key and it failed at
  `assert target.authorizations == [None]` with `['Bearer secret-catalogue-key'] == [None]`.
"""

from __future__ import annotations

import json
import threading
import urllib.error
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

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
        self.url = f"http://127.0.0.1:{self.httpd.server_port}"
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
