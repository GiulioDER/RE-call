"""The shadow embedder is built from the server's settings snapshot, not from `os.environ`.

`bootstrap_settings` resolves API keys from AWS Secrets Manager into `Settings.env` and never
writes them back to the process environment, so a key can exist ONLY in that snapshot. The main
embedder is built with `env=runtime_env`; the shadow embedder `recall_index` builds for a dual-write
generation used to call `make_profile_embedder(profile_id, shadow=True)` with no `env`, which falls
back to `os.environ`. A hosted shadow profile then got no key (or a different one) while the main
embedder worked.

Red proof, recorded per the project rule:

* node: ``tests/test_shadow_embedder_runtime_env.py::test_a_hosted_shadow_profile_gets_the_key_from_the_settings_snapshot``
* baseline: ``recall_mcp/server.py`` at ``fa9d503fb``, where ``recall_index`` calls
  ``make_profile_embedder(profile_id, shadow=True)`` without ``env``.
* failure: the intended assertion, ``assert built_with == ["snapshot-only-key"]``, failed with
  ``[None]``: the real factory read ``VOYAGE_API_KEY`` from the process environment, where it is
  absent.
* green: the same node after the call passes ``env=dict(_runtime_env_for(state))``.

The observation point is the registered profile's own ``build``, reached through the real
``make_profile_embedder`` and ``make_embedder``, so the test sees the key exactly where a hosted
embedder would receive it. Nothing is embedded and no network or database is touched: the service
call is replaced by a stop that records the shadow embedder it was handed.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

import recall_mcp.server as server_module
from recall.embedding_registry import find_registered_profile
from recall.embeddings import HashingEmbedder
from recall_mcp.auth import SCOPE_WRITE
from recall_mcp.server import build_server
from recall_mcp.settings import Settings

_TENANT = "acme"
_PROFILE = "voyage-4-v1"


class _ReachedIndex(Exception):
    def __init__(self, shadow_embedder: object) -> None:
        super().__init__("reached index_memory")
        self.shadow_embedder = shadow_embedder


class _Store:
    tenant = _TENANT


class _Registry:
    """One tenant with a shadow generation on a hosted profile."""

    def __init__(self, dimension: int) -> None:
        shadow = SimpleNamespace(embedding_profile=_PROFILE, dimension=dimension)
        route = SimpleNamespace(shadow=shadow)
        self.control_plane = SimpleNamespace(route=lambda tenant: route)

    def get(self, tenant: str) -> _Store:
        return _Store()

    def get_shadow(self, tenant: str) -> _Store:
        return _Store()


class _FakeEmbedder:
    def __init__(self, dim: int) -> None:
        self.dim = dim


def test_a_hosted_shadow_profile_gets_the_key_from_the_settings_snapshot(monkeypatch) -> None:
    entry = find_registered_profile(_PROFILE)
    assert entry is not None and entry.hosted and entry.api_key_env == "VOYAGE_API_KEY"

    # The key exists only in the snapshot, as it does after a Secrets Manager bootstrap.
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    monkeypatch.delenv("RECALL_EMBED_PROFILE", raising=False)
    settings = Settings.from_env({"VOYAGE_API_KEY": "snapshot-only-key"})

    built_with: list[str | None] = []

    def _build(self, *, api_key=None, env=None, **_kwargs):
        built_with.append(api_key)
        return _FakeEmbedder(self.dimension)

    monkeypatch.setattr(type(entry), "build", _build)

    def _stop(store, embedder, path, **kwargs):
        raise _ReachedIndex(kwargs.get("shadow_embedder"))

    monkeypatch.setattr(server_module, "index_memory", _stop)
    token = SimpleNamespace(scopes=[SCOPE_WRITE], claims={"tenant": _TENANT}, client_id="t")
    monkeypatch.setattr(server_module, "get_access_token", lambda: token)

    state = {
        "embedder": HashingEmbedder(dim=64),
        "store": None,
        "stores": _Registry(entry.dimension),
        "limiter": None,
        "calibration": None,
        "settings": settings,
        "shadow_embedders": {},
        "shadow_embedder_lock": server_module.threading.Lock(),
    }
    tools = {t.name: t for t in build_server(settings)._tool_manager.list_tools()}
    ctx = SimpleNamespace(request_context=SimpleNamespace(lifespan_context=state))

    with pytest.raises(_ReachedIndex) as reached:
        asyncio.run(tools["recall_index"].fn(ctx=ctx, path="notes"))

    assert built_with == ["snapshot-only-key"]
    assert isinstance(reached.value.shadow_embedder, _FakeEmbedder)
