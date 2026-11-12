"""Per-tenant atomic rescue mode, so one process can serve tenants whose generations differ.

`RECALL_ATOMIC_RESCUE_MODE` is process-wide. A process serving a tenant whose generation has a
rescue artifact (mode `active`) beside one whose generation has none fails every search of the
second with `atomic rescue manifest is unreadable`; with `off` for both, the first loses its
production ranking. `RECALL_ATOMIC_RESCUE_MODE_BY_TENANT` names the exceptions.

Red proof, 2026-11-12, one mutation at a time, each restored before the next:

- `tenant_retrieval_environment` returns `dict(values)`: the tenant-mode test and the retrieval
  test failed on `assert 'active' == 'off'`.
- `_retrieve_trusted` builds `values = dict(...)` again: only the retrieval test failed, same assert.
- `_validate_runtime_options` treats the override map as `{}`: both startup tests failed with
  `DID NOT RAISE ValueError`.
- the parser skips the mode check: `test_malformed_entries_are_refused[beta:sometimes]` and the
  malformed-at-startup test failed with `DID NOT RAISE ValueError`.
- the parser stops lower-casing: the parse test failed (`Off` is then refused as malformed).
- the parser allows a repeated tenant: `test_malformed_entries_are_refused[beta:off,beta:active]`
  failed with `DID NOT RAISE ValueError`.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from typing import Any

import pytest

from recall_mcp.retrieval import _retrieve_trusted
from recall_mcp.settings import (
    _validate_runtime_options,
    atomic_rescue_modes_by_tenant,
    tenant_retrieval_environment,
)

NAME = "RECALL_ATOMIC_RESCUE_MODE_BY_TENANT"


def test_entries_are_parsed_and_the_mode_normalised() -> None:
    assert atomic_rescue_modes_by_tenant({NAME: " beta: Off ,alpha:active,"}) == {
        "beta": "off",
        "alpha": "active",
    }
    assert atomic_rescue_modes_by_tenant({}) == {}


@pytest.mark.parametrize("raw", ["beta", "beta:sometimes", ":off", "beta:off,beta:active"])
def test_malformed_entries_are_refused(raw: str) -> None:
    with pytest.raises(ValueError, match=NAME):
        atomic_rescue_modes_by_tenant({NAME: raw})


def test_an_active_override_without_an_artifact_root_is_refused_at_startup() -> None:
    with pytest.raises(ValueError, match="RECALL_ATOMIC_RESCUE_ARTIFACT_ROOT"):
        _validate_runtime_options({NAME: "alpha:active"})


def test_a_malformed_override_is_refused_at_startup() -> None:
    with pytest.raises(ValueError, match=NAME):
        _validate_runtime_options({NAME: "beta:sometimes"})


def test_the_tenant_mode_replaces_the_process_mode_and_others_keep_it() -> None:
    base = {"RECALL_ATOMIC_RESCUE_MODE": "active", NAME: "beta:off"}
    assert tenant_retrieval_environment(base, "beta")["RECALL_ATOMIC_RESCUE_MODE"] == "off"
    assert tenant_retrieval_environment(base, "alpha")["RECALL_ATOMIC_RESCUE_MODE"] == "active"
    assert tenant_retrieval_environment(base, None)["RECALL_ATOMIC_RESCUE_MODE"] == "active"


class _TenantStore:
    def __init__(self, tenant: str) -> None:
        self.tenant = tenant
        self.generation_id = "gen-1"

    @contextmanager
    def snapshot(self) -> Iterator[str]:
        yield self.generation_id


def _mode_seen_by_trust_layer(tenant: str) -> str:
    seen: dict[str, Any] = {}

    def record(store: object, embedder: object, query: str, **kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs["env"])
        return {}

    _retrieve_trusted(
        _TenantStore(tenant),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        "what is the rate limit",
        None,
        5,
        None,
        None,
        env={"RECALL_ATOMIC_RESCUE_MODE": "active", NAME: "beta:off"},
        reranker_builder=lambda *_args, **_kwargs: None,
        admission_factory=lambda _profile: nullcontext(),
        trusted_search_fn=record,
    )
    return str(seen["RECALL_ATOMIC_RESCUE_MODE"])


def test_retrieval_hands_the_trust_layer_the_callers_tenant_mode() -> None:
    assert _mode_seen_by_trust_layer("beta") == "off"
    assert _mode_seen_by_trust_layer("alpha") == "active"
