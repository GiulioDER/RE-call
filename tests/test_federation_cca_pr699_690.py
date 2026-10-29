"""Regression proofs for the CCA audit of PRs 699 to 690 (federation, plan validation, CI triage).

Each test names the finding it pins and the production symbol it targets. Red proof, recorded
2026-10-29 on the remote test host (real pgvector container, three workers): the first eight tests below were run
unchanged against pre-fix ``32616a7fe`` and all failed, then passed on the fixed tree. Seven
failed an assertion (served ids memory-0..3 for 0..4; ``3 == 10``; ``'high' == 'low'``; ``None``
for ``served``; the proof names missing from the schema; two resolves where one was expected).
The two BUG-001 tests failed on the bare ``RuntimeError("primary federation leg did not produce a
trusted retrieval")``, which is the defect itself. The deleted-guard test is a control that the
pre-fix triage also passes; it was proven by mutating ``_guard_hunks`` to collect ``-`` lines
again, which failed it on ``'low' == 'high'``.

- BUG-002 ``recall.federation.federate``: an unfilled rescue slot served one hit fewer.
- BUG-001 ``prepare_federated_execution``: a refused primary leg became a bare RuntimeError, and a
  shadow primary that overran the plan budget failed a request the control path serves.
- NUM-001 ``prepare_federated_execution``: the primary leg was sized by the plan's leg limit, so
  shadow served fewer than the requested ``k``.
- DAT-001 ``federation_diagnostics``: active rescue hits carried no lineage of their own.
- PERF-001 ``StoreRegistry.validate_retrieval_plan``: every request resolved each routed leg's
  calibration on a fresh connection.
- CODE-001 ``_apply_deterministic_guard_triage``: the triage keyed on stale paths and line ranges.
- ENV-001 ``ENVIRONMENT_SCHEMA``: the proof provider's variables were missing from the schema.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import importlib.util
from pathlib import Path
import re
import threading
import time
from types import ModuleType, SimpleNamespace
from typing import Literal, cast

import pytest

from recall.embeddings import HashingEmbedder
from recall.federation import FederationConfig, FederationLeg, federate
from recall.profiles import resolve_retrieval_profile
from recall.retrieval_plan import RetrievalLeg, RetrievalPlan, TenantIdentity
from recall.timing import TimedEmbedder
from recall.types import (
    Chunk,
    Provenance,
    RetrievalDiagnostics,
    StalenessReport,
    TrustedHit,
    TrustedResult,
    Validity,
)
from recall_mcp.federation_adapter import federation_diagnostics, prepare_federated_execution
from recall_mcp.retrieval import _Retrieval
from recall_mcp.settings import ENVIRONMENT_SCHEMA
from recall_mcp.stores import StoreRegistry

NOW = datetime.now(timezone.utc)
ROOT = Path(__file__).resolve().parents[1]


def _hit(tenant: str, index: int) -> TrustedHit:
    chunk_id = f"{tenant}-{index}"
    return TrustedHit(
        chunk=Chunk(chunk_id, f"{tenant}.md", f"{tenant} text {index}"),
        cosine=0.2,
        confidence=0.9,
        verdict="ok",
        provenance=Provenance(f"{tenant}.md", f"{tenant}.md", 1, NOW),
        validity=Validity(None, None, None),
    )


def _result(tenant: str, hits: list[TrustedHit]) -> TrustedResult:
    return TrustedResult(
        query="q",
        hits=hits,
        abstained=not hits,
        reason="",
        gap_warning=False,
        staleness=StalenessReport(False, NOW, timedelta(days=1), timedelta(days=2)),
        diagnostics=RetrievalDiagnostics(embedding_profile="hashing-64"),
        calibration_id=f"{tenant}-calibration",
        calibration_status="certified",
        tenant_id=tenant,
        generation_id=f"{tenant}-generation",
        trust_state="trusted",
    )


def _leg(tenant: str, hits: list[TrustedHit]) -> FederationLeg:
    return FederationLeg(
        tenant_id=tenant,
        generation_id=f"{tenant}-generation",
        embedding_profile="hashing-64",
        calibration_id=f"{tenant}-calibration",
        calibration_status="certified",
        route="route",
        retrieve=lambda _query, _k: _result(tenant, hits),
    )


def test_unfilled_rescue_slot_goes_back_to_the_primary_leg() -> None:
    """BUG-002: with no novel secondary candidate, active mode serves result_k primary hits."""
    primary = [_hit("memory", index) for index in range(10)]
    result = federate(
        "q",
        [_leg("memory", primary), _leg("code", [])],
        FederationConfig(mode="active", result_k=5, primary_prefix=3, rescue_slots=1),
    )
    assert [candidate.hit.chunk.id for candidate in result.served] == [
        f"memory-{index}" for index in range(5)
    ]


class _Registry:
    def _get_federation_store(self, tenant: str, expected_profile: str) -> object:
        return f"store:{tenant}"


def _identity(tenant: str) -> TenantIdentity:
    return TenantIdentity(
        tenant=tenant,
        generation=f"{tenant}-generation",
        embedding_profile="hashing-64",
        calibration=f"{tenant}-calibration",
        calibration_status="certified",
        trust_state="trusted",
        provenance_identity=f"{tenant}:generation",
    )


def _plan(*, latency_budget_ms: int = 1_000) -> RetrievalPlan:
    return RetrievalPlan(
        route_id="memory-specialists",
        primary_tenant="memory",
        allowed_tenants=frozenset({"memory", "code"}),
        rescue_tenant="code",
        primary_limit=3,
        rescue_limit=1,
        max_fanout=2,
        latency_budget_ms=latency_budget_ms,
        selected_legs=(
            RetrievalLeg("primary", "memory", 3, _identity("memory")),
            RetrievalLeg("rescue", "code", 1, _identity("code")),
        ),
    )


def _retrieval(result: TrustedResult) -> _Retrieval:
    return _Retrieval(
        result=result,
        timed=cast(TimedEmbedder, object()),
        profile=resolve_retrieval_profile({}),
        request_started=0.0,
        admission_wait_ms=0.0,
        effective_k=5,
    )


def _execute(
    retrieve: object,
    *,
    mode: Literal["shadow", "active"],
    k: int = 5,
    plan: RetrievalPlan | None = None,
) -> tuple[_Retrieval, object]:
    execution = prepare_federated_execution(
        plan or _plan(),
        config=FederationConfig(mode=mode, max_legs=2, result_k=2, primary_prefix=1, rescue_slots=1),
        registry=_Registry(),  # type: ignore[arg-type]
        current_store="store:memory",
        current_embedder=HashingEmbedder(dim=64),
        env={},
        retrieve_trusted_fn=retrieve,  # type: ignore[arg-type]
    )
    assert execution.retrieve_trusted is not None
    merged = execution.retrieve_trusted(
        execution.store, execution.embedder, "q", None, k, None, None, None, None, {}
    )
    return merged, execution


class _PrimaryRefused(ValueError):
    """Stands in for a typed refusal (TrustRefusal, RetrievalOverloaded) from the primary leg."""


def test_a_refused_primary_leg_keeps_its_own_error_type() -> None:
    """BUG-001: the primary leg's own exception reaches the caller, not a bare RuntimeError."""

    def retrieve(store: object, _embedder: object, _query: str, *_args: object) -> _Retrieval:
        tenant = str(store).split(":", 1)[1]
        if tenant == "memory":
            raise _PrimaryRefused("strict trust refused the primary tenant")
        return _retrieval(_result(tenant, [_hit(tenant, 0)]))

    with pytest.raises(_PrimaryRefused):
        _execute(retrieve, mode="shadow")


def test_a_slow_primary_in_shadow_serves_the_control_result() -> None:
    """BUG-001: shadow observes only, so a primary over the plan budget still serves control."""
    calls = {"memory": 0}
    lock = threading.Lock()

    def retrieve(store: object, _embedder: object, _query: str, *_args: object) -> _Retrieval:
        tenant = str(store).split(":", 1)[1]
        if tenant == "memory":
            with lock:
                calls["memory"] += 1
                first = calls["memory"] == 1
            if first:
                time.sleep(0.5)
        return _retrieval(_result(tenant, [_hit(tenant, 0)]))

    merged, _ = _execute(retrieve, mode="shadow", plan=_plan(latency_budget_ms=50))
    assert [hit.chunk.id for hit in merged.result.hits] == ["memory-0"]


def test_shadow_primary_leg_covers_the_requested_k() -> None:
    """NUM-001: shadow serves the control's k hits even when the plan's primary limit is lower."""

    def retrieve(
        store: object, _embedder: object, _query: str, _source: object, k: int, *_args: object
    ) -> _Retrieval:
        tenant = str(store).split(":", 1)[1]
        return _retrieval(_result(tenant, [_hit(tenant, index) for index in range(k)]))

    merged, _ = _execute(retrieve, mode="shadow", k=10)
    assert len(merged.result.hits) == 10


def test_active_served_hits_report_their_own_tenant_lineage() -> None:
    """DAT-001: a rescued hit served inside the primary result names the tenant it came from."""

    def retrieve(store: object, _embedder: object, _query: str, *_args: object) -> _Retrieval:
        tenant = str(store).split(":", 1)[1]
        return _retrieval(_result(tenant, [_hit(tenant, 0)]))

    _, execution = _execute(retrieve, mode="active")
    diagnostics = federation_diagnostics(execution)  # type: ignore[arg-type]
    assert diagnostics is not None
    served = diagnostics.get("served")
    assert served == [
        {
            "chunk_id": "memory-0",
            "tenant": "memory",
            "generation": "memory-generation",
            "embedding_profile": "hashing-64",
            "calibration": "memory-calibration",
            "leg_rank": 1,
        },
        {
            "chunk_id": "code-0",
            "tenant": "code",
            "generation": "code-generation",
            "embedding_profile": "hashing-64",
            "calibration": "code-calibration",
            "leg_rank": 1,
        },
    ]


def test_plan_validation_reuses_a_certified_calibration(monkeypatch: pytest.MonkeyPatch) -> None:
    """PERF-001: two requests on one route resolve each leg's calibration once, not per request."""
    import recall.calibration_v2 as calibration_v2

    resolved: list[tuple[str, str]] = []

    class _Repository:
        def __init__(self, _dsn: str, tenant: str, *, actor: str) -> None:
            self._tenant = tenant

        def resolve(self, generation_id: str) -> object:
            resolved.append((self._tenant, generation_id))
            return SimpleNamespace(
                status=calibration_v2.CalibrationStatus.CERTIFIED,
                artifact=SimpleNamespace(calibration_id=f"{self._tenant}-calibration"),
            )

    monkeypatch.setattr(calibration_v2, "CalibrationRepository", _Repository)
    registry = object.__new__(StoreRegistry)
    registry._allowed = frozenset({"memory", "code"})  # type: ignore[attr-defined]
    registry._dsn = "postgresql://unused"  # type: ignore[attr-defined]
    registry._lock = threading.Lock()  # type: ignore[attr-defined]
    registry._plan_calibrations = {}  # type: ignore[attr-defined]

    def route(tenant: str) -> object:
        return SimpleNamespace(
            active=SimpleNamespace(
                generation_id=f"{tenant}-generation",
                embedding_profile="hashing-64",
                state="active",
            )
        )

    monkeypatch.setattr(registry, "_route", route)
    plan = _plan()
    registry.validate_retrieval_plan(plan)
    registry.validate_retrieval_plan(plan)
    assert resolved == [("memory", "memory-generation"), ("code", "code-generation")]


def _scan_module() -> ModuleType:
    script = ROOT / ".github" / "scripts" / "openrouter_security_scan.py"
    spec = importlib.util.spec_from_file_location("openrouter_scan_cca_pr699_690", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_guard_triage_follows_the_endpoint_guard_where_it_lives_now() -> None:
    """CODE-001: a HIGH on the moved endpoint guard is downgraded when the diff carries its proofs."""
    scan = _scan_module()
    markers = [
        "parsed_base_url = urlsplit(base_url)",
        "if is_approved_remote(hostname):",
        "elif is_approved_local(hostname):",
        'raise ValueError("base_url hostname is not an approved OpenAI-compatible endpoint")',
    ]
    diff = (
        "diff --git a/recall/embedding_providers/openai_compat.py "
        "b/recall/embedding_providers/openai_compat.py\n"
        "--- a/recall/embedding_providers/openai_compat.py\n"
        "+++ b/recall/embedding_providers/openai_compat.py\n"
        "@@ -120,2 +120,6 @@ class OpenAICompatEmbedder:\n"
        " context\n"
        + "".join(f"+{marker}\n" for marker in markers)
        + " context\n"
    )
    finding = {
        "severity": "high",
        "title": "SSRF",
        "file": "recall/embedding_providers/openai_compat.py",
        "line": 122,
        "description": "Base URL is not validated.",
        "recommendation": "Validate it.",
    }
    triaged = scan._apply_deterministic_guard_triage([finding], diff)
    assert triaged[0]["severity"] == "low"


def test_guard_triage_never_downgrades_a_change_that_deletes_the_guard() -> None:
    """CODE-001 control: a HIGH on a hunk that REMOVES the guard markers stays high."""
    scan = _scan_module()
    markers = [
        "parsed_base_url = urlsplit(base_url)",
        "if is_approved_remote(hostname):",
        "elif is_approved_local(hostname):",
        'raise ValueError("base_url hostname is not an approved OpenAI-compatible endpoint")',
    ]
    diff = (
        "diff --git a/recall/embedding_providers/openai_compat.py "
        "b/recall/embedding_providers/openai_compat.py\n"
        "--- a/recall/embedding_providers/openai_compat.py\n"
        "+++ b/recall/embedding_providers/openai_compat.py\n"
        "@@ -120,6 +120,2 @@ class OpenAICompatEmbedder:\n"
        " context\n"
        + "".join(f"-{marker}\n" for marker in markers)
        + " context\n"
    )
    finding = {
        "severity": "high",
        "title": "SSRF",
        "file": "recall/embedding_providers/openai_compat.py",
        "line": 121,
        "description": "The endpoint allowlist was removed.",
        "recommendation": "Restore it.",
    }
    triaged = scan._apply_deterministic_guard_triage([finding], diff)
    assert triaged[0]["severity"] == "high"


def test_every_proof_provider_variable_is_in_the_environment_schema() -> None:
    """ENV-001: each RECALL_REASONING_PROOF_* name the provider reads is a schema entry."""
    source = (ROOT / "recall" / "proof_provider.py").read_text(encoding="utf-8")
    read = set(re.findall(r'"(RECALL_REASONING_PROOF_[A-Z0-9_]+)"', source))
    assert len(read) == 11
    assert read <= {spec.name for spec in ENVIRONMENT_SCHEMA}
