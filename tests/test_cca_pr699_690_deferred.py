"""Regression proofs for the deferred pass of the CCA audit of PRs 699 to 690.

Each test names the finding it pins and the production symbol it targets. The red proof for each
behaviour test (run unchanged against the parent fix commit ``286ef349``) is recorded in the pull
request. ``test_retrieval_plan_bounds_fit_inside_federation_bounds`` is a deliberately
nonbehavioural control for CODE-010: both bounds already agree, so it passes before the change and
guards the coupling from now on.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import importlib.util
from pathlib import Path
import threading
from types import ModuleType
from typing import Any, cast

import pytest

from recall.embeddings import HashingEmbedder
from recall.evidence import EvidenceBundle, EvidenceItem
from recall.federation import (
    MAX_FEDERATION_CANDIDATE_K,
    MAX_FEDERATION_LATENCY_BUDGET_MS,
    FederationConfig,
    FederationLeg,
    federate,
)
from recall.profiles import resolve_retrieval_profile
from recall.proof_obligations import parse_proof_assessment, validate_proof_assessment
from recall.proof_provider import resolve_proof_provider
from recall.retrieval_plan import (
    MAX_RETRIEVAL_LATENCY_BUDGET_MS,
    MAX_RETRIEVAL_LEG_LIMIT,
    RetrievalLeg,
    RetrievalPlan,
    TenantIdentity,
)
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
from recall_aml.config import HostedSettings
from recall_aml.storage import PgHostedRepository
from recall_mcp.federation_adapter import prepare_federated_execution
from recall_mcp.retrieval import _Retrieval

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime.now(timezone.utc)


class _NeverWrites:
    table = "recall_chunks"

    def _with_retry(self, op: Any) -> int:
        return 0


def test_multimodal_bundle_refuses_a_repository_with_a_sparse_encoder() -> None:
    """BUG-005 ``PgHostedRepository.persist_multimodal_bundle``: it must not skip the sidecar."""
    repository = object.__new__(PgHostedRepository)
    repository._sparse_encoder = object()  # type: ignore[attr-defined]
    repository._embedder = HashingEmbedder(dim=8)  # type: ignore[attr-defined]
    repository._specialist_embedders = {}  # type: ignore[attr-defined]
    repository._base_store = _NeverWrites()  # type: ignore[attr-defined]
    with pytest.raises(RuntimeError, match="learned sparse sidecar"):
        repository.persist_multimodal_bundle("tenant", [], None, [], [], [])


def _bundle(*items: EvidenceItem) -> EvidenceBundle:
    return EvidenceBundle(
        query="Who owns project alpha and when does it ship?",
        decision="answer",
        reason_code=None,
        decision_state="supported",
        calibrated=True,
        stale=False,
        embedding_profile="test-profile",
        retrieval_profile="test-retrieval",
        index_generation="generation-1",
        items=items,
        trust_state="trusted",
    )


def test_repair_may_name_the_missing_slots_in_any_order() -> None:
    """BUG-007 ``validate_proof_assessment``: the missing slots are a set, not a sequence."""
    item = EvidenceItem(
        chunk_id="one",
        text="Project alpha ships.",
        source="alpha.md",
        ordinal=1,
        indexed_at=None,
        valid_from=None,
        valid_until=None,
        cosine=0.9,
        confidence=0.9,
    )
    payload = {
        "schema_version": 1,
        "slots": [
            {"slot_id": "project", "kind": "entity", "requirement": "project", "required": True},
            {"slot_id": "release_date", "kind": "time", "requirement": "date", "required": True},
            {"slot_id": "owner", "kind": "entity", "requirement": "owner", "required": True},
        ],
        "support": [
            {"slot_id": "project", "chunk_id": "one", "start": 0, "end": 13, "quote": "Project alpha"}
        ],
        "repair": {"query": "alpha owner and release date", "missing_slot_ids": ["owner", "release_date"]},
    }
    decision = validate_proof_assessment(parse_proof_assessment(payload), _bundle(item))
    assert decision.missing_slot_ids == ("release_date", "owner")
    assert decision.sufficient is False


@pytest.mark.parametrize(
    "base_url", ["https://openrouter.ai:443/api/v1", "https://OpenRouter.ai/api/v1"]
)
def test_an_approved_spelling_finds_its_provider_key(
    base_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """BUG-008 ``OpenAICompatEmbedder``: approval and the key lookup use the same normalisation.

    Offline by construction. The constructor probes the provider for its width, so with the real
    ``openai`` SDK installed this test used to call OpenRouter and fail on a 401 (it passed in CI
    only because CI installs no SDK). A fake ``openai`` module replaces the SDK: it records the
    client it was built with and answers the probe itself, so the test asserts the key was found
    rather than inferring it from an ImportError.

    Red: mutation of the key lookup to the raw spelling, ``(parsed_base_url.scheme,
    parsed_base_url.netloc, raw_path)`` in place of ``(scheme, hostname, normalized_path)``, failed
    both cases with ``RuntimeError: OpenAICompatEmbedder requires an explicit api_key for an
    unrecognized base_url``, reported through ``pytest.fail``. Green with ``HTTPS_PROXY`` and
    ``HTTP_PROXY`` pointed at a dead port, so no request left the machine.
    """
    import sys

    from recall.embedding_providers.openai_compat import OpenAICompatEmbedder

    built: dict[str, Any] = {}

    class _FakeOpenAI:
        def __init__(self, *, api_key: str, base_url: str, max_retries: int) -> None:
            built.update(api_key=api_key, base_url=base_url, max_retries=max_retries)
            self.embeddings = self

        def create(self, **_request: Any) -> Any:
            item = type("Item", (), {"embedding": [0.0, 1.0, 0.0]})()
            return type("Response", (), {"data": [item]})()

    fake = ModuleType("openai")
    fake.OpenAI = _FakeOpenAI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", fake)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    try:
        embedder = OpenAICompatEmbedder(base_url=base_url)
    except RuntimeError as exc:
        pytest.fail(f"an approved base_url spelling was refused: {exc}")
    assert built["api_key"] == "test-key"
    assert embedder.dim == 3, "the width probe must be answered by the fake, not the provider"


def test_hosted_user_binding_is_stored_stripped(monkeypatch: pytest.MonkeyPatch) -> None:
    """BUG-009 ``HostedSettings.from_env``: a padded binding must still match its user."""
    monkeypatch.setenv("RECALL_AML_DATABASE_URL", "postgresql://unused")
    monkeypatch.setenv("RECALL_AML_API_KEY", "k")
    monkeypatch.setenv("RECALL_AML_GIT_COMMIT", "c")
    monkeypatch.setenv("RECALL_AML_EMBED_LOCK_PATH", "lock")
    monkeypatch.setenv("RECALL_AML_AUTHORIZED_USER_ID", " alice\r")
    assert HostedSettings.from_env().authorized_user_id == "alice"


def test_a_bad_proof_value_is_reported_under_its_own_name() -> None:
    """ENV-004 ``resolve_proof_provider``: errors name RECALL_REASONING_PROOF_*, not ANSWER."""
    env = {
        "RECALL_REASONING_PROOF_ENABLED": "1",
        "RECALL_REASONING_PROOF_MODEL": "model",
        "RECALL_REASONING_PROOF_REVISION": "rev-1",
        "RECALL_REASONING_PROOF_TIMEOUT": "not-a-number",
        "OPENROUTER_API_KEY": "test-key",
    }
    with pytest.raises(ValueError, match="RECALL_REASONING_PROOF_TIMEOUT"):
        resolve_proof_provider(env)


def _scan_module() -> ModuleType:
    script = ROOT / ".github" / "scripts" / "openrouter_security_scan.py"
    spec = importlib.util.spec_from_file_location("openrouter_scan_cca_deferred", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_mistyped_fail_on_high_switch_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """ENV-005 ``_fail_on_high``: a typo must not read as an unset switch."""
    scan = _scan_module()
    monkeypatch.setenv("OPENROUTER_FAIL_ON_HIGH", "ture")
    with pytest.raises(RuntimeError, match="not a boolean"):
        scan._fail_on_high()
    for raw, expected in (("", False), ("0", False), ("1", True), ("on", True)):
        monkeypatch.setenv("OPENROUTER_FAIL_ON_HIGH", raw)
        assert scan._fail_on_high() is expected


def _hit(tenant: str) -> TrustedHit:
    return TrustedHit(
        chunk=Chunk(f"{tenant}-0", f"{tenant}.md", f"{tenant} text"),
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


@pytest.mark.parametrize(("primary_hits", "expected"), [(0, False), (1, True)])
def test_related_expansion_is_not_seeded_from_a_rescue_hit(primary_hits: int, expected: bool) -> None:
    """DAT-002 ``prepare_federated_execution``: a rescue hit is not in the serving store."""
    plan = RetrievalPlan(
        route_id="memory-specialists",
        primary_tenant="memory",
        allowed_tenants=frozenset({"memory", "code"}),
        rescue_tenant="code",
        primary_limit=3,
        rescue_limit=1,
        max_fanout=2,
        selected_legs=(
            RetrievalLeg("primary", "memory", 3, _identity("memory")),
            RetrievalLeg("rescue", "code", 1, _identity("code")),
        ),
    )

    def retrieve(store: object, _embedder: object, _query: str, *_args: object) -> _Retrieval:
        tenant = str(store).split(":", 1)[1]
        hits = [_hit(tenant)] if tenant == "code" or primary_hits else []
        return _Retrieval(
            result=_result(tenant, hits),
            timed=cast(TimedEmbedder, object()),
            profile=resolve_retrieval_profile({}),
            request_started=0.0,
            admission_wait_ms=0.0,
            effective_k=5,
        )

    execution = prepare_federated_execution(
        plan,
        config=FederationConfig(mode="active", max_legs=2, result_k=2, primary_prefix=1, rescue_slots=1),
        registry=_Registry(),  # type: ignore[arg-type]
        current_store="store:memory",
        current_embedder=HashingEmbedder(dim=64),
        env={},
        retrieve_trusted_fn=retrieve,
    )
    assert execution.retrieve_trusted is not None
    merged = execution.retrieve_trusted(
        execution.store, execution.embedder, "q", None, 5, None, None, None, None, {}
    )
    assert merged.result.hits, "the rescue leg must have been served for this test to discriminate"
    assert getattr(merged, "related_seed_in_store", True) is expected


def test_evidence_proof_runs_on_its_own_bounded_pool() -> None:
    """PERF-003 ``recall_mcp.server``: proof calls must not hold the general worker pool."""
    import recall_mcp.server as server

    limiter = getattr(server, "PROOF_THREAD_LIMITER", None)
    assert limiter is not None and limiter.total_tokens == 4
    source = (ROOT / "recall_mcp" / "server.py").read_text(encoding="utf-8")
    assert "return await _proof_to_thread(" in source


def test_setup_script_names_the_required_binding_before_reading_secrets() -> None:
    """DEPLOY-001, the AML experience setup script: the usage and an early check name the variable."""
    (script,) = (ROOT / "scripts").glob("aml_experience_*_setup.sh")
    source = script.read_text(encoding="utf-8")
    usage = next(line for line in source.splitlines() if "usage:" in line)
    assert "RECALL_AML_EXPERIMENT_AUTHORIZED_USER_ID" in usage
    early = source.index("RECALL_AML_EXPERIMENT_AUTHORIZED_USER_ID is required")
    assert early < source.index("set -a")


def test_runaway_legs_are_capped_by_the_shared_leg_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    """PERF-002 ``recall.federation.federate``: overrunning legs cannot pile up per request."""
    import recall.federation as federation

    pool = ThreadPoolExecutor(max_workers=2)
    monkeypatch.setattr(federation, "_LEG_EXECUTOR", pool, raising=False)
    release = threading.Event()
    lock = threading.Lock()
    counts = {"running": 0, "peak": 0}

    def slow(_query: str, _k: int) -> TrustedResult:
        with lock:
            counts["running"] += 1
            counts["peak"] = max(counts["peak"], counts["running"])
        release.wait(10)
        with lock:
            counts["running"] -= 1
        return _result("memory", [])

    leg = FederationLeg(
        tenant_id="memory",
        generation_id="memory-generation",
        embedding_profile="hashing-64",
        calibration_id="memory-calibration",
        calibration_status="certified",
        route="route",
        retrieve=slow,
    )
    try:
        for _ in range(5):
            result = federate("q", [leg], FederationConfig(mode="shadow"), latency_budget_ms=20)
            assert result.diagnostics.legs[0].rejection_reason == "latency_budget_exceeded"
        assert counts["peak"] <= 2
    finally:
        release.set()
        pool.shutdown(wait=True)


def test_a_leg_waiting_for_its_turn_never_runs_after_the_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PERF-002 ``federate``: a leg beyond max_concurrency is not started once the budget is spent."""
    import recall.federation as federation

    pool = ThreadPoolExecutor(max_workers=4)
    monkeypatch.setattr(federation, "_LEG_EXECUTOR", pool, raising=False)
    release = threading.Event()
    late = threading.Event()

    def slow(_query: str, _k: int) -> TrustedResult:
        release.wait(10)
        return _result("memory", [])

    def second(_query: str, _k: int) -> TrustedResult:
        late.set()
        return _result("code", [])

    def leg(tenant: str, retrieve: Any) -> FederationLeg:
        return FederationLeg(
            tenant_id=tenant,
            generation_id=f"{tenant}-generation",
            embedding_profile="hashing-64",
            calibration_id=f"{tenant}-calibration",
            calibration_status="certified",
            route="route",
            retrieve=retrieve,
        )

    try:
        result = federate(
            "q",
            [leg("memory", slow), leg("code", second)],
            FederationConfig(mode="shadow", max_legs=2, max_concurrency=1),
            latency_budget_ms=20,
        )
        assert [d.rejection_reason for d in result.diagnostics.legs] == [
            "latency_budget_exceeded",
            "latency_budget_exceeded",
        ]
    finally:
        release.set()
        pool.shutdown(wait=True)
    assert not late.is_set(), "a leg that never got a slot ran after the request returned"


def test_no_leg_is_submitted_once_the_budget_has_passed(monkeypatch: pytest.MonkeyPatch) -> None:
    """PERF-002 ``federate``: a completion seen after the deadline must not start the next leg."""
    import recall.federation as federation

    class _CountingPool(ThreadPoolExecutor):
        submitted = 0

        def submit(self, fn: Any, /, *args: Any, **kwargs: Any) -> Any:
            type(self).submitted += 1
            return super().submit(fn, *args, **kwargs)

    pool = _CountingPool(max_workers=4)
    monkeypatch.setattr(federation, "_LEG_EXECUTOR", pool, raising=False)
    first_done = threading.Event()
    late = threading.Event()

    class _Clock:
        """Reads 0 until the first leg has finished, then far past any budget."""

        @staticmethod
        def perf_counter() -> float:
            return 100.0 if first_done.is_set() else 0.0

    monkeypatch.setattr(federation, "time", _Clock)

    def first(_query: str, _k: int) -> TrustedResult:
        first_done.set()
        return _result("memory", [])

    def second(_query: str, _k: int) -> TrustedResult:
        late.set()
        return _result("code", [])

    def leg(tenant: str, retrieve: Any) -> FederationLeg:
        return FederationLeg(
            tenant_id=tenant,
            generation_id=f"{tenant}-generation",
            embedding_profile="hashing-64",
            calibration_id=f"{tenant}-calibration",
            calibration_status="certified",
            route="route",
            retrieve=retrieve,
        )

    try:
        result = federate(
            "q",
            [leg("memory", first), leg("code", second)],
            FederationConfig(mode="shadow", max_legs=2, max_concurrency=1),
            # Long in real time so the first leg always completes inside it; the faked clock
            # alone carries the request past the deadline, which keeps the proof deterministic.
            latency_budget_ms=10_000,
        )
    finally:
        pool.shutdown(wait=True)
    # Counting submissions is the invariant itself: whether a leg submitted late then runs depends
    # on whether a worker picks it up before cancel(), which a test cannot control.
    assert _CountingPool.submitted == 1, "a leg was submitted after the budget had passed"
    assert not late.is_set()
    assert result.diagnostics.legs[1].rejection_reason == "latency_budget_exceeded"


def test_retrieval_plan_bounds_fit_inside_federation_bounds() -> None:
    """CODE-010 control: a valid plan can never be refused by federation at request time."""
    assert MAX_RETRIEVAL_LEG_LIMIT <= MAX_FEDERATION_CANDIDATE_K
    assert MAX_RETRIEVAL_LATENCY_BUDGET_MS <= MAX_FEDERATION_LATENCY_BUDGET_MS
