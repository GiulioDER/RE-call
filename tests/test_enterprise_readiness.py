"""Every failure `check_enterprise_readiness` can report, executed at least once.

The function had no test of any kind. That is how #182 could remove its `calibration` argument
and leave the identity-mismatch failure unreachable from the only caller in the codebase, with a
permanent degraded-readiness warning standing in for a check: `recall/readiness.py`'s calibration
branch read as a gate and was not one.

Each test names one failure string and asserts that string, not merely `ready is False`. A test
that asserts `not result.ready` passes when a completely different branch fires, which on a
function that accumulates failures into a list is not a hypothetical: seed the fixture wrong and
every one of these would still be green.

The fixtures are fakes rather than a database because these are the branches, not the queries.
`tests/test_enterprise_integration.py` runs the same function against real PostgreSQL as an
unprivileged role.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import pytest

from recall.calibration import Calibration
from recall.control_plane import ControlPlaneLedgerState, IndexGeneration, TenantRoute
from recall.embeddings import EmbeddingProfile
from recall.readiness import check_enterprise_readiness

DIM = 8
PROFILE_ID = "bge-small-asymmetric-v1"

GOOD_PROFILE = EmbeddingProfile(
    profile_id=PROFILE_ID,
    model_name="BAAI/bge-small-en-v1.5",
    artifact_digest="a" * 64,
    dimension=DIM,
    query_mode="query_embed",
    passage_mode="passage_embed",
)


class _Embedder:
    def __init__(self, profile: EmbeddingProfile = GOOD_PROFILE, dim: int = DIM) -> None:
        self.profile = profile
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover - never called
        raise AssertionError("readiness must not embed anything")


class _Store:
    tenant = "acme"
    generation_id = "g1"

    def __init__(self, **facts: object) -> None:
        self._facts: dict[str, object] = {
            "rls_enabled": True,
            "indexes_valid": True,
            "dimension": DIM,
            "rows": 10,
            "rows_without_profile": 0,
        }
        self._facts.update(facts)
        self.rls_effective = True
        self.facts_error: Exception | None = None
        self.schema_error: Exception | None = None

    def readiness_facts(self) -> dict[str, object]:
        if self.facts_error is not None:
            raise self.facts_error
        return dict(self._facts)

    def check_rls_effective(self) -> bool:
        return self.rls_effective

    def check_schema(self) -> None:
        if self.schema_error is not None:
            raise self.schema_error


def _generation(**overrides: object) -> IndexGeneration:
    base = IndexGeneration(
        generation_id="g1", physical_table="chunks_g1", embedding_profile=PROFILE_ID,
        dimension=DIM, state="active", chunk_count=10, source_count=2,
        created_at=datetime.now(UTC), ready_at=datetime.now(UTC),
    )
    return replace(base, **overrides)  # type: ignore[arg-type]


#: `None` is a MEANINGFUL route value here (it is the "tenant route is missing" case), so the
#: fake cannot use it as "not supplied". Conflating the two silently turned three tests green
#: against the default route instead of against no route at all.
_UNSET = object()


class _ControlPlane:
    def __init__(self, route: TenantRoute | None | object = _UNSET, **kwargs: object) -> None:
        self._route = (
            TenantRoute("acme", _generation(), None, datetime.now(UTC))
            if route is _UNSET
            else route
        )
        self.route_error: Exception | None = kwargs.get("route_error")  # type: ignore[assignment]
        self.ledger_error: Exception | None = kwargs.get("ledger_error")  # type: ignore[assignment]
        self._ledger = kwargs.get("ledger") or ControlPlaneLedgerState(True, (), (), ())

    def route(self, _tenant: str) -> TenantRoute | None:
        if self.route_error is not None:
            raise self.route_error
        return self._route  # type: ignore[return-value]

    def ledger_state(self) -> ControlPlaneLedgerState:
        if self.ledger_error is not None:
            raise self.ledger_error
        return self._ledger  # type: ignore[return-value]


def _check(store=None, embedder=None, control=None, **kwargs):
    return check_enterprise_readiness(
        store if store is not None else _Store(),
        embedder if embedder is not None else _Embedder(),
        control_plane=control if control is not None else _ControlPlane(),
        **kwargs,
    )


def _fails_with(result, fragment: str) -> None:
    assert not result.ready
    assert any(fragment in failure for failure in result.failures), (
        f"expected a failure containing {fragment!r}; got {result.failures}"
    )


# --------------------------------------------------------------------------------------------
# The green case first, so every red case below is a difference from something known good
# --------------------------------------------------------------------------------------------


def test_a_fully_provisioned_deployment_is_ready_and_not_degraded() -> None:
    """A gate that cannot PASS is as useless as one that cannot fail.

    Every other test in this file asserts a refusal. Without this one they are all satisfied by
    `return ReadinessResult(False, ...)`, and nobody would notice until a correct deployment
    refused to boot.
    """
    certified = Calibration(
        embedder=PROFILE_ID, threshold=0.5, scale=8.0,
        separability=0.97, n_answerable=1000, n_unanswerable=1000,
    )
    assert certified.certified is True, certified.certification_reason
    result = _check(calibration=certified)
    assert result.ready, result.failures
    assert not result.degraded, result.warnings
    assert result.failures == ()


# --------------------------------------------------------------------------------------------
# Model artifacts
# --------------------------------------------------------------------------------------------


def test_an_unpinned_model_artifact_fails() -> None:
    embedder = _Embedder(replace(GOOD_PROFILE, artifact_digest="legacy-unverified"))
    _fails_with(_check(embedder=embedder), "not pinned by an immutable digest")


def test_an_unpinned_model_artifact_is_accepted_only_when_legacy_is_declared() -> None:
    embedder = _Embedder(replace(GOOD_PROFILE, artifact_digest="legacy-unverified"))
    assert _check(embedder=embedder, allow_legacy_profile=True).ready


def test_a_profile_dimension_that_disagrees_with_the_embedder_fails() -> None:
    embedder = _Embedder(replace(GOOD_PROFILE, dimension=DIM * 2))
    _fails_with(_check(embedder=embedder), "embedding profile dimension does not match")


# --------------------------------------------------------------------------------------------
# Generation metadata
# --------------------------------------------------------------------------------------------


def test_a_missing_tenant_route_fails() -> None:
    _fails_with(_check(control=_ControlPlane(route=None)), "tenant route is missing")


def test_a_missing_route_does_not_also_raise_on_the_absent_generation() -> None:
    """The regression that a `route is None` branch invites: reporting, then dereferencing."""
    result = _check(control=_ControlPlane(route=None))
    assert "tenant route is missing" in result.failures


def test_a_store_from_a_different_generation_than_the_route_fails() -> None:
    route = TenantRoute("acme", _generation(generation_id="g2"), None, datetime.now(UTC))
    _fails_with(_check(control=_ControlPlane(route)), "does not match the active tenant generation")


def test_a_generation_profile_that_disagrees_with_the_runtime_fails() -> None:
    route = TenantRoute(
        "acme", _generation(embedding_profile="some-other-profile"), None, datetime.now(UTC)
    )
    _fails_with(_check(control=_ControlPlane(route)), "embedding profile does not match runtime")


def test_a_generation_dimension_that_disagrees_with_the_runtime_fails() -> None:
    route = TenantRoute("acme", _generation(dimension=DIM * 2), None, datetime.now(UTC))
    _fails_with(_check(control=_ControlPlane(route)), "vector dimension does not match runtime")


@pytest.mark.parametrize("state", ["retired", "failed"])
def test_a_retired_or_failed_active_generation_cannot_serve(state: str) -> None:
    route = TenantRoute("acme", _generation(state=state), None, datetime.now(UTC))
    _fails_with(_check(control=_ControlPlane(route)), f"active generation is {state}")


# --------------------------------------------------------------------------------------------
# Indexes, RLS, the physical table
# --------------------------------------------------------------------------------------------


def test_an_invalid_index_fails() -> None:
    _fails_with(_check(store=_Store(indexes_valid=False)), "HNSW or full text index")


def test_rls_disabled_on_the_table_fails() -> None:
    _fails_with(_check(store=_Store(rls_enabled=False)), "row level security is ineffective")


def test_rls_that_the_role_bypasses_fails_even_when_the_table_has_it_enabled() -> None:
    """`relrowsecurity` is not the question; whether it constrains THIS role is.

    A superuser sees `rls_enabled=True` and is not constrained by a single policy, so a check
    that stopped at the table's flag would pass on precisely the deployment that has no
    isolation at all.
    """
    store = _Store()
    store.rls_effective = False
    _fails_with(_check(store=store), "row level security is ineffective")


def test_a_physical_dimension_that_disagrees_with_the_runtime_fails() -> None:
    _fails_with(_check(store=_Store(dimension=DIM * 2)), "physical table vector dimension")


def test_a_populated_legacy_table_without_a_legacy_declaration_fails() -> None:
    store = _Store(rows=100, rows_without_profile=100)
    _fails_with(_check(store=store), "rows without an explicit embedding profile")


def test_a_populated_legacy_table_is_accepted_when_legacy_is_declared() -> None:
    store = _Store(rows=100, rows_without_profile=100)
    assert _check(store=store, allow_legacy_profile=True).ready


def test_an_empty_table_without_profile_metadata_is_not_a_legacy_table() -> None:
    """Zero rows cannot be evidence of an undeclared legacy corpus."""
    assert _check(store=_Store(rows=0, rows_without_profile=0)).ready


# --------------------------------------------------------------------------------------------
# Unreachable database
# --------------------------------------------------------------------------------------------


def test_an_unreachable_database_is_reported_as_unready_rather_than_raised() -> None:
    import psycopg

    store = _Store()
    store.facts_error = psycopg.OperationalError("connection refused")
    result = _check(store=store)
    _fails_with(result, "database readiness query failed")
    assert "OperationalError" in "".join(result.failures)
    assert "connection refused" not in "".join(result.failures), (
        "a readiness failure must not echo a driver message that can carry a DSN"
    )


def test_an_unreachable_control_plane_is_reported_as_unready_rather_than_raised() -> None:
    """This one used to propagate. Startup distinguishes a failed check from a crash."""
    import psycopg

    result = _check(control=_ControlPlane(route_error=psycopg.OperationalError("down")))
    _fails_with(result, "control plane query failed")


def test_a_stale_chunk_table_schema_fails() -> None:
    from recall.schema import SchemaTooOld

    store = _Store()
    store.schema_error = SchemaTooOld("missing 0009")
    _fails_with(_check(store=store), "chunk table schema is not current")


# --------------------------------------------------------------------------------------------
# The SECOND ledger
# --------------------------------------------------------------------------------------------


def test_a_control_plane_ledger_that_is_behind_fails() -> None:
    """`check_schema` covers `recall_schema_migrations`. Nothing covered the other ledger.

    The two are separate by decision (docs/MIGRATIONS.md). Readiness verifying only one is what
    made that separation accidental rather than designed: an enterprise process could boot on a
    control plane whose SQL it had never verified.
    """
    ledger = ControlPlaneLedgerState(
        ledger_present=True, missing=(1,), unknown=(), checksum_mismatches=()
    )
    _fails_with(_check(control=_ControlPlane(ledger=ledger)), "control plane schema is not current")


def test_a_control_plane_ledger_with_a_changed_checksum_fails() -> None:
    ledger = ControlPlaneLedgerState(
        ledger_present=True, missing=(), unknown=(), checksum_mismatches=(1,)
    )
    result = _check(control=_ControlPlane(ledger=ledger))
    _fails_with(result, "checksum mismatch")


def test_a_ledger_ahead_of_the_package_is_degraded_and_says_so() -> None:
    """`ahead` must reach the operator, not just exist as a property.

    `unknown` was made non-fatal so a forward migration followed by an application rollback does
    not refuse every older replica. Doing only that left the condition reported NOWHERE: readiness
    returned ready and not degraded, and the doc claimed it was "reported as degraded". A property
    with no caller is the guard-that-cannot-fire shape this session kept finding, so this test
    exists to keep the wiring, not the property.
    """
    ledger = ControlPlaneLedgerState(
        ledger_present=True, missing=(), unknown=(99,), checksum_mismatches=()
    )
    result = _check(control=_ControlPlane(ledger=ledger))
    assert result.ready, "a database ahead of the package must still serve"
    assert result.degraded, "and the operator must be told it is ahead"
    assert any("AHEAD" in w for w in result.warnings), result.warnings


def test_a_connection_failure_does_not_leak_the_host_or_role() -> None:
    """psycopg puts host, port and role into an OperationalError, and this string reaches a
    startup RuntimeError. Catalog and permission errors keep their detail; connection ones do not.
    """
    import psycopg

    leaky = psycopg.OperationalError(
        'connection to server at "10.0.0.7", port 5432 failed: '
        'FATAL: password authentication failed for user "recall_server"'
    )
    result = _check(control=_ControlPlane(ledger_error=leaky))
    joined = "".join(result.failures)
    assert "control plane ledger query failed: OperationalError" in joined
    assert "10.0.0.7" not in joined and "recall_server" not in joined, joined

    # The other direction, so this is not satisfied by dropping every detail.
    detailed = _check(control=_ControlPlane(ledger_error=RuntimeError("relation X does not exist")))
    assert "relation X does not exist" in "".join(detailed.failures)


def test_an_absent_control_plane_ledger_fails() -> None:
    ledger = ControlPlaneLedgerState(
        ledger_present=False, missing=(1,), unknown=(), checksum_mismatches=()
    )
    _fails_with(_check(control=_ControlPlane(ledger=ledger)), "recall_schema_versions is absent")


def test_a_ledger_query_that_fails_is_a_readiness_failure() -> None:
    import psycopg

    result = _check(control=_ControlPlane(ledger_error=psycopg.OperationalError("down")))
    _fails_with(result, "control plane ledger query failed")


# --------------------------------------------------------------------------------------------
# Calibration: degraded, never silent, never fatal
# --------------------------------------------------------------------------------------------


def test_no_calibration_is_degraded_and_names_abstention() -> None:
    result = _check()
    assert result.ready, "a missing calibration must not stop the process serving"
    assert result.degraded
    assert any("abstention" in w for w in result.warnings), result.warnings


def test_an_uncertified_calibration_is_degraded_and_names_abstention() -> None:
    """The requirement in one test: degraded capability, reported, retrieval untouched."""
    uncertified = Calibration(embedder=PROFILE_ID, threshold=0.5, scale=8.0)
    assert uncertified.certified is not True
    result = _check(calibration=uncertified)
    assert result.ready
    assert result.degraded
    assert any("not certified" in w and "abstention" in w for w in result.warnings), result.warnings


def test_a_calibration_for_a_different_embedder_is_a_failure_not_a_warning() -> None:
    """The branch #182 made unreachable from the server. Reachable, and now executed."""
    mismatched = Calibration(embedder="some-other-model", threshold=0.5, scale=8.0)
    _fails_with(_check(calibration=mismatched), "calibration identity does not match")


def test_readiness_never_modifies_retrieval_state() -> None:
    """"Reports degraded" must not drift into "quietly adjusts the threshold".

    An uncertified calibration is the tempting place to apply a fallback. Nothing here may write
    to the calibration, the embedder or the store, so the check is a report and the retrieval
    path keeps whatever policy it already had.
    """
    calibration = Calibration(embedder=PROFILE_ID, threshold=0.5, scale=8.0)
    embedder = _Embedder()
    store = _Store()
    before = (
        dict(vars(calibration)), dict(vars(embedder)), dict(store.readiness_facts()),
    )

    result = _check(store=store, embedder=embedder, calibration=calibration)

    assert result.degraded
    assert dict(vars(calibration)) == before[0], "readiness mutated the calibration"
    assert dict(vars(embedder)) == before[1], "readiness mutated the embedder"
    assert dict(store.readiness_facts()) == before[2], "readiness mutated the store"


# --------------------------------------------------------------------------------------------
# Accumulation, and the mutation proof
# --------------------------------------------------------------------------------------------


def test_every_failing_condition_is_reported_not_just_the_first() -> None:
    """An operator fixing one failure at a time, per restart, is the reason this matters."""
    store = _Store(rls_enabled=False, indexes_valid=False, dimension=DIM * 2)
    embedder = _Embedder(replace(GOOD_PROFILE, artifact_digest="legacy-unverified"))
    route = TenantRoute("acme", _generation(generation_id="g2"), None, datetime.now(UTC))
    result = _check(store=store, embedder=embedder, control=_ControlPlane(route))
    assert len(result.failures) >= 5, result.failures


def test_the_failure_assertions_here_can_actually_fail() -> None:
    """Mutate the implementation and require the suite's own assertions to go red.

    Reading a config to confirm the config is not a test. This replaces the accumulator with one
    that drops everything, which is what "the check silently stopped checking" looks like, and
    requires that a representative failure assertion notices.
    """
    import recall.readiness as readiness

    real = readiness.check_enterprise_readiness

    def _always_ready(*_args, **_kwargs):
        return readiness.ReadinessResult(ready=True, degraded=False, failures=(), warnings=())

    readiness.check_enterprise_readiness = _always_ready  # type: ignore[assignment]
    try:
        with pytest.raises(AssertionError):
            _fails_with(
                check_enterprise_readiness_indirect(store=_Store(indexes_valid=False)),
                "HNSW or full text index",
            )
    finally:
        readiness.check_enterprise_readiness = real  # type: ignore[assignment]

    # And green again on the real implementation, so the mutation is what moved it.
    _fails_with(_check(store=_Store(indexes_valid=False)), "HNSW or full text index")


def check_enterprise_readiness_indirect(**kwargs):
    """Resolve through the module so the monkeypatch above is visible to the call."""
    import recall.readiness as readiness

    return readiness.check_enterprise_readiness(
        kwargs.pop("store", _Store()),
        kwargs.pop("embedder", _Embedder()),
        control_plane=kwargs.pop("control", _ControlPlane()),
        **kwargs,
    )
