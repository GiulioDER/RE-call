#!/usr/bin/env python3
"""Run the registered mutation sweep over the safety core.

This is intentionally an explicit registry rather than a broad operator generator. Each mutation
names the safety decision it breaks and the tests that must turn red. The source file is mutated in
place for one subprocess only, restored by bytes in ``finally``, and verified against its original
digest before the next mutation.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


@dataclass(frozen=True)
class Mutation:
    module: Path
    label: str
    find: bytes
    replace: bytes
    tests: tuple[str, ...]
    timeout_seconds: int = 240


TRUST_TESTS = (
    "tests/test_trust.py",
    "tests/test_bitemporal.py",
    "tests/test_advice_injection.py",
)
SERVER_TESTS = (
    "tests/test_mcp_tool_authorization.py",
    "tests/test_mcp_auth_wiring.py",
    "tests/test_mcp_trust_refusal.py",
)
SERVER_HTTP_TESTS = ("tests/test_mcp_auth_http.py",)
GENERATION_TESTS = (
    "tests/test_generations.py::test_promotion_is_explicitly_unsafe_and_unavailable_in_production",
    "tests/test_generations.py::test_the_gate_follows_the_serving_environment_not_the_build_one",
    "tests/test_generations.py::test_production_rejects_an_unverified_embedder_identity",
    "tests/test_generations.py::test_gc_retains_two_previous_active_generations_not_failed_builds",
)
PROVENANCE_TESTS = (
    "tests/test_provenance_controller.py",
    "tests/test_provenance_regressions.py",
)


MUTATIONS = (
    Mutation(
        ROOT / "recall/trust.py",
        "transaction time visibility guard is removed",
        b"    if known_as_of is not None and first_known is not None and first_known > known_as_of:",
        b"    if False:",
        TRUST_TESTS,
    ),
    Mutation(
        ROOT / "recall/trust.py",
        "ambiguous supersession is served as ordinary memory",
        b"    if file is not None and file in unresolved:",
        b"    if False:",
        TRUST_TESTS + ("tests/test_trusted_search.py",),
    ),
    Mutation(
        ROOT / "recall/trust.py",
        "superseded hit is treated as current",
        b"    if successor is not None:\n        return \"superseded\", validity",
        b"    if False:\n        return \"superseded\", validity",
        TRUST_TESTS,
    ),
    Mutation(
        ROOT / "recall/trust.py",
        "low confidence threshold guard is disabled",
        b"    if getattr(hit, \"score_kind\", \"dense_cosine\") == \"dense_cosine\" and hit.score < threshold:",
        b"    if False:",
        TRUST_TESTS,
    ),
    Mutation(
        ROOT / "recall/trust.py",
        "calibration is ignored in favour of the uncalibrated fallback",
        b"    cal = calibration or _UNCALIBRATED\n    trusted: list[TrustedHit] = []",
        b"    cal = _UNCALIBRATED\n    trusted: list[TrustedHit] = []",
        TRUST_TESTS,
    ),
    Mutation(
        ROOT / "recall/trust.py",
        "corpus controlled quotes are not stripped before rendering",
        b'''    cleaned = " ".join(cleaned.replace('"', "'").split())''',
        b"    cleaned = \" \".join(cleaned.split())",
        ("tests/test_advice_injection.py",),
    ),
    Mutation(
        ROOT / "recall_mcp/server.py",
        "HTTP server is built without a token verifier",
        b"        token_verifier=verifier,",
        b"        token_verifier=None,",
        SERVER_HTTP_TESTS,
        timeout_seconds=180,
    ),
    Mutation(
        ROOT / "recall_mcp/server.py",
        "requested tenant binding is removed",
        b"        if requested_tenant is not None and requested_tenant != tenant:",
        b"        if False:",
        SERVER_TESTS,
    ),
    Mutation(
        ROOT / "recall_mcp/server.py",
        "scope argument is replaced with read scope",
        b"            tenant = authorize(token.scopes, token.claims, scope)",
        b"            tenant = authorize(token.scopes, token.claims, SCOPE_READ)",
        SERVER_TESTS,
    ),
    Mutation(
        ROOT / "recall_mcp/server.py",
        "per tenant rate limit check is removed",
        b"            limiter.check(tenant, _SCOPE_BUDGETS[scope])",
        b"            pass",
        SERVER_TESTS,
    ),
    Mutation(
        ROOT / "recall_mcp/server.py",
        "authorised tenant is replaced by the process default",
        b"        return registry.get(tenant)\n\n    deps = _ToolDeps",
        b"        return registry.get(TENANT)\n\n    deps = _ToolDeps",
        SERVER_TESTS,
    ),
    Mutation(
        ROOT / "recall/generations.py",
        "production identity enforcement is removed",
        b"        if self.environment == \"production\":\n            pipeline.require_production_identity()",
        b"        if False:\n            pipeline.require_production_identity()",
        GENERATION_TESTS,
    ),
    Mutation(
        ROOT / "recall/generations.py",
        "production calibration gate is removed",
        b"            if self.certification_required:\n                self.require_certified_for_production(generation_id, conn=conn)",
        b"            if False:\n                self.require_certified_for_production(generation_id, conn=conn)",
        GENERATION_TESTS,
    ),
    Mutation(
        ROOT / "recall/generations.py",
        "promotion accepts a non ready generation",
        b"            if target.state != GenerationState.READY:\n                raise InvalidGenerationTransition(\n                    f\"promotion requires ready state, found {target.state.value}\"\n                )",
        b"            if False:\n                raise InvalidGenerationTransition(\n                    f\"promotion requires ready state, found {target.state.value}\"\n                )",
        GENERATION_TESTS,
    ),
    Mutation(
        ROOT / "recall/generations.py",
        "active generation retirement is removed during promotion",
        b"            if active:\n                conn.execute(\n                    \"UPDATE recall_generations SET state = 'retired', \"\n                    \"retired_at = clock_timestamp() \"\n                    \"WHERE tenant_id = %s AND generation_id = %s AND state = 'active'\",\n                    (self.tenant_id, active),\n                )",
        b"            if False:\n                conn.execute(\n                    \"UPDATE recall_generations SET state = 'retired', \"\n                    \"retired_at = clock_timestamp() \"\n                    \"WHERE tenant_id = %s AND generation_id = %s AND state = 'active'\",\n                    (self.tenant_id, active),\n                )",
        ("tests/test_generations.py::test_search_snapshot_sees_one_generation_during_concurrent_promotion",),
        timeout_seconds=300,
    ),
    Mutation(
        ROOT / "recall/generations.py",
        "failed generations are retained by the garbage collector",
        b"                if generation_state == GenerationState.RETIRED.value and (\n                    kept_retired < retain_previous\n                ):",
        b"                if False:",
        ("tests/test_generations.py::test_gc_retains_two_previous_active_generations_not_failed_builds",),
    ),
    Mutation(
        ROOT / "recall/provenance_controller.py",
        "untrusted evidence cards are accepted",
        b"            if card.trust_state != \"trusted\" or card.verdict != \"ok\" or not card.calibrated:",
        b"            if False:",
        PROVENANCE_TESTS,
    ),
    Mutation(
        ROOT / "recall/provenance_controller.py",
        "source digest revalidation is removed",
        b"                if current_digest != card.source_digest:\n                    return DecisionCode.SOURCE_CHANGED",
        b"                if False:\n                    return DecisionCode.SOURCE_CHANGED",
        PROVENANCE_TESTS,
    ),
    Mutation(
        ROOT / "recall/provenance_controller.py",
        "contradictions are accepted without authored supersession",
        b"        if conflicts and len(supersedes) != len(conflicts):",
        b"        if False:",
        PROVENANCE_TESTS,
    ),
    Mutation(
        ROOT / "recall/provenance_controller.py",
        "refusal decisions are marked allowed",
        b"            allowed=code in {DecisionCode.APPLIED, DecisionCode.DUPLICATE},",
        b"            allowed=True,",
        PROVENANCE_TESTS,
    ),
    Mutation(
        ROOT / "recall/provenance_controller.py",
        "materialization failure handling is skipped",
        b"        if self.materializer is not None:\n            try:",
        b"        if False:\n            try:",
        PROVENANCE_TESTS,
    ),
)


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _run(tests: tuple[str, ...], timeout_seconds: int) -> tuple[str, str]:
    result = subprocess.run(
        [PYTHON, "-m", "pytest", "-q", "--disable-warnings", "--maxfail=1", *tests],
        cwd=ROOT,
        env={**os.environ, "PYTHONUTF8": "1"},
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    output = (result.stdout or "") + (result.stderr or "")
    if result.returncode == 0:
        return "green", output
    if "no tests ran" in output.lower() or "collected 0 items" in output.lower():
        return "inconclusive", output
    return "red", output


def main() -> int:
    originals: dict[Path, bytes] = {}
    for mutation in MUTATIONS:
        originals.setdefault(mutation.module, mutation.module.read_bytes())

    baseline: dict[tuple[str, ...], str] = {}
    for tests in dict.fromkeys(mutation.tests for mutation in MUTATIONS):
        try:
            status, output = _run(tests, 360)
        except subprocess.TimeoutExpired as exc:
            print(f"baseline TIMEOUT: {' '.join(tests)}")
            print(str(exc))
            return 2
        baseline[tests] = status
        print(f"baseline {status}: {' '.join(tests)}")
        if status != "green":
            print(output[-4000:])
            print("baseline is not green; mutation results are not interpretable")
            return 2

    killed = survived = inconclusive = stale = 0
    for index, mutation in enumerate(MUTATIONS, 1):
        original = originals[mutation.module]
        if original.count(mutation.find) != 1:
            stale += 1
            print(f"{index:02d} STALE {mutation.module.relative_to(ROOT)}: {mutation.label}")
            continue
        mutant = original.replace(mutation.find, mutation.replace, 1)
        mutation.module.write_bytes(mutant)
        try:
            try:
                status, output = _run(mutation.tests, mutation.timeout_seconds)
            except subprocess.TimeoutExpired:
                status, output = "inconclusive", "pytest timed out"
        finally:
            mutation.module.write_bytes(original)

        restored = mutation.module.read_bytes() == original
        if not restored:
            print(f"{index:02d} RESTORE FAILURE {mutation.module.relative_to(ROOT)}")
            return 3
        if status == "red":
            killed += 1
            print(f"{index:02d} KILLED {mutation.module.relative_to(ROOT)}: {mutation.label}")
        elif status == "green":
            survived += 1
            print(f"{index:02d} SURVIVED {mutation.module.relative_to(ROOT)}: {mutation.label}")
        else:
            inconclusive += 1
            print(f"{index:02d} INCONCLUSIVE {mutation.module.relative_to(ROOT)}: {mutation.label}")
        if status != "green":
            print(output[-1200:])

    print(
        f"summary total={len(MUTATIONS)} killed={killed} survived={survived} "
        f"inconclusive={inconclusive} stale={stale}"
    )
    print("restoration digests:")
    for path, original in originals.items():
        print(f"  {path.relative_to(ROOT)} sha256={_digest(path.read_bytes())} expected={_digest(original)}")
    return 1 if survived or inconclusive or stale else 0


if __name__ == "__main__":
    raise SystemExit(main())
