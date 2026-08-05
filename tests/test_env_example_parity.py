"""Every `RECALL_*` variable the code reads is documented in `.env.example`.

`.env.example` is this repo's only enumeration of the deployable configuration surface, and it has
drifted from the code twice in two weeks: once when the OIDC block landed, once when the auth-mode
selector did. Both were caught by an auditor reading the file, which is not a mechanism.

The check is a gate rather than a report because the failure is silent by construction: an operator
provisioning from a template that omits a required variable gets a server that refuses to boot, and
nothing in the template tells them why.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ENV_EXAMPLE = ROOT / ".env.example"

#: Scoped to the AUTHENTICATION surface, deliberately, and the scope is stated rather than implied.
#:
#: A repo-wide sweep finds ~16 further pre-existing `RECALL_*` variables absent from the template
#: (logging, HNSW tuning, rerank, S3, pool sizing). Documenting those is real work and unrelated to
#: an auth change, so widening this gate here would either bury that decision in an auth PR or,
#: worse, invite an allowlist so large it stops meaning anything. A gate that covers less and says
#: so beats one whose exceptions have eaten it.
_VAR = re.compile(r"\bRECALL_(?:AUTH|OIDC)_[A-Z0-9_]+\b")

#: Variables in that surface deliberately absent from the template, each with its reason.
UNDOCUMENTED_ON_PURPOSE = {
    # Not a variable at all. `auth.py` names it in prose to state that it does NOT exist: there is
    # deliberately no env var accepting a raw token, because env vars leak through
    # /proc/<pid>/environ, `ps e`, container inspection and crash dumps. Documenting it in the
    # template would create the very thing that docstring exists to refuse.
    "RECALL_AUTH_TOKENS",
}


def _referenced_in_code() -> set[str]:
    found: set[str] = set()
    for package in ("recall", "recall_mcp"):
        for path in (ROOT / package).rglob("*.py"):
            found.update(_VAR.findall(path.read_text(encoding="utf-8")))
    # Line-wrapped prose in comments yields fragments such as `RECALL_AUTH_TOKENS` where the
    # `_FILE` continues on the next line. Keep only names the template could plausibly carry.
    return {v for v in found if not v.endswith("_")}


def test_env_example_documents_every_auth_variable_the_code_reads():
    documented = set(_VAR.findall(ENV_EXAMPLE.read_text(encoding="utf-8")))
    missing = sorted(_referenced_in_code() - documented - UNDOCUMENTED_ON_PURPOSE)
    assert not missing, (
        f"{len(missing)} authentication variable(s) are read by the code but absent from "
        f".env.example: {missing}. Add them there, or to UNDOCUMENTED_ON_PURPOSE in this test "
        f"with the reason they are deliberately undocumented."
    )


def test_the_scanner_actually_reads_the_source():
    """The gate is worthless if `_referenced_in_code` returns nothing, and it would still be green.

    An earlier version of this test did set arithmetic over a hardcoded invented name and never
    called the scanner at all: stubbing `_referenced_in_code` to `return set()` left the whole file
    passing. That is precisely the failure this module was written to prevent, committed inside the
    prevention. So assert the scanner finds a name that is definitely in the source.
    """
    referenced = _referenced_in_code()
    assert "RECALL_OIDC_ISSUER" in referenced
    assert "RECALL_AUTH_TOKENS_FILE" in referenced
    assert len(referenced) >= 5, f"the scanner found suspiciously little: {sorted(referenced)}"


def test_the_parity_check_fails_on_a_genuinely_undocumented_variable(tmp_path, monkeypatch):
    """End to end: an undocumented auth variable in a scanned file must trip the gate.

    Exercises the real scan over a real file rather than simulating the set difference, so a dead
    scanner, a broken regex and a too-greedy allowlist each show up here.
    """
    package = tmp_path / "recall_mcp"
    package.mkdir()
    (package / "planted.py").write_text(
        'CONFIG = os.environ["RECALL_AUTH_UNDOCUMENTED_KNOB"]\n', encoding="utf-8"
    )
    monkeypatch.setattr("tests.test_env_example_parity.ROOT", tmp_path)

    missing = sorted(_referenced_in_code() - UNDOCUMENTED_ON_PURPOSE)
    assert "RECALL_AUTH_UNDOCUMENTED_KNOB" in missing


@pytest.mark.parametrize(
    "required",
    [
        "RECALL_OIDC_ISSUER",
        "RECALL_OIDC_AUDIENCE",
        "RECALL_OIDC_TENANTS",
        "RECALL_OIDC_SUBJECT_TENANTS",
        "RECALL_OIDC_TRUST_TENANT_CLAIM",
        "RECALL_AUTH_MODE",
        "RECALL_AUTH_TOKENS_FILE",
    ],
)
def test_the_authentication_surface_is_documented(required):
    """Named individually as well as swept: these are the ones whose absence stops a boot."""
    assert required in ENV_EXAMPLE.read_text(encoding="utf-8")
