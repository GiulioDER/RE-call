"""Source-policy clearance and egress for MCP callers come from server-side configuration.

DEF-39 of the deferred pass over the MCP hunt (#1164), owner decision 2026-11-04: neither
verifier ever set the `clearance` or `egress_allowed` claims `_access_context` reads, so every MCP
caller was evaluated as `internal` with no egress and a `clearance` key in the token file was
silently ignored. The values now come from the token file (static tokens) and from
`RECALL_OIDC_SUBJECT_CLEARANCE` keyed on the verified `sub` (OIDC), never from an IdP claim.

Red proofs are recorded in the pull request that adds this file.
"""

from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from recall_mcp.auth import AuthConfigError, TokenRegistry, parse_principals
from recall_mcp.oidc import OidcConfig, OidcValidator, oidc_validator_from_env
from recall_mcp.server_auth import OidcTokenVerifier, RecallTokenVerifier

ISSUER = "https://idp.example.com"
AUDIENCE = "recall-api"
TOKEN = "t" * 40

_THROTTLE = SimpleNamespace(reserve=lambda: True, refund=lambda: None, record_failure=lambda: None)


def _static_claims(entry: dict) -> dict:
    registry = TokenRegistry(parse_principals({"principals": [{"name": "agent", "token": TOKEN, **entry}]}))
    access = asyncio.run(RecallTokenVerifier(registry, _THROTTLE).verify_token(TOKEN))  # type: ignore[arg-type]
    assert access is not None
    return access.claims or {}


def test_a_token_file_clearance_reaches_the_access_token_claims() -> None:
    """Invariant: a principal's configured clearance and egress travel to `_access_context`.

    Failure mode caught: the loader dropped both keys and the verifier emitted only tenant and
    principal, so every static principal was `internal` with no egress.
    """
    claims = _static_claims({"clearance": "confidential", "egress_allowed": True})
    assert (claims.get("clearance"), claims.get("egress_allowed")) == ("confidential", True)


def test_a_token_file_principal_without_a_clearance_keeps_the_policy_default() -> None:
    """Control: an entry without the keys is `internal` with no egress, as before."""
    claims = _static_claims({})
    assert (claims.get("clearance", "internal"), claims.get("egress_allowed", False)) == ("internal", False)


@pytest.mark.parametrize(
    "entry",
    [{"clearance": "secret"}, {"clearance": "CONFIDENTIAL"}, {"egress_allowed": "false"}],
)
def test_an_invalid_token_file_clearance_refuses_to_load(entry: dict) -> None:
    """Invariant: a value the policy cannot interpret refuses at load, it is not read as a default.

    Failure mode caught: an unknown or mistyped key is ignored, so the operator believes a
    principal is restricted (or denied egress) while it runs with whatever the default is.
    """
    with pytest.raises(AuthConfigError):
        parse_principals({"principals": [{"name": "agent", "token": TOKEN, **entry}]})


@pytest.fixture(scope="module")
def keypair():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return key, key.public_key()


def _token(private_key, *, sub: str, **extra) -> str:
    now = int(time.time())
    claims = {"iss": ISSUER, "aud": AUDIENCE, "iat": now, "nbf": now, "exp": now + 300, "sub": sub,
              "tenant": "acme", "scope": "recall:read", **extra}
    return jwt.encode(claims, private_key, algorithm="RS256", headers={"kid": "key-1"})


def _validator(keypair, **config) -> OidcValidator:
    from jwt.algorithms import RSAAlgorithm

    jwk = json.loads(RSAAlgorithm.to_jwk(keypair[1]))
    jwk.update({"kid": "key-1", "use": "sig", "alg": "RS256"})
    return OidcValidator(OidcConfig(issuer=ISSUER, audience=AUDIENCE, **config), _jwks_fetcher=lambda: {"keys": [jwk]})


def test_an_oidc_subject_binding_sets_clearance_and_ignores_the_idp_claim(keypair) -> None:
    """Invariant: OIDC clearance comes from the server-side binding by verified `sub`.

    Failure mode caught: no binding existed, so a bound subject still ran as `internal`; and an
    IdP-supplied `clearance` claim must never be the source.
    """
    validator = _validator(keypair, subject_clearance={"alice": ("confidential", True)})
    bound = validator.validate(_token(keypair[0], sub="alice", clearance="public"))
    assert (bound.clearance, bound.egress_allowed) == ("confidential", True)
    other = validator.validate(_token(keypair[0], sub="bob", clearance="restricted"))
    assert (other.clearance, other.egress_allowed) == ("internal", False)
    access = asyncio.run(OidcTokenVerifier(validator, _THROTTLE).verify_token(  # type: ignore[arg-type]
        _token(keypair[0], sub="alice")
    ))
    assert access is not None and (access.claims or {}).get("clearance") == "confidential"


def test_the_subject_clearance_variable_is_parsed_and_validated() -> None:
    """Invariant: RECALL_OIDC_SUBJECT_CLEARANCE builds the binding and refuses a bad entry.

    Failure mode caught: the variable is not read at all, or a typo in a clearance silently
    falls back to the default.
    """
    base = {"RECALL_OIDC_ISSUER": ISSUER, "RECALL_OIDC_AUDIENCE": AUDIENCE, "RECALL_OIDC_TENANTS": "acme",
            "RECALL_OIDC_TRUST_TENANT_CLAIM": "1"}
    validator = oidc_validator_from_env(
        {**base, "RECALL_OIDC_SUBJECT_CLEARANCE": "system:serviceaccount:ns:sa:restricted+egress,alice:public"}
    )
    assert dict(validator.config.subject_clearance or {}) == {
        "system:serviceaccount:ns:sa": ("restricted", True),
        "alice": ("public", False),
    }
    for bad in ("alice:secret", "alice", "alice:internal+write", "alice:public,alice:internal"):
        with pytest.raises(AuthConfigError):
            oidc_validator_from_env({**base, "RECALL_OIDC_SUBJECT_CLEARANCE": bad})
