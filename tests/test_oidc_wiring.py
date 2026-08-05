"""Wiring the OIDC validator into `build_auth`, and the tenant allowlist that makes it safe.

`recall_mcp/oidc.py` proves a token is genuine. This file covers the two things that stand
between "genuine" and "served": which tenants a genuine token is allowed to name, and how
`build_auth` chooses between the static token file and an identity provider.

The allowlist exists because `StoreRegistry` is built with a fixed `allowed_tenants` set and
documents the property that follows from it — the tenant set is decided by configuration, not by
traffic, so no request can grow the process's pool count. A token-borne tenant would break that,
so the deployment names its tenants and a token may only select from them.

Helpers are defined locally rather than imported from `test_oidc.py`: these two files have no
reason to move together, and the repo already avoids cross-module test imports for that reason.
"""

from __future__ import annotations

import json
import time

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from recall_mcp.auth import SCOPE_READ, SCOPE_WRITE, AuthConfigError
from recall_mcp.oidc import OidcConfig, OidcValidator, TokenRejected

ISSUER = "https://idp.example.com"
AUDIENCE = "recall-api"
RESOURCE = "https://recall.example.com"


@pytest.fixture(scope="module")
def keypair():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return key, key.public_key()


def jwks_for(public_key, kid: str = "key-1") -> dict:
    from jwt.algorithms import RSAAlgorithm

    data = json.loads(RSAAlgorithm.to_jwk(public_key))
    data.update({"kid": kid, "use": "sig", "alg": "RS256"})
    return {"keys": [data]}


def token_for(private_key, *, kid="key-1", **overrides) -> str:
    now = int(time.time())
    claims = {
        "iss": ISSUER,
        "aud": AUDIENCE,
        "iat": now,
        "nbf": now,
        "exp": now + 300,
        "sub": "user-1",
        "tenant": "acme",
        "scope": f"{SCOPE_READ} {SCOPE_WRITE}",
    }
    claims.update(overrides)
    return jwt.encode(claims, private_key, algorithm="RS256", headers={"kid": kid})


def make_validator(keypair, **config_kwargs) -> OidcValidator:
    _private, public = keypair
    jwks = jwks_for(public)
    return OidcValidator(
        OidcConfig(issuer=ISSUER, audience=AUDIENCE, **config_kwargs),
        _jwks_fetcher=lambda: jwks,
    )


# ------------------------------------------------------------------ the tenant allowlist


def test_a_tenant_outside_the_allowlist_is_refused(keypair):
    """The whole point of the allowlist: the IdP vouches for identity, not for our topology.

    An IdP that issues `tenant: initech` is not wrong — it simply knows nothing about which
    tenants this deployment provisioned. Admitting it would open a store for a namespace no
    operator configured.
    """
    private, _ = keypair
    validator = make_validator(keypair, allowed_tenants=frozenset({"acme", "globex"}))
    with pytest.raises(TokenRejected) as excinfo:
        validator.validate(token_for(private, tenant="initech"))
    assert excinfo.value.reason == "tenant_not_allowed"


def test_a_tenant_inside_the_allowlist_is_admitted(keypair):
    private, _ = keypair
    validator = make_validator(keypair, allowed_tenants=frozenset({"acme", "globex"}))
    principal = validator.validate(token_for(private, tenant="globex"))
    assert principal.tenant == "globex"


def test_no_allowlist_admits_any_tenant(keypair):
    """Unset stays permissive, so the validator is still usable on its own terms.

    `build_auth` is what refuses to *deploy* without one; that is a wiring decision and it is
    tested as one below. Baking it in here would make the validator untestable in isolation.
    """
    private, _ = keypair
    validator = make_validator(keypair)
    assert validator.validate(token_for(private, tenant="anything")).tenant == "anything"


def test_the_allowlist_is_checked_after_the_signature(keypair, other_key):
    """Otherwise the allowlist is a tenant-enumeration oracle for an UNAUTHENTICATED caller.

    If a forged token for `initech` came back `tenant_not_allowed` while one for `acme` came back
    `bad_signature`, the difference in reply would enumerate the deployment's tenant list to
    somebody holding no credential at all. The signature must be the first thing that fails.
    """
    validator = make_validator(keypair, allowed_tenants=frozenset({"acme"}))
    forged = token_for(other_key, tenant="initech")
    with pytest.raises(TokenRejected) as excinfo:
        validator.validate(forged)
    assert excinfo.value.reason == "bad_signature", (
        "a forged token must fail on its signature, not reveal whether its tenant exists"
    )


@pytest.fixture(scope="module")
def other_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def test_an_empty_allowlist_is_refused_at_construction():
    """An allowlist that permits nothing rejects every token: an outage, not a policy.

    Mirrors the existing treatment of an empty `algorithms` tuple.
    """
    with pytest.raises(ValueError, match="allowed_tenants"):
        OidcConfig(issuer=ISSUER, audience=AUDIENCE, allowed_tenants=frozenset())


@pytest.mark.parametrize("bad", [" acme", "acme ", "", "   "])
def test_an_allowlist_entry_that_could_never_match_is_refused(bad):
    """`validate` rejects a padded tenant claim rather than trimming it, so a padded entry here
    could never match any token. Silently un-matchable configuration is how a tenant appears
    provisioned and is refused at every request.
    """
    with pytest.raises(ValueError):
        OidcConfig(issuer=ISSUER, audience=AUDIENCE, allowed_tenants=frozenset({"acme", bad}))


# ------------------------------------------------------------------ build_auth: choosing a mode


def oidc_env(**extra) -> dict:
    env = {
        "RECALL_OIDC_ISSUER": ISSUER,
        "RECALL_OIDC_AUDIENCE": AUDIENCE,
        "RECALL_OIDC_TENANTS": "acme,globex",
        "RECALL_AUTH_RESOURCE_URL": RESOURCE,
    }
    env.update(extra)
    return env


def test_oidc_http_transport_yields_a_verifier_and_the_allowlisted_tenants():
    from recall_mcp.server import OidcTokenVerifier, build_auth

    verifier, settings, provisioning = build_auth("streamable-http", env=oidc_env())
    assert isinstance(verifier, OidcTokenVerifier)
    assert settings is not None
    assert provisioning.tenants == frozenset({"acme", "globex"})


def test_building_oidc_auth_contacts_no_network(monkeypatch):
    """Discovery is lazy. If construction fetched, an IdP blip would stop the server booting —
    the same coupling the stale-key window exists to avoid, moved to startup.

    The fake RECORDS rather than raises. Raising looks stricter and is in fact weaker here:
    `discover_jwks_uri` catches broad and re-raises as `IdentityProviderUnavailable`, so the
    assertion message is swallowed and the failure arrives disguised as an IdP outage. Verified by
    mutation — made eager, this fails on the recorded call with the message it was given.
    """
    import recall_mcp.oidc as oidc_module
    from recall_mcp.server import build_auth

    calls: list[str] = []
    # A WELL-FORMED discovery document, so nothing downstream refuses it for its own reasons. A
    # fake that returns junk fails the mutant on the junk instead of on the fetch, which reads as
    # the same red but proves a different thing.
    document = json.dumps({"issuer": ISSUER, "jwks_uri": f"{ISSUER}/jwks"}).encode()

    def record(url: str) -> bytes:
        calls.append(url)
        return document

    monkeypatch.setattr(oidc_module, "_http_get", record)

    build_auth("streamable-http", env=oidc_env())
    assert calls == [], f"build_auth must not perform network IO, but fetched {calls}"


def test_oidc_without_a_tenant_allowlist_refuses_to_start():
    """Absent is not "all tenants" — it is the same shape of mistake as an empty tenant claim."""
    from recall_mcp.server import build_auth

    env = oidc_env()
    del env["RECALL_OIDC_TENANTS"]
    with pytest.raises(AuthConfigError, match="RECALL_OIDC_TENANTS"):
        build_auth("streamable-http", env=env)


def test_oidc_and_static_tokens_together_refuse_to_start(tmp_path):
    """Two mechanisms with different trust models, and no stated precedence.

    Whichever won, the other would sit in the configuration looking effective. A static file that
    an operator believes is being enforced is worse than one they know is not.
    """
    from recall_mcp.server import build_auth

    path = tmp_path / "tokens.json"
    path.write_text(
        json.dumps({"principals": [{"name": "a", "token": "t" * 40, "tenant": "acme"}]}),
        encoding="utf-8",
    )
    env = oidc_env(RECALL_AUTH_TOKENS_FILE=str(path))
    with pytest.raises(AuthConfigError, match="both"):
        build_auth("streamable-http", env=env)


def test_the_metadata_issuer_defaults_to_the_identity_provider():
    """With an IdP there is exactly one right answer, and making an operator restate it is a
    chance to state it differently.
    """
    from recall_mcp.server import build_auth

    _verifier, settings, _prov = build_auth("streamable-http", env=oidc_env())
    assert str(settings.issuer_url).rstrip("/") == ISSUER


def test_an_explicit_metadata_issuer_still_wins():
    from recall_mcp.server import build_auth

    other = "https://gateway.example.com"
    _v, settings, _p = build_auth(
        "streamable-http", env=oidc_env(RECALL_AUTH_ISSUER_URL=other)
    )
    assert str(settings.issuer_url).rstrip("/") == other


def test_oidc_still_needs_the_resource_url():
    from recall_mcp.server import build_auth

    env = oidc_env()
    del env["RECALL_AUTH_RESOURCE_URL"]
    with pytest.raises(AuthConfigError, match="RECALL_AUTH_RESOURCE_URL"):
        build_auth("streamable-http", env=env)


def test_stdio_with_oidc_configured_warns_that_it_is_unused(caplog):
    from recall_mcp.server import build_auth

    with caplog.at_level("WARNING"):
        result = build_auth("stdio", env=oidc_env())
    assert result == (None, None, None)
    assert any("unused" in r.getMessage() for r in caplog.records)


def test_the_no_auth_error_names_both_mechanisms():
    """An operator who has never seen this repo needs to know OIDC is an option at all."""
    from recall_mcp.server import build_auth

    with pytest.raises(AuthConfigError, match="RECALL_OIDC_ISSUER"):
        build_auth("streamable-http", env={})


# ------------------------------------------------------------------ the verifier adapter


def verify(verifier, token):
    import asyncio

    return asyncio.run(verifier.verify_token(token))


def test_the_verifier_maps_a_valid_token_to_scopes_a_tenant_and_an_expiry(keypair):
    from recall_mcp.server import OidcTokenVerifier

    private, _ = keypair
    verifier = OidcTokenVerifier(make_validator(keypair, allowed_tenants=frozenset({"acme"})))
    access = verify(verifier, token_for(private))
    assert access is not None
    assert access.claims["tenant"] == "acme"
    assert set(access.scopes) == {SCOPE_READ, SCOPE_WRITE}
    # Carried, not dropped: without it a five-minute JWT becomes a credential the SDK believes
    # never expires.
    assert access.expires_at is not None


def test_the_verifier_returns_none_without_logging_the_token(keypair, other_key, caplog):
    from recall_mcp.server import OidcTokenVerifier

    verifier = OidcTokenVerifier(make_validator(keypair, allowed_tenants=frozenset({"acme"})))
    forged = token_for(other_key)
    with caplog.at_level("WARNING"):
        assert verify(verifier, forged) is None
    assert caplog.records, "a rejection should be logged"
    assert all(forged[:12] not in r.getMessage() for r in caplog.records)


def test_the_rejection_reason_reaches_the_log(keypair, other_key, caplog):
    """The taxonomy is the API. Collapsing it to "invalid token" at the last hop throws away the
    distinction between a forgery and a rotation this process has not caught up with.
    """
    from recall_mcp.server import OidcTokenVerifier

    verifier = OidcTokenVerifier(make_validator(keypair, allowed_tenants=frozenset({"acme"})))
    with caplog.at_level("WARNING"):
        verify(verifier, token_for(other_key))
    assert any("bad_signature" in r.getMessage() for r in caplog.records)


def test_an_unavailable_idp_is_logged_as_an_outage_not_a_forgery(keypair, caplog):
    """`verify_token` can only say yes or no, so the 503/401 split cannot survive this hop.

    It must at least survive into the log, because "the IdP is down" and "someone is forging
    tokens" call for opposite responses and both arrive here as a refusal.
    """
    from recall_mcp.oidc import IdentityProviderUnavailable
    from recall_mcp.server import OidcTokenVerifier

    private, _ = keypair

    def unavailable() -> dict:
        raise IdentityProviderUnavailable("idp_unavailable", "connection refused")

    validator = OidcValidator(
        OidcConfig(issuer=ISSUER, audience=AUDIENCE, allowed_tenants=frozenset({"acme"})),
        _jwks_fetcher=unavailable,
    )
    with caplog.at_level("WARNING"):
        assert verify(OidcTokenVerifier(validator), token_for(private)) is None
    assert any("identity provider" in r.getMessage().lower() for r in caplog.records)
