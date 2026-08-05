"""Wiring the OIDC validator into `build_auth`, and the tenant allowlist that makes it safe.

`recall_mcp/oidc.py` proves a token is genuine. This file covers the two things that stand
between "genuine" and "served": which tenants a genuine token is allowed to name, and how
`build_auth` chooses between the static token file and an identity provider.

The allowlist exists because `StoreRegistry` is built with a fixed `allowed_tenants` set and
documents the property that follows from it — the tenant set is decided by configuration, not by
traffic, so no request can grow the process's pool count. A token-borne tenant would break that,
so the deployment names its tenants and a token may only select from them.

Helpers are defined locally rather than imported from `test_oidc.py` because these two files have
no reason to move together: this one pins deployment wiring, that one pins the validator's refusal
taxonomy, and a shared token factory would make a change to either an edit to both.

(An earlier version of this docstring also claimed the repo avoids cross-module test imports. It
does not — `test_calibration_v2.py` imports helpers straight from `test_generations.py` — and the
coupling argument stands on its own without a false premise propping it up.)
"""

from __future__ import annotations

import json
import threading
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


def test_a_cold_cache_makes_concurrent_callers_wait_not_fail(keypair):
    """BUG-001. Genuine tokens must not be refused while the first JWKS fetch is in flight.

    `_refresh` takes the fetch lock non-blocking and serves the cache to whoever loses, which is
    right when the cache is warm and wrong when it is empty: `{}` is not "what we have". Every
    loser then reports `unknown_kid` for a valid token — a 401 that reads as a forgery while the
    IdP is merely slow, on every restart and every rollout with concurrent traffic.

    Latent until the verifier moved off the event loop: serialised coroutines never raced.
    """
    private, public = keypair
    jwks = jwks_for(public)
    fetches = []

    def slow_fetch() -> dict:
        fetches.append(1)
        time.sleep(0.3)
        return jwks

    validator = OidcValidator(
        OidcConfig(issuer=ISSUER, audience=AUDIENCE, allowed_tenants=frozenset({"acme"})),
        _jwks_fetcher=slow_fetch,
    )
    token = token_for(private)
    results: list[str] = []
    barrier = threading.Barrier(5)

    def run() -> None:
        barrier.wait()
        try:
            results.append(validator.validate(token).tenant)
        except TokenRejected as exc:
            results.append(f"rejected:{exc.reason}")

    threads = [threading.Thread(target=run, daemon=True) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
        assert not t.is_alive(), "a caller blocked forever waiting for the JWKS fetch"

    assert results == ["acme"] * 5, f"a valid token was refused during the first fetch: {results}"
    assert len(fetches) == 1, f"single-flight broken: {len(fetches)} concurrent fetches"


def test_an_allowlist_given_as_a_bare_string_is_refused():
    """BUG-002. `frozenset("acme")` is {"a","c","e","m"}.

    That allowlist refuses the tenant the operator named and admits four one-letter tenants nobody
    provisioned, and it passes the empty and padded checks because single characters are neither.
    """
    with pytest.raises(ValueError, match="not a single string"):
        OidcConfig(issuer=ISSUER, audience=AUDIENCE, allowed_tenants="acme")


def test_provisioned_tenants_given_as_a_bare_string_is_refused():
    """BUG-002, second instance: the same coercion, one layer up, feeding StoreRegistry."""
    from recall_mcp.server import ProvisionedTenants

    with pytest.raises(AuthConfigError, match="not a single string"):
        ProvisionedTenants("acme")


def test_a_trailing_slash_on_the_issuer_does_not_reject_every_token():
    """BUG-003. Discovery rstrips both sides and tolerates it; `jwt.decode` is an exact compare.

    So the un-normalised form booted, discovered its JWKS, and then refused every genuine token as
    `bad_issuer`: a total authentication outage from one trailing character, reported as though
    the IdP were signing the wrong tokens.
    """
    assert OidcConfig(issuer=ISSUER + "/", audience=AUDIENCE).issuer == ISSUER


@pytest.mark.parametrize("bad", ["RS255", "rs256", "EdDSA"])
def test_an_unserviceable_algorithm_is_refused_at_construction(bad):
    """BUG-004. Each boots healthy and then refuses every token at request time.

    `rs256` is compared case-sensitively against the header; `EdDSA` passes the symmetric filter
    while `_usable_keys` loads no OKP key to match it.
    """
    with pytest.raises(ValueError, match="cannot verify"):
        OidcConfig(issuer=ISSUER, audience=AUDIENCE, algorithms=(bad,))


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
    # RECALL_OIDC_TRUST_TENANT_CLAIM is required as of the subject-binding change (SEC-004): a
    # deployment must now say whether the IdP is authoritative for the `tenant` claim, or pin the
    # mapping itself. These tests are about the WIRING, so they take the trust stance explicitly.
    # The refusal when neither is set has its own test in test_oidc_subject_binding.py.
    env = {
        "RECALL_OIDC_ISSUER": ISSUER,
        "RECALL_OIDC_AUDIENCE": AUDIENCE,
        "RECALL_OIDC_TENANTS": "acme,globex",
        "RECALL_OIDC_TRUST_TENANT_CLAIM": "1",
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


@pytest.mark.parametrize(
    "stray",
    [
        "RECALL_OIDC_AUDIENCE",
        "RECALL_OIDC_TENANTS",
        "RECALL_OIDC_ALGORITHMS",
        # Every key added since. The guard's key list is DERIVED from the ENV_* constants rather
        # than hand-written precisely because these two were forgotten when they were added, which
        # reopened the hazard through the newest variable.
        "RECALL_OIDC_SUBJECT_TENANTS",
        "RECALL_OIDC_TRUST_TENANT_CLAIM",
    ],
)
def test_an_oidc_block_without_its_issuer_refuses_to_start(stray, tmp_path):
    """DEPLOY-001. The issuer is the only key that switches OIDC on.

    Misspell it and the other three are never read, so the server boots on static tokens with a
    complete-looking OIDC block doing nothing — the same "sits in the configuration looking
    effective" hazard the both-set conflict refuses, arriving through the one door that guard
    cannot watch. The misspelling is invited: the variable next door is RECALL_AUTH_ISSUER_URL.
    """
    from recall_mcp.server import build_auth

    path = tmp_path / "tokens.json"
    path.write_text(
        json.dumps({"principals": [{"name": "a", "token": "t" * 40, "tenant": "acme"}]}),
        encoding="utf-8",
    )
    env = {
        "RECALL_AUTH_TOKENS_FILE": str(path),
        # RECALL_AUTH_ISSUER_URL is SET on purpose. Without it, `build_auth` raises the unrelated
        # protected-resource-metadata error, whose text also contains the literal string
        # "RECALL_OIDC_ISSUER" — so a `match=` on the bare variable name passed against a build
        # that had stopped enforcing this guard entirely. The test was green for 24 hours while
        # the guard was dead. Match the guard's own sentence, not a substring it shares.
        "RECALL_AUTH_ISSUER_URL": RESOURCE,
        "RECALL_AUTH_RESOURCE_URL": RESOURCE,
        "RECALL_OIDC_ISSUER_URL": ISSUER,  # the typo: _URL does not exist for this variable
        stray: "acme" if stray == "RECALL_OIDC_TENANTS" else "recall-api",
    }
    with pytest.raises(AuthConfigError, match="set without RECALL_OIDC_ISSUER"):
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


@pytest.mark.parametrize(
    "exp",
    [
        1786000000000,  # milliseconds, the units bug an IdP actually ships
        10**12,
        10**19,  # past time_t entirely
    ],
)
def test_an_unrepresentable_expiry_is_a_rejection_not_a_crash(keypair, exp):
    """NUM-001. PyJWT's expiry check is `exp <= now - leeway`, which any far-future value passes.

    So an `exp` in milliseconds is a correctly signed token carrying an expiry that
    `datetime.fromtimestamp` cannot represent. `OSError` and `OverflowError` are not
    `InvalidTokenError`, so this escaped every handler and left `validate` as a 500 on a genuine
    token — falsifying the module's opening contract that every ambiguity becomes a
    `TokenRejected`.
    """
    private, _ = keypair
    validator = make_validator(keypair, allowed_tenants=frozenset({"acme"}))
    with pytest.raises(TokenRejected) as excinfo:
        validator.validate(token_for(private, exp=exp))
    assert excinfo.value.reason == "malformed_expiry"


def test_the_verifier_never_raises_on_a_validator_defect(keypair):
    """NUM-001, defence in depth. The SDK does not wrap `verify_token`.

    Any escape here is a 500 where a 401 belongs, so authentication must fail CLOSED whatever the
    validator does. Proved with a validator that raises something outside the taxonomy entirely.
    """
    from recall_mcp.server import OidcTokenVerifier

    class Exploding:
        config = OidcConfig(issuer=ISSUER, audience=AUDIENCE, allowed_tenants=frozenset({"acme"}))

        def validate(self, token: str):
            raise RuntimeError("a defect the taxonomy does not cover")

    assert verify(OidcTokenVerifier(Exploding()), "any-token") is None


def test_verifying_a_token_does_not_block_the_event_loop(keypair):
    """PERF-001. `validate` is synchronous and its JWKS path does blocking HTTPS IO.

    Called inline from an `async def`, it stalls the whole loop for up to two 10s timeouts, and
    the single-flight `acquire(blocking=False)` that exists to prevent that outage is inert here:
    it lets the other THREADS carry on with cached keys, and under an event loop there are no
    other threads, only coroutines that never get scheduled.

    So this measures the LOOP, not the call. A ticker counts how often it is scheduled while one
    token is verified against a deliberately slow key fetch. Blocked, it stays near zero.
    """
    import asyncio

    from recall_mcp.server import OidcTokenVerifier

    private, public = keypair
    jwks = jwks_for(public)

    def slow_fetch() -> dict:
        time.sleep(0.4)  # a blocking fetch, exactly like urlopen
        return jwks

    validator = OidcValidator(
        OidcConfig(issuer=ISSUER, audience=AUDIENCE, allowed_tenants=frozenset({"acme"})),
        _jwks_fetcher=slow_fetch,
    )
    verifier = OidcTokenVerifier(validator)

    async def scenario():
        ticks = 0
        stop = False

        async def ticker() -> None:
            nonlocal ticks
            while not stop:
                await asyncio.sleep(0.005)
                ticks += 1

        task = asyncio.ensure_future(ticker())
        access = await verifier.verify_token(token_for(private))
        stop = True
        await task
        return access, ticks

    access, ticks = asyncio.run(scenario())
    assert access is not None, "the token itself is valid; this test is about the loop"
    # 0.4s of blocking fetch against a 5ms tick is ~80 opportunities. Ten is far below that and
    # far above the 0-1 a fully blocked loop manages, so it discriminates without being flaky.
    assert ticks >= 10, (
        f"the event loop was blocked during token verification (only {ticks} ticks); "
        "validate() must run off the loop"
    )


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
