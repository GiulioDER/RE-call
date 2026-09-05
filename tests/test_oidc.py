"""External OIDC / JWT validation: every rejection the spec names, proved individually.

The tests are organised by ATTACK rather than by code path, because a validator is only as good
as the specific forgeries it refuses. A single "invalid token is rejected" test would pass against
a validator that rejects for the wrong reason, which is indistinguishable from correct until the
day the right reason stops firing.

The signing keys are generated per test session and never leave memory, so nothing here is a
credential.
"""

from __future__ import annotations

import json
import time

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from recall_mcp.auth import SCOPE_ADMIN, SCOPE_READ, SCOPE_WRITE
from recall_mcp.oidc import (
    DEFAULT_CLOCK_SKEW_S,
    DEFAULT_JWKS_REFRESH_S,
    DEFAULT_MAX_STALE_KEY_S,
    OidcConfig,
    OidcValidator,
    TokenRejected,
)

ISSUER = "https://idp.example.com"
AUDIENCE = "recall-api"


@pytest.fixture(scope="module")
def keypair():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return key, key.public_key()


@pytest.fixture(scope="module")
def other_keypair():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return key, key.public_key()


def _jwks(public_key, kid: str = "key-1") -> dict:
    from jwt.algorithms import RSAAlgorithm

    data = json.loads(RSAAlgorithm.to_jwk(public_key))
    data.update({"kid": kid, "use": "sig", "alg": "RS256"})
    return {"keys": [data]}


def _token(private_key, *, kid="key-1", alg="RS256", **overrides) -> str:
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
    for key, value in list(claims.items()):
        if value is _ABSENT:
            del claims[key]
    headers = {} if kid is None else {"kid": kid}
    return jwt.encode(claims, private_key, algorithm=alg, headers=headers)


class _Absent:
    pass


_ABSENT = _Absent()


@pytest.fixture
def validator(keypair):
    _private, public = keypair
    jwks = _jwks(public)
    return OidcValidator(
        OidcConfig(issuer=ISSUER, audience=AUDIENCE),
        _jwks_fetcher=lambda: jwks,
    )


class TestDefaults:
    def test_spec_defaults(self) -> None:
        assert DEFAULT_CLOCK_SKEW_S == 60
        assert DEFAULT_JWKS_REFRESH_S == 15 * 60
        assert DEFAULT_MAX_STALE_KEY_S == 60 * 60
        cfg = OidcConfig(issuer=ISSUER, audience=AUDIENCE)
        assert cfg.clock_skew_s == 60
        assert cfg.jwks_refresh_s == 900
        assert cfg.max_stale_key_s == 3600

    def test_algorithm_allowlist_defaults_to_asymmetric_only(self) -> None:
        cfg = OidcConfig(issuer=ISSUER, audience=AUDIENCE)
        assert all(a.startswith(("RS", "ES", "PS")) for a in cfg.algorithms), cfg.algorithms
        assert not any(a.startswith("HS") for a in cfg.algorithms)
        assert "none" not in {a.lower() for a in cfg.algorithms}


class TestHappyPath:
    def test_a_valid_token_yields_a_principal(self, validator, keypair) -> None:
        private, _ = keypair
        principal = validator.validate(_token(private))
        assert principal.tenant == "acme"
        assert principal.scopes == frozenset({SCOPE_READ, SCOPE_WRITE})

    def test_scope_claim_may_be_a_list(self, validator, keypair) -> None:
        private, _ = keypair
        principal = validator.validate(_token(private, scope=[SCOPE_READ, SCOPE_ADMIN]))
        assert principal.scopes == frozenset({SCOPE_READ, SCOPE_ADMIN})


class TestAlgorithmConfusion:
    """The classic: re-sign with HS256 using the RSA PUBLIC key as the HMAC secret."""

    def test_hs256_signed_with_the_public_key_is_rejected(self, validator, keypair) -> None:
        """Built BY HAND, because PyJWT refuses to encode HS256 with an asymmetric key.

        That refusal protects the signer, not the verifier — an attacker does not call
        `jwt.encode`, they assemble the token themselves. Using PyJWT's guard as the test would
        mean asserting that the attack library declines to help, which says nothing about whether
        OUR validator would have accepted the result.
        """
        import base64
        import hashlib
        import hmac as hmac_mod

        from cryptography.hazmat.primitives import serialization

        _private, public = keypair
        pem = public.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )

        def b64(raw: bytes) -> bytes:
            return base64.urlsafe_b64encode(raw).rstrip(b"=")

        now = int(time.time())
        header = b64(json.dumps({"alg": "HS256", "typ": "JWT", "kid": "key-1"}).encode())
        payload = b64(
            json.dumps(
                {"iss": ISSUER, "aud": AUDIENCE, "exp": now + 300, "tenant": "acme"}
            ).encode()
        )
        signing_input = header + b"." + payload
        # The public key IS the secret. That is the whole attack.
        signature = b64(hmac_mod.new(pem, signing_input, hashlib.sha256).digest())
        forged = (signing_input + b"." + signature).decode()

        with pytest.raises(TokenRejected) as excinfo:
            validator.validate(forged)
        assert excinfo.value.reason == "algorithm_not_allowed"

    def test_alg_none_is_rejected(self, validator) -> None:
        now = int(time.time())
        unsigned = jwt.encode(
            {"iss": ISSUER, "aud": AUDIENCE, "exp": now + 300, "tenant": "acme"},
            key="",
            algorithm="none",
            headers={"kid": "key-1"},
        )
        with pytest.raises(TokenRejected) as excinfo:
            validator.validate(unsigned)
        assert excinfo.value.reason == "algorithm_not_allowed"


class TestKeyIdentification:
    def test_missing_kid_is_rejected(self, validator, keypair) -> None:
        private, _ = keypair
        with pytest.raises(TokenRejected) as excinfo:
            validator.validate(_token(private, kid=None))
        assert excinfo.value.reason == "missing_kid"

    def test_unknown_kid_is_rejected(self, validator, keypair) -> None:
        private, _ = keypair
        with pytest.raises(TokenRejected) as excinfo:
            validator.validate(_token(private, kid="not-a-real-key"))
        assert excinfo.value.reason == "unknown_kid"

    def test_a_token_signed_by_another_key_is_rejected(self, validator, other_keypair) -> None:
        other_private, _ = other_keypair
        with pytest.raises(TokenRejected) as excinfo:
            validator.validate(_token(other_private))
        assert excinfo.value.reason == "bad_signature"


class TestClaims:
    def test_wrong_issuer(self, validator, keypair) -> None:
        private, _ = keypair
        with pytest.raises(TokenRejected) as excinfo:
            validator.validate(_token(private, iss="https://evil.example.com"))
        assert excinfo.value.reason == "bad_issuer"

    def test_wrong_audience(self, validator, keypair) -> None:
        private, _ = keypair
        with pytest.raises(TokenRejected) as excinfo:
            validator.validate(_token(private, aud="some-other-api"))
        assert excinfo.value.reason == "bad_audience"

    def test_expired(self, validator, keypair) -> None:
        private, _ = keypair
        now = int(time.time())
        with pytest.raises(TokenRejected) as excinfo:
            validator.validate(_token(private, exp=now - 3600, iat=now - 7200, nbf=now - 7200))
        assert excinfo.value.reason == "expired"

    def test_not_yet_valid(self, validator, keypair) -> None:
        private, _ = keypair
        now = int(time.time())
        with pytest.raises(TokenRejected) as excinfo:
            validator.validate(_token(private, nbf=now + 3600, exp=now + 7200))
        assert excinfo.value.reason == "not_yet_valid"

    def test_missing_expiry_is_rejected(self, validator, keypair) -> None:
        """A token with no expiry never stops being valid; that is not acceptable."""
        private, _ = keypair
        with pytest.raises(TokenRejected) as excinfo:
            validator.validate(_token(private, exp=_ABSENT))
        assert excinfo.value.reason == "missing_expiry"

    @pytest.mark.parametrize("bad", [_ABSENT, "", "   ", None, 123, []])
    def test_missing_or_empty_tenant_is_rejected(self, validator, keypair, bad) -> None:
        private, _ = keypair
        with pytest.raises(TokenRejected) as excinfo:
            validator.validate(_token(private, tenant=bad))
        assert excinfo.value.reason == "missing_tenant"

    def test_clock_skew_tolerates_a_just_expired_token(self, keypair) -> None:
        private, public = keypair
        v = OidcValidator(
            OidcConfig(issuer=ISSUER, audience=AUDIENCE, clock_skew_s=120),
            _jwks_fetcher=lambda: _jwks(public),
        )
        now = int(time.time())
        assert v.validate(_token(private, exp=now - 30)).tenant == "acme"


class TestTenantComesFromTheTokenOnly:
    def test_the_validator_takes_no_tenant_parameter(self) -> None:
        """Requirement 7, enforced by the SIGNATURE rather than by discipline.

        If `validate` accepted a tenant argument, some caller would eventually pass one from a
        request parameter, and the isolation model would be decided by whoever wrote that call
        site. Not having the parameter makes that unrepresentable.
        """
        import inspect

        params = set(inspect.signature(OidcValidator.validate).parameters)
        assert params == {"self", "token"}, params


class TestJwksRefreshAndStaleWindow:
    def test_keys_are_cached_between_calls(self, keypair) -> None:
        private, public = keypair
        calls = []

        def fetcher():
            calls.append(1)
            return _jwks(public)

        v = OidcValidator(OidcConfig(issuer=ISSUER, audience=AUDIENCE), _jwks_fetcher=fetcher)
        for _ in range(5):
            v.validate(_token(private))
        assert len(calls) == 1, "JWKS refetched per request"

    def test_a_stale_cache_is_served_when_the_idp_is_unreachable(self, keypair) -> None:
        """An IdP blip must not take the fleet down while the keys are still fresh enough."""
        private, public = keypair
        state = {"fail": False, "now": 1000.0}

        def fetcher():
            if state["fail"]:
                raise OSError("idp unreachable")
            return _jwks(public)

        v = OidcValidator(
            OidcConfig(issuer=ISSUER, audience=AUDIENCE),
            _jwks_fetcher=fetcher,
            _clock=lambda: state["now"],
        )
        v.validate(_token(private))
        state["fail"] = True
        state["now"] += DEFAULT_JWKS_REFRESH_S + 1  # refresh is due, and the IdP is down
        assert v.validate(_token(private)).tenant == "acme"

    def test_beyond_the_stale_window_it_fails_closed(self, keypair) -> None:
        private, public = keypair
        state = {"fail": False, "now": 1000.0}

        def fetcher():
            if state["fail"]:
                raise OSError("idp unreachable")
            return _jwks(public)

        v = OidcValidator(
            OidcConfig(issuer=ISSUER, audience=AUDIENCE),
            _jwks_fetcher=fetcher,
            _clock=lambda: state["now"],
        )
        v.validate(_token(private))
        state["fail"] = True
        state["now"] += DEFAULT_MAX_STALE_KEY_S + 1
        with pytest.raises(TokenRejected) as excinfo:
            v.validate(_token(private))
        assert excinfo.value.reason == "jwks_unavailable"

    def test_a_rotation_is_picked_up_without_waiting_the_refresh_interval(
        self, keypair, other_keypair
    ) -> None:
        """An unknown kid triggers an out-of-band refresh, so a new signing key works at once."""
        private, public = keypair
        other_private, other_public = other_keypair
        state = {"rotated": False, "calls": 0}

        def fetcher():
            state["calls"] += 1
            return _jwks(other_public, kid="key-2") if state["rotated"] else _jwks(public)

        v = OidcValidator(OidcConfig(issuer=ISSUER, audience=AUDIENCE), _jwks_fetcher=fetcher)
        v.validate(_token(private))
        assert state["calls"] == 1

        state["rotated"] = True
        assert v.validate(_token(other_private, kid="key-2")).tenant == "acme"
        assert state["calls"] == 2, "a rotation must not wait out the refresh interval"

    def test_unknown_kids_cannot_drive_unbounded_idp_traffic(self, keypair) -> None:
        """The bound is per PROCESS, not per call.

        The original version of this test asserted `calls == before + 1` inside a SINGLE
        validate(), which reads as proving a rate limit while only describing one call. Measured
        against that code, 200 requests with 200 random kids produced 200 JWKS fetches — the exact
        amplification the module's docstring claimed to prevent. Sweeping many kids across
        separate calls is what makes this test able to fail.
        """
        private, public = keypair
        state = {"calls": 0}

        def fetcher():
            state["calls"] += 1
            return _jwks(public)

        v = OidcValidator(OidcConfig(issuer=ISSUER, audience=AUDIENCE), _jwks_fetcher=fetcher)
        v.validate(_token(private))
        baseline = state["calls"]

        for i in range(50):
            with pytest.raises(TokenRejected):
                v.validate(_token(private, kid=f"unknown-{i}"))

        assert state["calls"] - baseline <= 1, (
            f"50 unknown kids caused {state['calls'] - baseline} IdP fetches"
        )

    def test_a_repeated_unknown_kid_is_not_refetched(self, keypair) -> None:
        private, public = keypair
        state = {"calls": 0}

        def fetcher():
            state["calls"] += 1
            return _jwks(public)

        v = OidcValidator(OidcConfig(issuer=ISSUER, audience=AUDIENCE), _jwks_fetcher=fetcher)
        v.validate(_token(private))
        for _ in range(5):
            with pytest.raises(TokenRejected):
                v.validate(_token(private, kid="ghost"))
        assert state["calls"] <= 2

    def test_a_failed_fetch_never_renews_the_stale_window(self, keypair) -> None:
        """The property a single-failure test cannot see.

        A validator that refreshed `fetched_at` on a FAILED fetch would serve a withdrawn key
        forever and never fail closed — and it passed the whole suite, because every temporal test
        advanced the clock once and failed once. This one fails repeatedly, which is what makes it
        able to catch that.
        """
        private, public = keypair
        state = {"fail": False, "now": 1000.0}

        def fetcher():
            if state["fail"]:
                raise OSError("idp unreachable")
            return _jwks(public)

        v = OidcValidator(
            OidcConfig(issuer=ISSUER, audience=AUDIENCE),
            _jwks_fetcher=fetcher,
            _clock=lambda: state["now"],
        )
        v.validate(_token(private))
        state["fail"] = True

        state["now"] += 1000  # inside the window
        assert v.validate(_token(private)).tenant == "acme"
        state["now"] += 1000  # still inside, and a SECOND failure
        assert v.validate(_token(private)).tenant == "acme"
        state["now"] += 2000  # now past 3600s from the last SUCCESSFUL fetch
        with pytest.raises(TokenRejected) as excinfo:
            v.validate(_token(private))
        assert excinfo.value.reason == "jwks_unavailable"


class TestRejectionsCarryNoSecrets:
    def test_the_error_does_not_echo_the_token(self, validator, keypair) -> None:
        private, _ = keypair
        token = _token(private, iss="https://evil.example.com")
        with pytest.raises(TokenRejected) as excinfo:
            validator.validate(token)
        rendered = f"{excinfo.value}|{excinfo.value.reason}"
        assert token not in rendered
        assert token.split(".")[2] not in rendered, "the signature must never be logged"


class TestStaticTokensAreDevelopmentOnly:
    """B.8: the insecure mechanism requires an explicit opt-in, and production removes it."""

    def test_production_refuses_to_load_a_token_file(self, tmp_path, monkeypatch) -> None:
        from recall_mcp.auth import AuthConfigError, load_token_registry

        token_file = tmp_path / "tokens.json"
        token_file.write_text("{}", encoding="utf-8")
        monkeypatch.setenv("RECALL_ENV", "production")
        with pytest.raises(AuthConfigError, match="development-only"):
            load_token_registry(token_file)

    def test_the_refusal_names_the_replacement(self, tmp_path, monkeypatch) -> None:
        """An error that blocks a deploy without naming the fix just moves the outage."""
        from recall_mcp.auth import AuthConfigError, load_token_registry

        token_file = tmp_path / "tokens.json"
        token_file.write_text("{}", encoding="utf-8")
        monkeypatch.setenv("RECALL_ENV", "production")
        with pytest.raises(AuthConfigError) as excinfo:
            load_token_registry(token_file)
        assert "RECALL_OIDC_ISSUER" in str(excinfo.value)

    def test_development_still_loads_them(self, tmp_path, monkeypatch) -> None:
        from recall_mcp.auth import load_token_registry

        token_file = tmp_path / "tokens.json"
        token_file.write_text(
            json.dumps(
                {
                    "principals": [
                        {
                            "name": "local-dev",
                            "tenant": "acme",
                            "token": "d" * 40,
                            "scopes": [SCOPE_READ],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.delenv("RECALL_ENV", raising=False)
        assert load_token_registry(token_file) is not None

    @pytest.mark.parametrize("value", ["prod", "PRODUCTION ", "Production", "development", ""])
    def test_only_the_exact_word_means_production(self, value, monkeypatch) -> None:
        from recall_mcp.auth import is_production

        monkeypatch.setenv("RECALL_ENV", value)
        assert is_production() is (value.strip().lower() == "production")


class TestAdminScope:
    def test_admin_exists_and_is_not_implied_by_write(self) -> None:
        from recall_mcp.auth import ALL_SCOPES, SCOPE_FACT_WRITE, SCOPE_FORGET

        assert SCOPE_ADMIN == "recall:admin"
        assert set(ALL_SCOPES) == {
            SCOPE_READ,
            SCOPE_WRITE,
            SCOPE_FORGET,
            SCOPE_ADMIN,
            SCOPE_FACT_WRITE,
        }

    def test_a_write_token_does_not_get_admin(self, validator, keypair) -> None:
        """Promoting a generation and indexing a document are different blast radii."""
        private, _ = keypair
        principal = validator.validate(_token(private, scope=SCOPE_WRITE))
        assert not principal.has_scope(SCOPE_ADMIN)
