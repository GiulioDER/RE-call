"""STAKES audit regression tests for the OIDC validator.

Two kinds of test live here:

* Plain tests assert an invariant that CURRENTLY HOLDS and must keep holding.
* `xfail(strict=True)` tests assert the invariant the audit says SHOULD hold and that the
  current code does NOT. Strict, so that fixing the finding turns the test red until the
  marker is removed: a test that silently starts passing is not a guard.
"""

from __future__ import annotations

import json
import time

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from jwt.algorithms import ECAlgorithm, RSAAlgorithm

from recall_mcp.auth import SCOPE_READ, SCOPE_WRITE, Principal
from recall_mcp.oidc import OidcConfig, OidcValidator, TokenRejected

ISSUER = "https://idp.example.com"
AUDIENCE = "recall-api"
_ABSENT = object()


@pytest.fixture(scope="module")
def keypair():
    k = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return k, k.public_key()


def _rsa_jwk(public_key, kid="key-1"):
    d = json.loads(RSAAlgorithm.to_jwk(public_key))
    d.update({"kid": kid, "use": "sig", "alg": "RS256"})
    return d


def _token(private_key, *, kid="key-1", alg="RS256", **over):
    now = int(time.time())
    claims = {"iss": ISSUER, "aud": AUDIENCE, "iat": now, "nbf": now, "exp": now + 300,
              "sub": "user-1", "tenant": "acme", "scope": SCOPE_READ + " " + SCOPE_WRITE}
    claims.update(over)
    for k, v in list(claims.items()):
        if v is _ABSENT:
            del claims[k]
    return jwt.encode(claims, private_key, algorithm=alg,
                      headers=({} if kid is None else {"kid": kid}))


@pytest.fixture
def validator(keypair):
    _p, pub = keypair
    return OidcValidator(OidcConfig(issuer=ISSUER, audience=AUDIENCE),
                         _jwks_fetcher=lambda: {"keys": [_rsa_jwk(pub)]})


class TestInvariant4NoScopeIsGrantedByDefault:
    """The `Principal` dataclass defaults `scopes` to `frozenset({SCOPE_READ})`.

    `validate()` always passes `scopes=` explicitly, so that default is NOT reachable from the
    OIDC path. This class pins that, because the default is one omitted keyword away from
    granting read to every token an IdP issues for some other audience entirely.
    """

    def test_a_token_with_no_scope_claim_gets_no_scopes(self, validator, keypair):
        private, _ = keypair
        principal = validator.validate(_token(private, scope=_ABSENT))
        assert principal.scopes == frozenset(), principal.scopes
        assert not principal.has_scope(SCOPE_READ)
        # The default exists and is read; it just must never be what a token gets.
        assert Principal(name="n", tenant="t").scopes == frozenset({SCOPE_READ})

    def test_a_token_whose_scopes_are_all_foreign_gets_no_scopes(self, validator, keypair):
        private, _ = keypair
        principal = validator.validate(_token(private, scope="openid profile email offline_access"))
        assert principal.scopes == frozenset()

    @pytest.mark.parametrize("bad", [None, 0, {}, {"a": 1}, True])
    def test_a_non_string_non_list_scope_claim_gets_no_scopes(self, validator, keypair, bad):
        private, _ = keypair
        assert validator.validate(_token(private, scope=bad)).scopes == frozenset()

    def test_empty_scopes_are_denied_downstream_not_defaulted(self):
        """`authorize` does `required not in (token_scopes or ())`; an empty frozenset is falsy."""
        from recall_mcp.auth import authorize

        for scope in (SCOPE_READ, SCOPE_WRITE):
            with pytest.raises(PermissionError):
                authorize(frozenset(), {"tenant": "acme"}, scope)


class TestInvariant1NoPrincipalWithoutAVerifiedSignature:
    def test_signature_is_verified_before_any_claim_check(self, validator):
        """A wrong-key token missing exp AND tenant must fail on the SIGNATURE, not the claims."""
        other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        with pytest.raises(TokenRejected) as excinfo:
            validator.validate(_token(other, exp=_ABSENT, tenant=_ABSENT))
        assert excinfo.value.reason == "bad_signature"

    def test_validate_has_exactly_one_return(self):
        """Enumerating the exits by hand does not survive an edit; this does."""
        import ast
        import inspect
        import textwrap

        src = inspect.getsource(OidcValidator.validate)
        tree = ast.parse(textwrap.dedent(src))
        returns = [n for n in ast.walk(tree) if isinstance(n, ast.Return)]
        assert len(returns) == 1, f"{len(returns)} returns; every one needs a verified signature"


class TestInvariant2TenantIsExactlyTheClaim:
    def test_a_blank_tenant_is_refused_on_the_stripped_value(self, validator, keypair):
        private, _ = keypair
        with pytest.raises(TokenRejected) as excinfo:
            validator.validate(_token(private, tenant="   \t\n "))
        assert excinfo.value.reason == "missing_tenant"

    def test_an_ambiguous_tenant_is_refused_rather_than_normalised(self, validator, keypair):
        """The audit offered two fixes; this is the one taken, and why.

        `tenant.strip()` collapsed four distinct claims onto one tenant, while
        `auth.parse_principals` stores tenants verbatim and `StoreRegistry` does exact string
        membership — so the same claim meant two different tenants depending on which factory
        built the Principal. Normalising both sides identically would work, but it makes the
        isolation key something two code paths have to keep agreeing about forever.

        Refusing removes the disagreement instead of maintaining it: a tenant claim with
        surrounding whitespace is not silently repaired into a neighbouring identity, it is
        rejected, and the operator fixes the IdP claim.
        """
        private, _ = keypair
        for ambiguous in ("  acme  ", "acme ", " acme", "acme\n"):
            with pytest.raises(TokenRejected) as excinfo:
                validator.validate(_token(private, tenant=ambiguous))
            assert excinfo.value.reason == "malformed_tenant"

    def test_a_clean_tenant_is_passed_through_verbatim(self, validator, keypair):
        private, _ = keypair
        assert validator.validate(_token(private, tenant="acme")).tenant == "acme"


class TestInvariant3ExpiryTravelsWithThePrincipal:
    def test_the_principal_carries_the_token_expiry(self, validator, keypair):
        private, _ = keypair
        exp = int(time.time()) + 300
        principal = validator.validate(_token(private, exp=exp))
        assert principal.expires_at is not None, "is_expired() is False forever"
        assert int(principal.expires_at.timestamp()) == exp


class TestEveryRejectionIsATokenRejected:
    """The module contract: every ambiguity resolves to a rejection with a STABLE reason."""

    def test_a_kid_naming_a_key_of_the_wrong_type_is_a_TokenRejected(self, keypair):
        """Attacker-reachable with NO credential: kids are published in the public JWKS."""
        private, pub = keypair
        eck = ec.generate_private_key(ec.SECP256R1())
        ejwk = json.loads(ECAlgorithm.to_jwk(eck.public_key()))
        ejwk.update({"kid": "ec-1", "use": "sig", "alg": "ES256"})
        v = OidcValidator(OidcConfig(issuer=ISSUER, audience=AUDIENCE),
                          _jwks_fetcher=lambda: {"keys": [_rsa_jwk(pub), ejwk]})
        with pytest.raises(TokenRejected):
            v.validate(_token(private, kid="ec-1"))  # RS256 header, EC key

    def test_a_malformed_jwks_entry_is_a_TokenRejected(self, keypair):
        private, _ = keypair
        v = OidcValidator(OidcConfig(issuer=ISSUER, audience=AUDIENCE),
                          _jwks_fetcher=lambda: {"keys": ["not-a-dict"]})
        with pytest.raises(TokenRejected):
            v.validate(_token(private))


class TestUnknownKidRefreshIsBounded:
    def test_unknown_kids_cannot_drive_unbounded_idp_traffic(self, keypair):
        private, pub = keypair
        calls = {"n": 0}

        def fetcher():
            calls["n"] += 1
            return {"keys": [_rsa_jwk(pub)]}

        v = OidcValidator(OidcConfig(issuer=ISSUER, audience=AUDIENCE), _jwks_fetcher=fetcher)
        v.validate(_token(private))
        base = calls["n"]
        for i in range(20):
            with pytest.raises(TokenRejected):
                v.validate(_token(private, kid="random-" + str(i)))
        assert calls["n"] - base <= 2, (
            f"{calls['n'] - base} IdP fetches from 20 unauthenticated requests"
        )


class TestProductionGateHonoursAnInjectedEnv:
    def test_an_injected_production_env_refuses_static_tokens(self, tmp_path, monkeypatch):
        from recall_mcp.auth import AuthConfigError, token_registry_from_env

        f = tmp_path / "tokens.json"
        f.write_text(json.dumps({"principals": [
            {"name": "n", "tenant": "acme", "token": "d" * 40}]}), encoding="utf-8")
        monkeypatch.delenv("RECALL_ENV", raising=False)
        with pytest.raises(AuthConfigError, match="development-only"):
            token_registry_from_env(
                {"RECALL_ENV": "production", "RECALL_AUTH_TOKENS_FILE": str(f)}
            )
