"""Subject-to-tenant binding (SEC-004), and the auth-mode selector that makes a cutover staged.

Two things the OIDC wiring shipped without.

**SEC-004.** The tenant allowlist bounds WHICH tenants exist. It does not bind WHO may name one.
Nothing correlated `sub` with `tenant`, so any subject able to obtain a token from the issuer with
the right audience reached whichever provisioned tenant its claim named. The allowlist is not
wrong, it just answers a different question, and the gap is invisible precisely because every
individual check passes.

**The cutover.** Refusing both mechanisms at once is right when the operator did not choose, and
it also made the static-to-OIDC transition atomic and un-canaryable: the intermediate state of a
two-step rollout would not boot, and because the conflict is checked before the transport branch,
stdio processes sharing the environment file died with it. `RECALL_AUTH_MODE` lets precedence be
DECLARED instead of refused, so the ambiguity guard survives for the case it was written for.
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


def token_for(private_key, **overrides) -> str:
    now = int(time.time())
    claims = {
        "iss": ISSUER,
        "aud": AUDIENCE,
        "iat": now,
        "nbf": now,
        "exp": now + 300,
        "sub": "alice@corp",
        "tenant": "acme",
        "scope": f"{SCOPE_READ} {SCOPE_WRITE}",
    }
    claims.update(overrides)
    return jwt.encode(claims, private_key, algorithm="RS256", headers={"kid": "key-1"})


def make_validator(keypair, **config_kwargs) -> OidcValidator:
    _private, public = keypair
    jwks = jwks_for(public)
    config_kwargs.setdefault("allowed_tenants", frozenset({"acme", "globex"}))
    return OidcValidator(
        OidcConfig(issuer=ISSUER, audience=AUDIENCE, **config_kwargs),
        _jwks_fetcher=lambda: jwks,
    )


BOUND = {"alice@corp": frozenset({"acme"}), "svc-etl": frozenset({"globex"})}


# ------------------------------------------------------------------ SEC-004: the binding


def test_a_subject_reaching_another_tenant_is_refused(keypair):
    """THE test in this file. This is the cross-tenant read the allowlist could not prevent.

    `globex` IS provisioned, so the allowlist admits it. The token is genuine, correctly signed,
    unexpired, right audience. Only the subject is wrong for that tenant.
    """
    private, _ = keypair
    validator = make_validator(keypair, subject_tenants=BOUND)
    with pytest.raises(TokenRejected) as excinfo:
        validator.validate(token_for(private, sub="alice@corp", tenant="globex"))
    assert excinfo.value.reason == "subject_tenant_mismatch"


def test_a_bound_subject_reaches_its_own_tenant(keypair):
    private, _ = keypair
    validator = make_validator(keypair, subject_tenants=BOUND)
    assert validator.validate(token_for(private)).tenant == "acme"
    principal = validator.validate(token_for(private, sub="svc-etl", tenant="globex"))
    assert principal.tenant == "globex"


def test_an_unknown_subject_is_refused(keypair):
    """A subject absent from the map has no tenant, which is not the same as "any tenant"."""
    private, _ = keypair
    validator = make_validator(keypair, subject_tenants=BOUND)
    with pytest.raises(TokenRejected) as excinfo:
        validator.validate(token_for(private, sub="mallory@corp", tenant="acme"))
    assert excinfo.value.reason == "subject_not_bound"


def test_a_token_with_no_subject_is_refused_when_binding_is_on(keypair):
    """`sub` is optional to the validator generally; with a binding it is the lookup key.

    Falling back to the tenant as the principal name (which `validate` does when `sub` is absent)
    would mean the binding silently checked the tenant against itself.
    """
    private, _ = keypair
    validator = make_validator(keypair, subject_tenants=BOUND)
    token = token_for(private)
    stripped = jwt.encode(
        {k: v for k, v in jwt.decode(token, options={"verify_signature": False}).items()
         if k != "sub"},
        private,
        algorithm="RS256",
        headers={"kid": "key-1"},
    )
    with pytest.raises(TokenRejected) as excinfo:
        validator.validate(stripped)
    assert excinfo.value.reason == "missing_subject"


def test_the_binding_is_checked_after_the_signature(keypair):
    """Same reasoning as the tenant allowlist: ahead of the signature this enumerates subjects.

    A forged token would otherwise answer "is alice a known subject here?" for a caller holding no
    credential.
    """
    other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    validator = make_validator(keypair, subject_tenants=BOUND)
    with pytest.raises(TokenRejected) as excinfo:
        validator.validate(token_for(other, sub="mallory@corp", tenant="globex"))
    assert excinfo.value.reason == "bad_signature"


def test_no_binding_admits_any_subject(keypair):
    """Unset stays permissive at the VALIDATOR layer; the env factory is what refuses to deploy."""
    private, _ = keypair
    validator = make_validator(keypair)
    assert validator.validate(token_for(private, sub="anyone", tenant="globex")).tenant == "globex"


def test_a_binding_naming_an_unprovisioned_tenant_is_refused_at_construction():
    """Un-matchable configuration: the subject could never reach that tenant anyway.

    Same treatment as a padded allowlist entry. A binding that reads as provisioning and refuses
    at every request is worse than one that fails at boot.
    """
    with pytest.raises(ValueError, match="not in allowed_tenants"):
        OidcConfig(
            issuer=ISSUER,
            audience=AUDIENCE,
            allowed_tenants=frozenset({"acme"}),
            subject_tenants={"alice@corp": frozenset({"initech"})},
        )


def test_an_empty_binding_is_refused_at_construction():
    with pytest.raises(ValueError, match="subject_tenants"):
        OidcConfig(
            issuer=ISSUER, audience=AUDIENCE, allowed_tenants=frozenset({"acme"}), subject_tenants={}
        )


# ------------------------------------------------------------------ SEC-004: the env gate


def oidc_env(**extra) -> dict:
    env = {
        "RECALL_OIDC_ISSUER": ISSUER,
        "RECALL_OIDC_AUDIENCE": AUDIENCE,
        "RECALL_OIDC_TENANTS": "acme,globex",
        "RECALL_AUTH_RESOURCE_URL": RESOURCE,
    }
    env.update(extra)
    return env


def test_oidc_without_a_binding_or_an_explicit_trust_refuses_to_start():
    """The unsafe path is opt-IN and loud, never a default.

    Trusting the tenant claim is a legitimate deployment, but only when someone has decided the
    IdP is authoritative for it. A warning would be the fail-open shape this module refuses
    everywhere else.
    """
    from recall_mcp.server import build_auth

    with pytest.raises(AuthConfigError, match="RECALL_OIDC_TRUST_TENANT_CLAIM"):
        build_auth("streamable-http", env=oidc_env())


def test_an_explicit_trust_acknowledgement_boots():
    from recall_mcp.server import build_auth

    verifier, _settings, prov = build_auth(
        "streamable-http", env=oidc_env(RECALL_OIDC_TRUST_TENANT_CLAIM="1")
    )
    assert verifier is not None
    assert prov.tenants == frozenset({"acme", "globex"})


def test_a_subject_map_boots_without_the_trust_flag():
    from recall_mcp.server import build_auth

    verifier, _s, _p = build_auth(
        "streamable-http",
        env=oidc_env(RECALL_OIDC_SUBJECT_TENANTS="alice@corp:acme,svc-etl:globex"),
    )
    assert verifier._validator.config.subject_tenants == {
        "alice@corp": frozenset({"acme"}),
        "svc-etl": frozenset({"globex"}),
    }


def test_one_subject_may_hold_several_tenants():
    from recall_mcp.server import build_auth

    verifier, _s, _p = build_auth(
        "streamable-http",
        env=oidc_env(RECALL_OIDC_SUBJECT_TENANTS="svc-etl:acme,svc-etl:globex"),
    )
    assert verifier._validator.config.subject_tenants == {"svc-etl": frozenset({"acme", "globex"})}


def test_setting_both_the_map_and_the_trust_flag_refuses():
    """They answer the same question differently; whichever won, the other would look effective."""
    from recall_mcp.server import build_auth

    with pytest.raises(AuthConfigError, match="both"):
        build_auth(
            "streamable-http",
            env=oidc_env(
                RECALL_OIDC_SUBJECT_TENANTS="alice@corp:acme",
                RECALL_OIDC_TRUST_TENANT_CLAIM="1",
            ),
        )


@pytest.mark.parametrize("malformed", ["alice@corp", "alice@corp:", ":acme", ":"])
def test_a_malformed_subject_map_entry_refuses_to_start(malformed):
    from recall_mcp.server import build_auth

    with pytest.raises(AuthConfigError, match="RECALL_OIDC_SUBJECT_TENANTS"):
        build_auth("streamable-http", env=oidc_env(RECALL_OIDC_SUBJECT_TENANTS=malformed))


def test_the_derived_key_list_only_ever_contains_oidc_keys():
    """It filters on the VALUE prefix, not just the `ENV_` name prefix.

    Importing `auth.ENV_TOKENS_FILE` into the oidc module — one plausible line, and `server.py`
    already imports that alias — would otherwise put `RECALL_AUTH_TOKENS_FILE` on this list, and
    the stray-key guard would then refuse **every static-token deployment** as a partial OIDC
    block. Found by mutating exactly that.
    """
    import recall_mcp.oidc as oidc_module

    oidc_module.ENV_A_NON_OIDC_ALIAS = "RECALL_AUTH_TOKENS_FILE"
    try:
        keys = oidc_module.oidc_non_issuer_env_keys()
    finally:
        del oidc_module.ENV_A_NON_OIDC_ALIAS
    assert keys, "the derivation found nothing at all"
    assert all(k.startswith("RECALL_OIDC_") for k in keys), keys
    assert "RECALL_OIDC_ISSUER" not in keys, "the issuer is what switches OIDC on, not a stray"


def test_a_key_added_after_the_derivation_is_still_covered():
    """The first version was a module-level constant, so it saw only constants defined ABOVE it.

    That swapped a remembered enumeration for a remembered definition ORDER, which nothing
    enforced: a new key declared below the line was silently dropped and the whole OIDC suite
    stayed green. A function reads globals when called, so position stops mattering.
    """
    import recall_mcp.oidc as oidc_module

    oidc_module.ENV_A_LATE_OIDC_KNOB = "RECALL_OIDC_A_LATE_KNOB"
    try:
        assert "RECALL_OIDC_A_LATE_KNOB" in oidc_module.oidc_non_issuer_env_keys()
        with pytest.raises(AuthConfigError, match="RECALL_OIDC_A_LATE_KNOB"):
            oidc_module.oidc_env_present({"RECALL_OIDC_A_LATE_KNOB": "v"})
    finally:
        del oidc_module.ENV_A_LATE_OIDC_KNOB


def test_a_config_can_be_rebuilt_from_another_configs_binding():
    """The `Mapping` check, not `dict`: this class normalises to a MappingProxyType.

    An exact-dict guard made OidcConfig reject its own output, so `dataclasses.replace` and
    building one config from another's binding raised "must be a mapping" about a mapping.
    """
    import dataclasses

    first = OidcConfig(
        issuer=ISSUER,
        audience=AUDIENCE,
        allowed_tenants=frozenset({"acme"}),
        subject_tenants={"alice@corp": frozenset({"acme"})},
    )
    second = OidcConfig(
        issuer=ISSUER,
        audience=AUDIENCE,
        allowed_tenants=frozenset({"acme"}),
        subject_tenants=first.subject_tenants,
    )
    assert second.subject_tenants == first.subject_tenants
    assert dataclasses.replace(first, audience="other").subject_tenants == first.subject_tenants


@pytest.mark.parametrize("entry", ["a::b", "alice::acme"])
def test_a_doubled_separator_is_refused(entry):
    """`a::b` used to bind a subject with a trailing colon, which can never match a `sub`."""
    from recall_mcp.server import build_auth

    with pytest.raises(AuthConfigError, match="trailing colon"):
        build_auth("streamable-http", env=oidc_env(RECALL_OIDC_SUBJECT_TENANTS=entry))


def test_a_forgotten_tenant_is_caught_by_the_allowlist_cross_check():
    """The residual cost of splitting on the LAST colon, and what actually catches it.

    `system:serviceaccount:recall:etl` with the tenant forgotten parses as a shorter subject bound
    to `etl`. The parser cannot know that was a mistake; the allowlist cross-check can, because
    `etl` was never provisioned. Pinned so the cross-check is not later moved or removed on the
    grounds that the parser validates the entry.
    """
    from recall_mcp.server import build_auth

    with pytest.raises(AuthConfigError, match="not in allowed_tenants"):
        build_auth(
            "streamable-http",
            env=oidc_env(RECALL_OIDC_SUBJECT_TENANTS="system:serviceaccount:recall:etl"),
        )


def test_a_subject_containing_colons_is_bindable():
    """The tenant is everything after the LAST colon, so the subject may contain them.

    Kubernetes mints `system:serviceaccount:ns:sa`, and URN- and URL-shaped subjects are ordinary.
    Splitting on the FIRST colon made every such deployment unable to use the binding at all,
    which pushed exactly those operators onto the trust flag: the unsafe path this feature exists
    to replace.

    (An earlier version of this docstring also claimed the change fixed a silent misparse of
    `urn:acme` into subject `urn`. It did not: rpartition yields exactly the same result there. A
    one-colon entry is inherently ambiguous and no choice of split side resolves it.)
    """
    from recall_mcp.server import build_auth

    verifier, _s, _p = build_auth(
        "streamable-http",
        env=oidc_env(
            RECALL_OIDC_SUBJECT_TENANTS="system:serviceaccount:recall:etl:acme",
            RECALL_OIDC_TENANTS="acme,globex",
        ),
    )
    assert verifier._validator.config.subject_tenants == {
        "system:serviceaccount:recall:etl": frozenset({"acme"})
    }


def test_the_subject_binding_cannot_be_widened_at_runtime():
    """`allowed_tenants` beside it is a frozenset and cannot be. This is the same boundary.

    A plain dict on a frozen dataclass let `config.subject_tenants[x] = ...` add an unvalidated
    binding to a live validator, turning `subject_not_bound` into an accepted cross-tenant
    principal without passing a single check.
    """
    config = OidcConfig(
        issuer=ISSUER,
        audience=AUDIENCE,
        allowed_tenants=frozenset({"acme", "globex"}),
        subject_tenants={"alice@corp": frozenset({"acme"})},
    )
    with pytest.raises(TypeError):
        config.subject_tenants["mallory"] = frozenset({"globex"})


@pytest.mark.parametrize(
    "bad",
    [
        {"alice@corp": "acme"},  # a bare string explodes into {'a','c','e','m'}
        {" alice@corp ": frozenset({"acme"})},  # padded key never matches a `sub`
        {"": frozenset({"acme"})},
        {"alice@corp": frozenset({" acme "})},  # padded tenant never matches a claim
        "not-a-mapping",
    ],
)
def test_an_unmatchable_or_mistyped_binding_is_refused_at_construction(bad):
    """The same guards `allowed_tenants` carries. The comment claimed parity before the code had it."""
    with pytest.raises(ValueError):
        OidcConfig(issuer=ISSUER, audience=AUDIENCE, subject_tenants=bad)


# ------------------------------------------------------------------ the cutover selector


def tokens_file(tmp_path) -> str:
    p = tmp_path / "tokens.json"
    p.write_text(
        json.dumps({"principals": [{"name": "agent", "token": "t" * 40, "tenant": "acme"}]}),
        encoding="utf-8",
    )
    return str(p)


def both_env(tmp_path, **extra) -> dict:
    # RECALL_AUTH_ISSUER_URL is present because a deployment mid-cutover ALREADY had it: it is
    # required on the static path and only becomes optional once OIDC is the active mechanism.
    # Omitting it here would test a state no real rollout passes through.
    env = oidc_env(
        RECALL_AUTH_TOKENS_FILE=tokens_file(tmp_path),
        RECALL_AUTH_ISSUER_URL=RESOURCE,
        RECALL_OIDC_TRUST_TENANT_CLAIM="1",
    )
    env.update(extra)
    return env


def test_both_configured_without_a_mode_still_refuses(tmp_path):
    """The ambiguity guard survives. It was never wrong, it was only too broad."""
    from recall_mcp.server import build_auth

    with pytest.raises(AuthConfigError, match="RECALL_AUTH_MODE"):
        build_auth("streamable-http", env=both_env(tmp_path))


def test_mode_static_keeps_the_token_file_active_while_oidc_sits_ready(tmp_path):
    """Step 1 of a staged cutover: add every OIDC variable and change nothing."""
    from recall_mcp.server import RecallTokenVerifier, build_auth

    verifier, _s, _p = build_auth(
        "streamable-http", env=both_env(tmp_path, RECALL_AUTH_MODE="static")
    )
    assert isinstance(verifier, RecallTokenVerifier)


def test_mode_oidc_activates_the_provider_with_the_token_file_still_present(tmp_path):
    """Step 2: flip one variable. Rollback is flipping it back, which is the whole point."""
    from recall_mcp.server import OidcTokenVerifier, build_auth

    verifier, _s, _p = build_auth(
        "streamable-http", env=both_env(tmp_path, RECALL_AUTH_MODE="oidc")
    )
    assert isinstance(verifier, OidcTokenVerifier)


def test_mode_oidc_never_loads_the_static_token_file(tmp_path):
    """Load it and `RECALL_ENV=production` refuses the FILE before the selector is consulted.

    That is the state a real production cutover passes through, so the selected mechanism must be
    the only one built, not merely the one preferred afterwards.
    """
    from recall_mcp.server import OidcTokenVerifier, build_auth

    env = both_env(tmp_path, RECALL_AUTH_MODE="oidc", RECALL_ENV="production")
    verifier, _s, _p = build_auth("streamable-http", env=env)
    assert isinstance(verifier, OidcTokenVerifier)


def test_a_mode_naming_an_unconfigured_mechanism_refuses(tmp_path):
    from recall_mcp.server import build_auth

    with pytest.raises(AuthConfigError, match="RECALL_AUTH_MODE=oidc"):
        build_auth(
            "streamable-http",
            env={
                "RECALL_AUTH_TOKENS_FILE": tokens_file(tmp_path),
                "RECALL_AUTH_ISSUER_URL": RESOURCE,
                "RECALL_AUTH_RESOURCE_URL": RESOURCE,
                "RECALL_AUTH_MODE": "oidc",
            },
        )


def test_an_unknown_mode_refuses_and_names_the_valid_set():
    from recall_mcp.server import build_auth

    with pytest.raises(AuthConfigError, match="oidc.*static|static.*oidc"):
        build_auth("streamable-http", env=oidc_env(RECALL_AUTH_MODE="odic"))


def test_the_inactive_mechanism_is_logged_so_it_does_not_look_effective(tmp_path, caplog):
    """The reason both-set refused in the first place. Declaring precedence removes the refusal,
    so the log has to carry what the refusal used to say.
    """
    from recall_mcp.server import build_auth

    with caplog.at_level("WARNING"):
        build_auth("streamable-http", env=both_env(tmp_path, RECALL_AUTH_MODE="oidc"))
    assert any("RECALL_AUTH_TOKENS_FILE" in r.getMessage() for r in caplog.records)
