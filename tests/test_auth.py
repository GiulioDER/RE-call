"""Tests for bearer-token auth, scope enforcement and principal -> tenant mapping.

The bias here is toward the REFUSAL paths. A test that a valid token works proves the happy path
a user would notice broken within a minute; a test that an HTTP transport refuses to start without
tokens proves the thing that, when it regresses, nobody notices until the corpus is public.
"""

from __future__ import annotations

import json
import os
import stat
import sys
from datetime import datetime, timedelta, timezone

import pytest

from recall_mcp.auth import (
    ALL_SCOPES,
    MIN_TOKEN_LENGTH,
    SCOPE_FORGET,
    SCOPE_READ,
    SCOPE_WRITE,
    AuthConfigError,
    TokenRegistry,
    _sha256_hex,
    authorize,
    load_token_registry,
    parse_principals,
    token_registry_from_env,
)

GOOD_TOKEN = "t" * 40
OTHER_TOKEN = "u" * 40


def doc(**overrides) -> dict:
    entry = {"name": "agent", "token": GOOD_TOKEN, "tenant": "team-a"}
    entry.update(overrides)
    return {"principals": [entry]}


def write_tokens(tmp_path, payload) -> str:
    p = tmp_path / "tokens.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return str(p)


# --------------------------------------------------------------------------- config validation


def test_valid_document_yields_a_principal_bound_to_its_tenant():
    registry = TokenRegistry(parse_principals(doc()))
    principal = registry.verify(GOOD_TOKEN)
    assert principal is not None
    assert principal.name == "agent"
    assert principal.tenant == "team-a"


def test_scopes_default_to_read_only():
    """Least privilege by default: an entry that names no scopes cannot write or forget."""
    principal = TokenRegistry(parse_principals(doc())).verify(GOOD_TOKEN)
    assert principal.scopes == frozenset({SCOPE_READ})
    assert not principal.has_scope(SCOPE_WRITE)
    assert not principal.has_scope(SCOPE_FORGET)


def test_short_token_is_refused_and_the_message_does_not_echo_it():
    # A distinctive sentinel, not a plausible-looking word: "short" would collide with the word
    # "shorter" in the message and make this assertion pass for the wrong reason.
    weak = "Zq7-SENTINEL"
    with pytest.raises(AuthConfigError) as exc:
        parse_principals(doc(token=weak))
    assert str(MIN_TOKEN_LENGTH) in str(exc.value)
    # An error string is the single most likely thing in this module to reach a bug report,
    # a CI log or an aggregator. It must not carry the credential — not even a prefix.
    assert weak not in str(exc.value)


def test_precomputed_digest_is_accepted_so_plaintext_need_never_be_written_to_disk():
    registry = TokenRegistry(
        parse_principals(doc(token=None, token_sha256=_sha256_hex(GOOD_TOKEN)))
    )
    assert registry.verify(GOOD_TOKEN).name == "agent"


@pytest.mark.parametrize(
    "entry",
    [
        {},  # neither
        {"token": GOOD_TOKEN, "token_sha256": _sha256_hex(GOOD_TOKEN)},  # both
    ],
    ids=["neither", "both"],
)
def test_exactly_one_of_token_or_digest_is_required(entry):
    payload = {"principals": [{"name": "a", "tenant": "t", **entry}]}
    with pytest.raises(AuthConfigError, match="exactly one"):
        parse_principals(payload)


def test_malformed_digest_is_refused():
    with pytest.raises(AuthConfigError, match="64-char hex"):
        parse_principals(doc(token=None, token_sha256="abc"))


def test_two_principals_sharing_a_token_is_refused():
    """A shared token makes the audit trail a lie — you could never tell which principal acted."""
    payload = {
        "principals": [
            {"name": "a", "token": GOOD_TOKEN, "tenant": "t1"},
            {"name": "b", "token": GOOD_TOKEN, "tenant": "t2"},
        ]
    }
    with pytest.raises(AuthConfigError, match="collides"):
        parse_principals(payload)


def test_duplicate_principal_name_is_refused():
    payload = {
        "principals": [
            {"name": "a", "token": GOOD_TOKEN, "tenant": "t1"},
            {"name": "a", "token": OTHER_TOKEN, "tenant": "t2"},
        ]
    }
    with pytest.raises(AuthConfigError, match="duplicate"):
        parse_principals(payload)


def test_unknown_scope_is_refused_rather_than_silently_dropped():
    """A typo'd scope that is ignored gives a principal less access than the operator wrote."""
    with pytest.raises(AuthConfigError, match="unknown scope"):
        parse_principals(doc(scopes=["recall:red"]))


def test_naive_expiry_is_refused_at_config_time_not_request_time():
    """A naive datetime would raise when compared to an aware `now` — i.e. on the first call."""
    with pytest.raises(AuthConfigError, match="timezone"):
        parse_principals(doc(expires_at="2099-01-01T00:00:00"))


@pytest.mark.parametrize(
    "payload",
    [{}, {"principals": []}, {"principals": "nope"}, []],
    ids=["no-key", "empty", "not-a-list", "not-an-object"],
)
def test_structurally_invalid_documents_are_refused(payload):
    with pytest.raises(AuthConfigError):
        parse_principals(payload)


def test_blank_tenant_is_refused():
    with pytest.raises(AuthConfigError, match="tenant"):
        parse_principals(doc(tenant="  "))


# --------------------------------------------------------------------------- verification


def test_unknown_token_is_rejected():
    assert TokenRegistry(parse_principals(doc())).verify(OTHER_TOKEN) is None


@pytest.mark.parametrize("token", ["", "  ", "\x00"], ids=["empty", "blank", "nul"])
def test_degenerate_tokens_are_rejected(token):
    assert TokenRegistry(parse_principals(doc())).verify(token) is None


def test_expired_token_is_rejected_even_though_it_is_configured():
    past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    registry = TokenRegistry(parse_principals(doc(expires_at=past)))
    assert registry.verify(GOOD_TOKEN) is None


def test_unexpired_token_is_accepted_and_expiry_is_evaluated_against_now():
    at = datetime(2030, 1, 1, tzinfo=timezone.utc)
    registry = TokenRegistry(parse_principals(doc(expires_at=at.isoformat())))
    assert registry.verify(GOOD_TOKEN, now=at - timedelta(seconds=1)) is not None
    # Exactly at the expiry instant the token is already dead — `>=`, not `>`.
    assert registry.verify(GOOD_TOKEN, now=at) is None


def test_registry_never_retains_the_plaintext_token():
    """A heap dump or a stray repr must not hand over live credentials."""
    registry = TokenRegistry(parse_principals(doc()))
    blob = repr(registry.__dict__)
    assert GOOD_TOKEN not in blob
    assert _sha256_hex(GOOD_TOKEN) in blob  # the digest is what is kept


def test_tenants_are_fixed_by_configuration():
    """StoreRegistry bounds its pools on this set, so it must reflect config and nothing else."""
    payload = {
        "principals": [
            {"name": "a", "token": GOOD_TOKEN, "tenant": "t1"},
            {"name": "b", "token": OTHER_TOKEN, "tenant": "t2"},
        ]
    }
    assert TokenRegistry(parse_principals(payload)).tenants == frozenset({"t1", "t2"})


# --------------------------------------------------------------------------- authorize()


def test_authorize_returns_the_tenant_when_the_scope_is_held():
    assert authorize([SCOPE_READ], {"tenant": "team-a"}, SCOPE_READ) == "team-a"


def test_authorize_refuses_a_missing_scope():
    with pytest.raises(PermissionError, match="recall:forget"):
        authorize([SCOPE_READ, SCOPE_WRITE], {"tenant": "team-a"}, SCOPE_FORGET)


@pytest.mark.parametrize(
    "claims", [None, {}, {"tenant": ""}, {"tenant": 7}], ids=["none", "empty", "blank", "not-str"]
)
def test_authorize_fails_closed_when_the_tenant_claim_is_unusable(claims):
    """It must NOT fall back to a default — that is the namespace single-tenant installs use."""
    with pytest.raises(PermissionError, match="tenant"):
        authorize(ALL_SCOPES, claims, SCOPE_READ)


def test_authorize_refuses_when_the_token_carries_no_scopes_at_all():
    with pytest.raises(PermissionError):
        authorize(None, {"tenant": "team-a"}, SCOPE_READ)


# --------------------------------------------------------------------------- file loading


def test_load_from_file_round_trips(tmp_path):
    registry = load_token_registry(write_tokens(tmp_path, doc()))
    assert registry.verify(GOOD_TOKEN).tenant == "team-a"
    assert len(registry) == 1


def test_missing_file_is_a_config_error(tmp_path):
    with pytest.raises(AuthConfigError, match="not found"):
        load_token_registry(tmp_path / "absent.json")


def test_invalid_json_is_a_config_error(tmp_path):
    p = tmp_path / "tokens.json"
    p.write_text("{not json", encoding="utf-8")
    with pytest.raises(AuthConfigError, match="not valid JSON"):
        load_token_registry(p)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits")
def test_world_readable_token_file_warns_but_still_loads(tmp_path, caplog):
    """A hard failure here would push operators toward disabling auth entirely — worse."""
    path = write_tokens(tmp_path, doc())
    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IROTH)
    with caplog.at_level("WARNING"):
        assert load_token_registry(path) is not None
    assert any("readable by group or other" in r.message for r in caplog.records)


def test_env_without_the_file_variable_means_no_auth_configured():
    assert token_registry_from_env({}) is None


def test_env_pointing_at_a_real_file_loads_it(tmp_path):
    env = {"RECALL_AUTH_TOKENS_FILE": write_tokens(tmp_path, doc())}
    assert token_registry_from_env(env).verify(GOOD_TOKEN) is not None


def test_there_is_no_env_var_that_accepts_a_raw_token(tmp_path):
    """Guards the deliberate omission: env vars leak via /proc, `ps e` and child processes.

    If someone later adds `RECALL_AUTH_TOKENS=<secret>` as a convenience, this fails.
    """
    env = {"RECALL_AUTH_TOKENS": json.dumps(doc())}
    assert token_registry_from_env(env) is None
