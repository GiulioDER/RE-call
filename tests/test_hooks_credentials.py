"""Where a hosted client's credential lives, and what happens when it cannot be found.

Every test here is pure: no keychain, no network, no database. The OS keyring is substituted, both
present and absent, because the fallback to a plaintext file is a real path on a headless Linux box
and "it probably works" is not a claim this file is willing to make about a secret.
"""

from __future__ import annotations

import json
import stat
import sys
from pathlib import Path

import pytest

from recall_hooks import credentials as cred


@pytest.fixture(autouse=True)
def _isolate_config_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Never read or write the developer's real ~/.claude while testing a credential store."""
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))


class FakeKeyring:
    """Enough of the keyring API to exercise both branches, including a hostile one."""

    def __init__(self, *, broken: bool = False) -> None:
        self.store: dict[tuple[str, str], str] = {}
        self.broken = broken

    def get_password(self, service: str, account: str) -> str | None:
        if self.broken:
            raise RuntimeError("no Secret Service available")
        return self.store.get((service, account))

    def set_password(self, service: str, account: str, value: str) -> None:
        if self.broken:
            raise RuntimeError("no Secret Service available")
        self.store[(service, account)] = value

    def delete_password(self, service: str, account: str) -> None:
        if self.broken:
            raise RuntimeError("no Secret Service available")
        del self.store[(service, account)]


def _with_keyring(monkeypatch: pytest.MonkeyPatch, ring) -> None:
    monkeypatch.setattr(cred, "_keyring", lambda: ring)


CONFIG = {
    "account": "someone@example.test",
    "auth": {"token_url": "https://id.example.test/token", "client_id": "recall-mcp"},
}


# --------------------------------------------------------------------------- config


def test_a_local_config_is_refused_as_not_hosted() -> None:
    """A config with no auth block is a local-DSN install, not a broken hosted one."""
    with pytest.raises(cred.AuthError, match="not a hosted config"):
        cred.auth_config({"dsn": "postgresql://x/y"})


def test_a_half_written_auth_block_names_what_is_missing() -> None:
    with pytest.raises(cred.AuthError, match="client_id"):
        cred.auth_config({"auth": {"token_url": "https://id.example.test/token"}})


def test_no_endpoint_is_hardcoded() -> None:
    """The public package ships a generic client; a hosting layer supplies the values.

    A hostname baked in here would make the open-source client point at one vendor.
    """
    source = Path(cred.__file__).read_text(encoding="utf-8")
    for marker in ("recall.example", ".com/", "api.", "auth0", "okta"):
        assert marker not in source, f"{marker!r} looks like a hardcoded endpoint"


# --------------------------------------------------------------------------- storage


def test_the_refresh_token_prefers_the_os_keychain(monkeypatch: pytest.MonkeyPatch) -> None:
    ring = FakeKeyring()
    _with_keyring(monkeypatch, ring)
    assert cred.write_refresh_token(CONFIG, "<token-value>") == "keyring"
    assert cred.read_refresh_token(CONFIG) == "<token-value>"
    assert not cred.fallback_path().exists(), "a keychain write must not also write plaintext"


def test_a_broken_keychain_falls_back_to_a_file_and_says_which(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A headless Linux box has no Secret Service, and the caller must be able to tell the user.

    `write_refresh_token` returns the store rather than logging it, so `login` can report the
    downgrade and a hook can stay quiet. A silent move from a keychain to a plaintext file is not
    a decision to make on somebody's behalf.
    """
    _with_keyring(monkeypatch, FakeKeyring(broken=True))
    assert cred.write_refresh_token(CONFIG, "<token-value>") == "file"
    assert cred.read_refresh_token(CONFIG) == "<token-value>"
    assert cred.fallback_path().exists()


def test_an_absent_keyring_module_is_normal_not_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """`keyring` ships in the `desktop` extra, so a hosted-only install may not have it."""
    _with_keyring(monkeypatch, None)
    assert cred.write_refresh_token(CONFIG, "<token-value>") == "file"
    assert cred.read_refresh_token(CONFIG) == "<token-value>"


@pytest.mark.skipif(sys.platform == "win32", reason="Windows has no meaningful mode bits here")
def test_the_fallback_file_is_readable_only_by_its_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_keyring(monkeypatch, None)
    cred.write_refresh_token(CONFIG, "<token-value>")
    mode = cred.fallback_path().stat().st_mode
    assert not mode & stat.S_IRGRP and not mode & stat.S_IROTH


def test_two_accounts_on_one_machine_do_not_overwrite_each_other(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _with_keyring(monkeypatch, None)
    a = {**CONFIG, "account": "a@example.test"}
    b = {**CONFIG, "account": "b@example.test"}
    cred.write_refresh_token(a, "<token-a>")
    cred.write_refresh_token(b, "<token-b>")
    assert cred.read_refresh_token(a) == "<token-a>"
    assert cred.read_refresh_token(b) == "<token-b>"


def test_a_missing_credential_reads_as_none_rather_than_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _with_keyring(monkeypatch, FakeKeyring())
    assert cred.read_refresh_token(CONFIG) is None


def test_a_corrupt_fallback_file_does_not_take_the_hook_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _with_keyring(monkeypatch, None)
    cred.fallback_path().parent.mkdir(parents=True, exist_ok=True)
    cred.fallback_path().write_text("{ not json", encoding="utf-8")
    assert cred.read_refresh_token(CONFIG) is None
    assert cred.write_refresh_token(CONFIG, "<token-value>") == "file"
    assert cred.read_refresh_token(CONFIG) == "<token-value>"


# --------------------------------------------------------------------------- access token cache


def test_a_fresh_cached_token_is_reused() -> None:
    cred.store_access_token(CONFIG, cred.Credential(access_token="<cached>", expires_at=1000.0))
    got = cred.cached_access_token(CONFIG, now=0.0)
    assert got is not None and got.access_token == "<cached>"


def test_a_token_about_to_expire_is_not_reused() -> None:
    """Expiring mid-request costs a 401 and a retry; a minute of slack costs nothing.

    ⚠️ The times here are FIXED, not derived from `REFRESH_MARGIN_S`. The first version computed
    `now` from the constant, so it moved with it: setting the margin to 0 left this test green,
    and a surviving mutant found it. A test parameterised by the value it exists to pin cannot
    detect that value changing.
    """
    cred.store_access_token(CONFIG, cred.Credential(access_token="<stale>", expires_at=1000.0))
    assert cred.cached_access_token(CONFIG, now=970.0) is None, (
        "a token with 30s of life left was reused; the refresh margin is too small or gone"
    )
    assert cred.cached_access_token(CONFIG, now=0.0) is not None, "a token with ample life was discarded"


def test_a_missing_or_corrupt_cache_is_a_miss_not_a_crash() -> None:
    assert cred.cached_access_token(CONFIG) is None
    cred.token_cache_path(CONFIG).parent.mkdir(parents=True, exist_ok=True)
    cred.token_cache_path(CONFIG).write_text("{{{", encoding="utf-8")
    assert cred.cached_access_token(CONFIG) is None


def test_no_stored_credential_says_what_to_run(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_keyring(monkeypatch, FakeKeyring())
    with pytest.raises(cred.AuthError, match="recall-hooks login"):
        cred.access_token(CONFIG, now=0.0)


def test_a_plaintext_token_url_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """A bearer token must not cross a plaintext hop, and the check is before the request."""
    _with_keyring(monkeypatch, None)
    cred.write_refresh_token(CONFIG, "<token-value>")
    insecure = {**CONFIG, "auth": {**CONFIG["auth"], "token_url": "http://id.example.test/token"}}
    with pytest.raises(cred.AuthError, match="https"):
        cred.access_token(insecure, now=0.0)


# --------------------------------------------------------------------------- the helper contract


def test_auth_headers_exits_zero_and_prints_empty_json_when_there_is_no_credential(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """⛔ The contract. A non-zero exit makes the MCP client treat the server as broken.

    An expired token would then read as "recall is broken" rather than "recall needs a login",
    which is the wrong diagnosis pointing at the wrong remedy.
    """
    _with_keyring(monkeypatch, FakeKeyring())
    assert cred.print_auth_headers(CONFIG) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out.strip()) == {}
    assert "no credential" in captured.err


def test_auth_headers_exits_zero_even_on_an_unexpected_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Whatever goes wrong, stdout stays valid JSON and the exit code stays 0."""

    def boom(*_args, **_kwargs):
        raise ValueError("something nobody predicted")

    monkeypatch.setattr(cred, "headers", boom)
    assert cred.print_auth_headers(CONFIG) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out.strip()) == {}
    assert "credential lookup failed" in captured.err


def test_auth_headers_prints_the_bearer_when_there_is_one(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cred.store_access_token(CONFIG, cred.Credential(access_token="<live>", expires_at=1e12))
    assert cred.print_auth_headers(CONFIG) == 0
    assert json.loads(capsys.readouterr().out) == {"Authorization": "Bearer <live>"}


def test_the_module_does_not_import_recall() -> None:
    """The package rule, asserted rather than assumed: SessionStart runs before every first turn."""
    import subprocess

    code = "import sys, recall_hooks.credentials; print('recall' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.stdout.strip() == "False", out.stdout + out.stderr


# --------------------------------------------------------------------------- login


class _Out:
    """A stream that remembers, so a test can assert on what a person was actually told."""

    def __init__(self) -> None:
        self.text = ""

    def write(self, chunk: str) -> int:
        self.text += chunk
        return len(chunk)

    def flush(self) -> None:
        pass


class _Stdin:
    def __init__(self, text: str) -> None:
        self.text = text

    def read(self) -> str:
        return self.text


def _credential() -> cred.Credential:
    return cred.Credential(access_token="at-1", expires_at=9e18)


def test_login_with_no_token_refuses_and_says_how(monkeypatch: pytest.MonkeyPatch) -> None:
    out, err = _Out(), _Out()
    _with_keyring(monkeypatch, FakeKeyring())
    code = cred.login(CONFIG, [], stdin=_Stdin("  \n"), out=out, err=err)
    assert code == 2
    assert "recall-hooks login" in err.text
    assert "shell history" in err.text, "the --token hazard is named where it is offered"


def test_login_stores_nothing_when_the_config_cannot_exchange_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """⛔ Refusing must actually refuse. Storing a token that nothing can exchange, and reporting
    success, is the same false success as the missing command this replaces."""
    ring = FakeKeyring()
    _with_keyring(monkeypatch, ring)
    out, err = _Out(), _Out()
    code = cred.login({"dsn": "postgresql://x/y"}, [], stdin=_Stdin("rt-1"), out=out, err=err)
    assert code == 2
    assert ring.store == {}
    assert cred.read_refresh_token({"dsn": "postgresql://x/y"}) is None
    assert not cred.fallback_path().exists()


def test_login_stores_and_verifies(monkeypatch: pytest.MonkeyPatch) -> None:
    ring = FakeKeyring()
    _with_keyring(monkeypatch, ring)
    monkeypatch.setattr(cred, "refresh", lambda *_a, **_k: _credential())
    out, err = _Out(), _Out()
    assert cred.login(CONFIG, [], stdin=_Stdin("rt-1\n"), out=out, err=err) == 0
    assert cred.read_refresh_token(CONFIG) == "rt-1"
    assert cred.cached_access_token(CONFIG) is not None, "the access token is cached, not re-fetched"
    assert "signed in" in out.text


def test_a_token_that_cannot_be_verified_is_kept_but_does_not_exit_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """3, not 0 and not 1. The token is kept because being offline is the benign case and the next
    session should retry; the exit code is non-zero because from here offline and revoked look
    identical and only one of them is safe to call success."""
    _with_keyring(monkeypatch, FakeKeyring())

    def unreachable(*_a, **_k):
        raise OSError("getaddrinfo failed")

    monkeypatch.setattr(cred, "refresh", unreachable)
    out, err = _Out(), _Out()
    assert cred.login(CONFIG, [], stdin=_Stdin("rt-1"), out=out, err=err) == 3
    assert cred.read_refresh_token(CONFIG) == "rt-1", "kept, so the next session can retry"
    assert "could not be verified" in err.text


def test_a_rejected_token_is_also_three(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_keyring(monkeypatch, FakeKeyring())

    def rejected(*_a, **_k):
        raise cred.AuthError("invalid_grant")

    monkeypatch.setattr(cred, "refresh", rejected)
    out, err = _Out(), _Out()
    assert cred.login(CONFIG, [], stdin=_Stdin("rt-1"), out=out, err=err) == 3
    assert "invalid_grant" in err.text


def test_the_plaintext_fallback_is_announced(monkeypatch: pytest.MonkeyPatch) -> None:
    """🔑 A silent downgrade from a keychain to a plaintext file is not a decision to make on
    somebody's behalf, so it is said out loud even on the success path."""
    _with_keyring(monkeypatch, None)
    monkeypatch.setattr(cred, "refresh", lambda *_a, **_k: _credential())
    out, err = _Out(), _Out()
    assert cred.login(CONFIG, [], stdin=_Stdin("rt-1"), out=out, err=err) == 0
    assert "plaintext" in err.text
    assert str(cred.fallback_path()) in err.text


def test_token_flag_needs_a_value(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_keyring(monkeypatch, FakeKeyring())
    out, err = _Out(), _Out()
    assert cred.login(CONFIG, ["--token"], stdin=_Stdin(""), out=out, err=err) == 2


def test_token_flag_does_not_read_stdin(monkeypatch: pytest.MonkeyPatch) -> None:
    ring = FakeKeyring()
    _with_keyring(monkeypatch, ring)
    monkeypatch.setattr(cred, "refresh", lambda *_a, **_k: _credential())
    out, err = _Out(), _Out()
    assert cred.login(CONFIG, ["--token", "rt-flag"], stdin=_Stdin("rt-stdin"),
                      out=out, err=err) == 0
    assert cred.read_refresh_token(CONFIG) == "rt-flag"


# --------------------------------------------------------------------------- logout


def test_logout_clears_both_stores_and_the_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both, whatever the keychain says. `write_refresh_token` falls back to the file, so a later
    install of `keyring` would otherwise leave a stale plaintext token nobody remembers."""
    ring = FakeKeyring()
    _with_keyring(monkeypatch, None)
    cred.write_refresh_token(CONFIG, "rt-file")     # lands in the plaintext fallback
    _with_keyring(monkeypatch, ring)
    cred.write_refresh_token(CONFIG, "rt-ring")     # and in the keychain
    cred.store_access_token(CONFIG, _credential())
    assert cred.token_cache_path(CONFIG).exists()

    out = _Out()
    assert cred.logout(CONFIG, out=out) == 0
    assert cred.read_refresh_token(CONFIG) is None
    assert ring.store == {}
    assert not cred.token_cache_path(CONFIG).exists()


def test_logout_with_nothing_stored_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_keyring(monkeypatch, FakeKeyring())
    out = _Out()
    assert cred.logout(CONFIG, out=out) == 0
    assert "no stored credential" in out.text


# --------------------------------------------------------------- the cache is bound to an account


OTHER = {
    "account": "somebody-else@example.test",
    "auth": {"token_url": "https://id.example.test/token", "client_id": "recall-mcp"},
}


def test_one_accounts_token_is_never_served_to_another() -> None:
    """⛔ THE LEAK THIS BINDING PREVENTS, asserted rather than reasoned about.

    The refresh token was keyed by account from the start; this cache was not. A hook in a project
    configured for account A cached A's token, and a hook in a project configured for account B
    within that token's lifetime got it back, uploaded B's memos under A's bearer, and the server
    resolved the tenant from the token. B's memories landed in A's corpus, with no error on either
    side.
    """
    cred.store_access_token(CONFIG, cred.Credential(access_token="alice", expires_at=1e12))
    assert cred.cached_access_token(CONFIG).access_token == "alice"
    assert cred.cached_access_token(OTHER) is None, "another account must not see this token"


def test_two_accounts_keep_separate_caches() -> None:
    """Separate paths rather than one file that is overwritten, so alternating between two
    projects does not cost a token round trip on every switch."""
    cred.store_access_token(CONFIG, cred.Credential(access_token="alice", expires_at=1e12))
    cred.store_access_token(OTHER, cred.Credential(access_token="bob", expires_at=1e12))
    assert cred.cached_access_token(CONFIG).access_token == "alice"
    assert cred.cached_access_token(OTHER).access_token == "bob"
    assert cred.token_cache_path(CONFIG) != cred.token_cache_path(OTHER)


def test_the_filename_does_not_carry_an_email_address() -> None:
    assert "example.test" not in cred.token_cache_path(CONFIG).name
    assert "someone" not in cred.token_cache_path(CONFIG).name


def test_a_payload_naming_the_wrong_account_is_refused() -> None:
    """Belt to the path's braces. Reaching this means a hand-edited file or an older payload, and
    every reason to be here is a reason to fetch a fresh token instead."""
    path = cred.token_cache_path(CONFIG)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"account": "someone-else", "access_token": "x", "expires_at": 1e12}),
        encoding="utf-8",
    )
    assert cred.cached_access_token(CONFIG) is None


def test_logout_removes_only_this_accounts_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_keyring(monkeypatch, FakeKeyring())
    cred.store_access_token(CONFIG, cred.Credential(access_token="alice", expires_at=1e12))
    cred.store_access_token(OTHER, cred.Credential(access_token="bob", expires_at=1e12))
    cred.logout(CONFIG, out=_Out())
    assert cred.cached_access_token(CONFIG) is None
    assert cred.cached_access_token(OTHER) is not None, "signing one account out is not signing all"

