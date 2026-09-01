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
    cred.store_access_token(cred.Credential(access_token="<cached>", expires_at=1000.0))
    got = cred.cached_access_token(now=0.0)
    assert got is not None and got.access_token == "<cached>"


def test_a_token_about_to_expire_is_not_reused() -> None:
    """Expiring mid-request costs a 401 and a retry; a minute of slack costs nothing.

    ⚠️ The times here are FIXED, not derived from `REFRESH_MARGIN_S`. The first version computed
    `now` from the constant, so it moved with it: setting the margin to 0 left this test green,
    and a surviving mutant found it. A test parameterised by the value it exists to pin cannot
    detect that value changing.
    """
    cred.store_access_token(cred.Credential(access_token="<stale>", expires_at=1000.0))
    assert cred.cached_access_token(now=970.0) is None, (
        "a token with 30s of life left was reused; the refresh margin is too small or gone"
    )
    assert cred.cached_access_token(now=0.0) is not None, "a token with ample life was discarded"


def test_a_missing_or_corrupt_cache_is_a_miss_not_a_crash() -> None:
    assert cred.cached_access_token() is None
    cred.token_cache_path().parent.mkdir(parents=True, exist_ok=True)
    cred.token_cache_path().write_text("{{{", encoding="utf-8")
    assert cred.cached_access_token() is None


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
    cred.store_access_token(cred.Credential(access_token="<live>", expires_at=1e12))
    assert cred.print_auth_headers(CONFIG) == 0
    assert json.loads(capsys.readouterr().out) == {"Authorization": "Bearer <live>"}


def test_the_module_does_not_import_recall() -> None:
    """The package rule, asserted rather than assumed: SessionStart runs before every first turn."""
    import subprocess

    code = "import sys, recall_hooks.credentials; print('recall' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.stdout.strip() == "False", out.stdout + out.stderr
