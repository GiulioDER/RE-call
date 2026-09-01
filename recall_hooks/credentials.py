"""Where a hosted client's credential lives, and how it reaches both consumers of it.

Hosted mode has two processes that need an `Authorization` header and cannot share one in memory:
the MCP client, which runs this module through a `headersHelper` subprocess, and the hooks, which
call it in-process. One credential store with two readers is the whole design.

**Nothing here imports `recall`.** Same measured rule as the rest of this package:
`recall/__init__.py` costs about a second and `SessionStart` runs before the user's first turn of
every session. In particular this does NOT import `recall.desktop.profiles`, whose own keyring
helper is the obvious thing to reuse and which drags in `recall.desktop.models` and therefore the
whole package.

**No hostname, client id or scope is written here.** They come from the hook config, so the public
package ships a generic device-code client and a hosting layer supplies only values.
"""

from __future__ import annotations

import json
import os
import stat
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Seconds of remaining life below which a cached access token is refreshed rather than used. A
#: token that expires mid-request costs a 401 and a retry; a minute of slack costs nothing.
REFRESH_MARGIN_S = 60

#: Where the short-lived access token is cached. Beside the hook config rather than inside it: the
#: config is read by every session start, and a token has a different lifetime and a different
#: sensitivity from the settings around it.
TOKEN_CACHE_NAME = "recall-hook-token"  # noqa: S105 - a filename, not a secret

#: Fallback store for the refresh token when no OS keychain is reachable. Plaintext, 0600, and
#: `login` says so out loud when it uses this — a silent downgrade from a keychain to a file is not
#: a decision to make on somebody's behalf.
FALLBACK_NAME = "recall-credentials.json"

KEYRING_SERVICE = "recall"


class AuthError(RuntimeError):
    """No usable credential, and the caller must decide what that means.

    Raised rather than returned because the two callers want opposite things: `auth-headers` must
    still exit 0 and print an empty object, while a sync wants to record the failure and surface it
    at the next `SessionStart`.
    """


@dataclass(frozen=True)
class Credential:
    access_token: str
    expires_at: float

    def fresh(self, *, now: float | None = None) -> bool:
        return self.expires_at - (time.time() if now is None else now) > REFRESH_MARGIN_S


def _config_home() -> Path:
    """The Claude config directory, honouring `CLAUDE_CONFIG_DIR` as the rest of the package does."""
    override = os.environ.get("CLAUDE_CONFIG_DIR")
    return Path(override) if override else Path.home() / ".claude"


def token_cache_path() -> Path:
    return _config_home() / TOKEN_CACHE_NAME


def fallback_path() -> Path:
    return _config_home() / FALLBACK_NAME


def auth_config(config: dict[str, Any]) -> dict[str, str]:
    """The OAuth endpoints, from the hook config. Raises `AuthError` if it is not hosted-shaped."""
    block = config.get("auth")
    if not isinstance(block, dict):
        raise AuthError("this config names no auth block, so it is not a hosted config")
    missing = [k for k in ("token_url", "client_id") if not block.get(k)]
    if missing:
        raise AuthError(f"auth block is missing {', '.join(missing)}")
    return {
        "token_url": str(block["token_url"]),
        "client_id": str(block["client_id"]),
        "scopes": str(block.get("scopes", "")),
    }


def _keyring() -> Any:
    """The keyring module, or None. Absent is normal: it ships in the `desktop` extra."""
    try:
        import keyring  # noqa: PLC0415 - optional and deliberately lazy
    except Exception:
        return None
    return keyring


def _account(config: dict[str, Any]) -> str:
    """Which credential this is, so two accounts on one machine do not overwrite each other."""
    return str(config.get("account") or config.get("tenant") or "default")


def _write_private_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically, and readable only by this user.

    `mkstemp` rather than a fixed temp name, and chmod BEFORE the rename, so the secret is never
    momentarily world-readable at its final path. Both points are the same ones `_save_config`
    makes about the hook config, for a file that matters more.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle)
        try:
            os.chmod(tmp, stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            # Windows has no meaningful mode bits here. Not fatal, and not silent either: the
            # caller that chose this path has already been told it is a plaintext fallback.
            pass
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def read_refresh_token(config: dict[str, Any]) -> str | None:
    """From the OS keychain if there is one, else from the plaintext fallback, else None."""
    account = _account(config)
    ring = _keyring()
    if ring is not None:
        try:
            value = ring.get_password(KEYRING_SERVICE, account)
            if value:
                return str(value)
        except Exception:
            pass
    try:
        stored = json.loads(fallback_path().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    value = stored.get(account) if isinstance(stored, dict) else None
    return str(value) if value else None


def write_refresh_token(config: dict[str, Any], token: str) -> str:
    """Store it. Returns `"keyring"` or `"file"` — WHICH one is the caller's to report.

    Returning the store rather than logging it keeps this function quiet enough to call from a
    hook, while making it impossible for `login` to downgrade to plaintext without knowing.
    """
    account = _account(config)
    ring = _keyring()
    if ring is not None:
        try:
            ring.set_password(KEYRING_SERVICE, account, token)
            return "keyring"
        except Exception:
            pass
    path = fallback_path()
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(stored, dict):
            stored = {}
    except (OSError, json.JSONDecodeError):
        stored = {}
    stored[account] = token
    _write_private_json(path, stored)
    return "file"


def cached_access_token(*, now: float | None = None) -> Credential | None:
    """The cached token if it has real life left, else None.

    Without this cache every MCP client start and every hook run pays a token round trip.
    """
    try:
        raw = json.loads(token_cache_path().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    token = raw.get("access_token") if isinstance(raw, dict) else None
    expires = raw.get("expires_at") if isinstance(raw, dict) else None
    if not token or not isinstance(expires, (int, float)):
        return None
    cred = Credential(access_token=str(token), expires_at=float(expires))
    return cred if cred.fresh(now=now) else None


def store_access_token(cred: Credential) -> None:
    _write_private_json(
        token_cache_path(), {"access_token": cred.access_token, "expires_at": cred.expires_at}
    )


def refresh(config: dict[str, Any], refresh_token: str, *, now: float | None = None) -> Credential:
    """Exchange a refresh token for an access token. One round trip, no retries.

    A retry loop here would turn an expired or revoked credential into a slow failure on the
    critical path of a session, and the remedy for both is the same: sign in again.
    """
    import urllib.error  # noqa: PLC0415 - stdlib, kept off the SessionStart import path
    import urllib.parse
    import urllib.request

    cfg = auth_config(config)
    body = urllib.parse.urlencode(
        {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": cfg["client_id"],
            **({"scope": cfg["scopes"]} if cfg["scopes"] else {}),
        }
    ).encode("ascii")
    request = urllib.request.Request(  # noqa: S310 - scheme is checked below
        cfg["token_url"], data=body, headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    if not cfg["token_url"].startswith("https://"):
        raise AuthError("token_url must be https; a bearer token must not cross a plaintext hop")
    try:
        with urllib.request.urlopen(request, timeout=15) as response:  # noqa: S310
            payload = json.loads(response.read(1_000_000))
    except urllib.error.HTTPError as exc:
        raise AuthError(f"the identity provider refused the refresh ({exc.code})") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise AuthError(f"could not reach the identity provider: {exc}") from exc
    token = payload.get("access_token")
    if not token:
        raise AuthError("the identity provider returned no access_token")
    lifetime = float(payload.get("expires_in", 3600))
    cred = Credential(
        access_token=str(token), expires_at=(time.time() if now is None else now) + lifetime
    )
    rotated = payload.get("refresh_token")
    if rotated:
        # Providers that rotate refresh tokens invalidate the old one on use. Losing the new one
        # here would sign the user out on the next refresh, with nothing to explain why.
        write_refresh_token(config, str(rotated))
    return cred


def access_token(config: dict[str, Any], *, now: float | None = None) -> Credential:
    """A usable access token: cached if fresh, refreshed if not, `AuthError` if neither."""
    cached = cached_access_token(now=now)
    if cached is not None:
        return cached
    stored = read_refresh_token(config)
    if not stored:
        raise AuthError("no stored credential; run `recall-hooks login`")
    cred = refresh(config, stored, now=now)
    store_access_token(cred)
    return cred


def headers(config: dict[str, Any], *, now: float | None = None) -> dict[str, str]:
    """The Authorization header, or raise `AuthError`."""
    return {"Authorization": f"Bearer {access_token(config, now=now).access_token}"}


def print_auth_headers(
    config: dict[str, Any], out: Any = None, err: Any = None
) -> int:
    """The `headersHelper` contract: JSON on stdout, and **exit 0 whatever happens**.

    ⛔ A non-zero exit from a headers helper makes the MCP client treat the server as broken, so an
    expired token would read as "recall is broken" rather than as "recall needs a login". Printing
    an empty object lets the client fail the request with a 401 instead, which is the accurate
    thing and the one the user can act on. The diagnosis goes to stderr, where the client shows it
    without interpreting it as the header.
    """
    out = out or sys.stdout
    err = err or sys.stderr
    try:
        print(json.dumps(headers(config)), file=out)
    except AuthError as exc:
        print("{}", file=out)
        print(f"recall: no credential ({exc})", file=err)
    except Exception as exc:  # noqa: BLE001 - the exit code is the contract, not the taxonomy
        print("{}", file=out)
        print(f"recall: credential lookup failed ({type(exc).__name__}: {exc})", file=err)
    return 0


# --------------------------------------------------------------------------- the human commands
#
# ⚠️ These are the OPPOSITE of the hook events in how they fail. A hook must degrade to silence and
# exit 0, because a hook that reports an error takes a session down. A command a person typed must
# do the reverse: say what went wrong and exit non-zero, because its whole purpose is to tell them
# whether they are signed in. Reusing the hooks' habit here is what produced the defect this code
# fixes, a `login` that did not exist and returned 0.


def login(
    config: dict[str, Any],
    argv: list[str] | None = None,
    *,
    stdin: Any = None,
    out: Any = None,
    err: Any = None,
    now: float | None = None,
) -> int:
    """Store a refresh token and prove it works.

    Exit codes are the point of this command, so they are specific: 0 stored AND verified against
    the token endpoint, 3 stored but NOT verified, 2 nothing was stored.

    ⛔ 3 is not a warning dressed as success. "Signed in" that cannot fetch a token is the same
    false success as the missing command this replaces: the user follows the instruction, sees it
    succeed, and their sync keeps failing. Being offline is the benign case and it still gets a 3,
    because from here the two are indistinguishable and only one of them is safe to assume.
    """
    out = out or sys.stdout
    err = err or sys.stderr
    args = list(argv or [])

    token = ""
    if "--token" in args:
        index = args.index("--token")
        if index + 1 >= len(args):
            print("recall: --token needs a value", file=err)
            return 2
        token = args[index + 1].strip()
    else:
        source = stdin if stdin is not None else sys.stdin
        try:
            token = source.read().strip()
        except (OSError, ValueError):
            token = ""
    if not token:
        print(
            "recall: no token given. Paste the token from your dashboard:\n"
            "    recall-hooks login < token.txt\n"
            "`--token` exists for scripts and puts the credential in your shell history.",
            file=err,
        )
        return 2

    try:
        auth_config(config)
    except AuthError as exc:
        # Storing it would be worse than refusing: nothing could ever exchange it, and the user
        # would be told they are signed in.
        print(f"recall: {exc}", file=err)
        return 2

    where = write_refresh_token(config, token)
    if where == "file":
        # 🔑 Said out loud, always. A silent downgrade from an OS keychain to a plaintext file is
        # not a decision to make on somebody's behalf.
        print(
            f"recall: no OS keychain available, so the token is in {fallback_path()} "
            "(plaintext, permissions 0600). Install the `desktop` extra for keychain storage.",
            file=err,
        )
    else:
        print("recall: token stored in the OS keychain.", file=out)

    try:
        cred = refresh(config, token, now=now)
    except AuthError as exc:
        print(f"recall: stored, but it could not be verified: {exc}", file=err)
        return 3
    except Exception as exc:  # noqa: BLE001 - offline and rejected look the same from here
        print(
            f"recall: stored, but it could not be verified ({type(exc).__name__}: {exc}). "
            "If this machine is offline, the next session will try again.",
            file=err,
        )
        return 3
    store_access_token(cred)
    print(f"recall: signed in as {_account(config)}.", file=out)
    return 0


def logout(config: dict[str, Any], *, out: Any = None) -> int:
    """Remove the stored credential from every place this module puts one.

    Both stores are cleared whatever the keychain says, because `write_refresh_token` falls back
    to the file and a later install of `keyring` would otherwise leave a stale plaintext token
    behind that nothing reads and nobody remembers.
    """
    out = out or sys.stdout
    removed = []
    account = _account(config)
    ring = _keyring()
    if ring is not None:
        try:
            ring.delete_password(KEYRING_SERVICE, account)
            removed.append("keychain")
        except Exception:
            pass
    path = fallback_path()
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(stored, dict) and account in stored:
            del stored[account]
            _write_private_json(path, stored)
            removed.append(str(path))
    except (OSError, json.JSONDecodeError):
        pass
    try:
        token_cache_path().unlink(missing_ok=True)
    except OSError:
        pass
    print(
        f"recall: removed the credential from {', '.join(removed)}."
        if removed
        else "recall: no stored credential to remove.",
        file=out,
    )
    return 0

