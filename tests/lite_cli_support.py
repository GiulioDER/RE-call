"""What the lite CLI test modules share: running `recall` in process, and an offline environment."""

from __future__ import annotations

import contextlib
import io

import pytest

from recall.cli import main


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """No route, DSN or trust mode from the developer's shell, and no shared embedding cache."""
    for name in ("RECALL_ENV", "RECALL_INDEX_MODE", "RECALL_SERVING_DSN", "RECALL_DSN", "RECALL_TRUST_MODE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("RECALL_EMBED_CACHE", "off")


def run_cli(argv: list[str]) -> str:
    """The command's stdout, or the exit message it stopped with: either way something to assert."""
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            main(argv)
    except SystemExit as exc:
        return out.getvalue() + f"\nEXIT: {exc.code}"
    except Exception as exc:  # noqa: BLE001  # the pre-wiring failure is an arbitrary driver error
        return out.getvalue() + f"\nRAISED: {type(exc).__name__}: {exc}"
    return out.getvalue()
