"""The desktop configuration page does not offer a sparse backend that no search would use.

Invariant: the page offers no SPLADE control and leaves no `splade` choice in `pipeline.json`.
Failure mode it catches: the page had an "Enable learned sparse SPLADE retrieval" checkbox whose
value was saved to `pipeline.json`, but the only reader of that file is the page restoring its own
widgets (`recall.desktop.profiles.load_pipelines`), and no call outside `recall/eval/` passes
`sparse_backend=` or `sparse_encoder=`, so ticking it changed nothing. The owner chose, on
2026-11-19, to stop offering it, as `recall setup` did; SPLADE stays a library option through
`HybridRetriever(sparse_backend=...)`.

Red proof receipt ``desktop-no-splade-offer-01``, 2026-11-19: run against `6099ded6b` (the page
with `MainWindow.splade_check`), the test failed on its first assertion, because the page showed
the SPLADE checkbox and its "SPLADE" form row. Green after the checkbox was removed. Production
symbols targeted: `MainWindow._build_configuration_page` (the checkbox) and `MainWindow.__init__`
(the saved choices layered over the defaults).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import recall.desktop.profiles as profiles
from recall.desktop.models import RuntimeMode, RuntimeProfile


class _UiRuntime:
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def health(self) -> dict[str, str]:
        return {"status": "ready"}
    def list_tenants(self) -> list[str]:
        return ["default"]


def test_the_configuration_page_neither_offers_nor_keeps_a_splade_choice(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication, QWidget

    from recall.desktop.ui import MainWindow

    monkeypatch.setattr(profiles, "profile_path", lambda: tmp_path / "runtime.json")
    # A file saved by an earlier version, when the checkbox existed and was ticked.
    pipeline = tmp_path / "pipeline.json"
    pipeline.write_text(
        json.dumps({kind: {"splade": True} for kind in ("Documents", "Memory", "Code")}),
        encoding="utf-8",
    )
    app = QApplication.instance() or QApplication([])
    window = MainWindow(
        RuntimeProfile(mode=RuntimeMode.DOCKER, compose_file="docker-compose.desktop.yml"),
        runtime=_UiRuntime(),
    )
    try:
        shown = [
            str(widget.text())
            for widget in window.findChildren(QWidget)
            if callable(getattr(widget, "text", None))
        ]
        assert not [text for text in shown if "splade" in text.lower()], shown

        window._save_configuration()
        assert "saved" in window.status.text().lower(), window.status.text()
    finally:
        window.close()
        app.processEvents()

    saved = json.loads(pipeline.read_text(encoding="utf-8"))
    assert all("splade" not in choices for choices in saved.values()), saved
