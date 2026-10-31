"""Offline consumer coverage for the installer model probe."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


def test_selftest_skips_unrelated_cache_and_probes_target_offline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """BUG-003: self-test must never provision weights because another model is cached.

    Consumer: recall.desktop.main.install_main -> _selftest. Baseline: 160a23a3f.
    The intended baseline failure is the no-provisioning assertion after an unrelated cache
    entry makes the old _model_cache_exists return True. Only the external UI/native payload
    is replaced, so this needs neither PySide6 nor weights and still reaches the real resolver
    on the baseline. The same consumer then checks that a broken target cache fails and a
    usable target is actually embedded with local_files_only=True.
    Red/green execution belongs on the remote test runner; no execution is claimed by this docstring.
    """
    from recall.desktop import main as desktop_main
    from recall.desktop import ui

    cache = tmp_path / "fastembed"
    (cache / "models--someone--another-model").mkdir(parents=True)
    monkeypatch.setenv("FASTEMBED_CACHE_PATH", str(cache))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "huggingface"))
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.delenv("RECALL_EMBED_PROFILE", raising=False)
    monkeypatch.chdir(tmp_path)

    def module(name: str, **members: object) -> None:
        replacement = ModuleType(name)
        replacement.__dict__.update(members)
        monkeypatch.setitem(sys.modules, name, replacement)

    class Window:
        _fields: dict[str, object] = {}

        def __init__(self, **kwargs: object) -> None:
            pass

        def _answers(self) -> dict[str, object]:
            return {}

        def close(self) -> None:
            pass

    module("recall.desktop.install_ui", InstallerWindow=Window)
    monkeypatch.setattr(ui, "application_and_window", lambda factory: (object(), factory()))
    module(
        "recall.wizard.questions",
        question_plan=lambda **kwargs: [],
        build_config=lambda answers: dict.fromkeys(
            ("docs_root", "code_root", "memory_root", "corpus_version")
        ),
    )
    module("recall.wizard.headless", load_config=object(), run_headless=object())
    module("recall.wizard.pipeline", run_corpus=object())
    module("recall.wizard.uninstall", plan_uninstall=object())

    state = SimpleNamespace(broken=False)
    constructions: list[dict[str, object]] = []
    provisioning: list[str] = []
    probes: list[list[str]] = []

    class TextEmbedding:
        def __init__(self, **kwargs: object) -> None:
            constructions.append(kwargs)
            if not kwargs.get("local_files_only"):
                provisioning.append(str(kwargs.get("model_name")))
            if state.broken:
                raise RuntimeError("the provisioned ONNX model cannot load")

        def embed(self, texts: list[str]) -> list[list[float]]:
            probes.append(texts)
            return [[0.0] * 384 for _ in texts]

    module("fastembed", TextEmbedding=TextEmbedding)
    module("onnxruntime")
    module("tokenizers")

    result = desktop_main.install_main(["--selftest"])
    assert provisioning == [], "an unrelated cache must not trigger model provisioning"
    assert result == 0
    assert constructions == [], "the target model is absent, so its optional probe must skip"
    assert "NOT resolved" in capsys.readouterr().err

    target = cache / "models--Qdrant--bge-small-en-v1.5-onnx-Q"
    target.mkdir()
    state.broken = True
    assert desktop_main.install_main(["--selftest"]) == 1
    assert "the provisioned ONNX model cannot load" in capsys.readouterr().err
    assert constructions[-1]["local_files_only"] is True

    state.broken = False
    assert desktop_main.install_main(["--selftest"]) == 0
    assert constructions[-1]["local_files_only"] is True
    assert Path(str(constructions[-1]["cache_dir"])) == cache
    assert constructions[-1]["model_name"] == "BAAI/bge-small-en-v1.5"
    assert probes == [["probe"]], "a usable cached model must run the native embedding path"
    assert provisioning == []
