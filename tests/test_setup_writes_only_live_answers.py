"""`recall setup` writes no answer that nothing reads, and offers no step that never reaches search.

Three issues from the 2026-11-19 audit of settings whose value never reaches the code:

- #1396: setup wrote `RECALL_SECURITY_REQUIRED`, which nothing reads after setup. The owner chose
  option A: keep the question (it filters the cloud choices during the run), stop saving it, and
  say in the summary that the answer applies to this run only.
- #1398: setup always wrote `RECALL_SPARSE=fts`, the built-in default, which nothing reads.
- #1395: setup offered to "calibrate the threshold now" and wrote a legacy `calibration.json`
  that search never loads (`_cmd_search` passes `calibration=None`; only a calibration published
  against a generation serves). Setup now points at `recall wizard` instead.

Red proof receipt ``setup-live-answers-01``, 2026-11-19: each test was run against
`34dd229cb` (PR 1411's head, setup still writing both keys and asking to calibrate) and failed in
its first assertion: `RECALL_SECURITY_REQUIRED=0` in `.env`, `RECALL_SPARSE=fts` in `.env`, and a
prompt containing "calibrate the threshold". Green after the change to `run_setup_wizard`.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from recall.seed import SeedPlan
from recall.setup import HardwareProbe, run_setup_wizard


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch: pytest.MonkeyPatch) -> None:
    probe = HardwareProbe(
        cpu_count=8,
        gpu=None,
        free_bytes=100 * 2**30,
        internet=True,
        fastembed_available=True,
        sentence_transformers_available=True,
    )
    monkeypatch.setattr("recall.setup.probe_hardware", lambda: probe)
    monkeypatch.setattr("recall.setup._module_available", lambda name: True)
    monkeypatch.setattr("recall.setup.claude_code_detected", lambda: False)
    monkeypatch.setattr("recall.setup.codex_code_detected", lambda: False)
    monkeypatch.setattr("recall.setup.plugin_skill_sources", lambda: {})
    monkeypatch.setattr(
        "recall.setup.plan_seed",
        lambda root, **kw: SeedPlan(root=Path(root), files=(), total_bytes=0),
    )
    monkeypatch.setattr(
        "recall.setup.calibrate_from_files",
        lambda **kw: (_ for _ in ()).throw(AssertionError("setup must not fit a calibration")),
    )


def _run(
    tmp_path: Path, *, security: str, accept_calibration: bool = False
) -> tuple[str, str, list[str]]:
    """Run setup answering by the prompt just printed; return (.env text, output, prompts)."""
    output = io.StringIO()
    last = {"text": ""}
    prompts: list[str] = []

    def _print(*args, **kwargs) -> None:
        last["text"] = " ".join(str(a) for a in args)
        print(*args, **kwargs, file=output)

    def _answer(prompt: str = "") -> str:
        shown = prompt or last["text"]
        prompts.append(shown)
        if "data security necessary" in shown:
            return security
        if shown.endswith("Select a number: "):
            return "1"
        if shown.rstrip().endswith("]"):
            return "y" if accept_calibration and "calibrat" in shown.lower() else "n"
        return ""

    run_setup_wizard(
        dsn="postgresql://example/recall",
        env_path=tmp_path / ".env",
        claude_md_path=tmp_path / "CLAUDE.md",
        memory_dir=tmp_path / "memory",
        input_fn=_answer,
        print_fn=_print,
    )
    return (tmp_path / ".env").read_text(encoding="utf-8"), output.getvalue(), prompts


@pytest.mark.parametrize("security", ["y", "n"])
def test_the_security_answer_is_not_saved_and_the_summary_says_so(
    tmp_path: Path, security: str
) -> None:
    env, out, _ = _run(tmp_path, security=security)

    assert "RECALL_SECURITY_REQUIRED" not in env
    expected = "yes" if security == "y" else "no"
    assert f"Data security required: {expected}." in out
    assert "applied to this setup run only" in out


def test_setup_does_not_write_recall_sparse(tmp_path: Path) -> None:
    env, _, _ = _run(tmp_path, security="y")

    assert "RECALL_SPARSE" not in env
    # Positive control: the block was written, so the absence above is not an empty file.
    assert "RECALL_DSN=postgresql://example/recall" in env


def test_setup_offers_no_calibration_step_and_points_at_the_wizard(tmp_path: Path) -> None:
    # A calibration offer, if made, is accepted, so it would also reach the patched
    # `calibrate_from_files`, which raises.
    env, out, prompts = _run(tmp_path, security="y", accept_calibration=True)

    assert not [p for p in prompts if "calibrat" in p.lower()], prompts
    assert "RECALL_CALIBRATION" not in env
    assert "Run recall wizard" in out
