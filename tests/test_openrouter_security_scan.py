"""The OpenRouter security review must parse the replies the model actually sends.

Invariant: a reply that contains a complete findings object is reviewed, not reported as a parse
failure; a reply that cannot be parsed says why in the log; an endpoint that returns no content is
retried rather than failing the review.

The shapes are the ones observed 2026-09-22 when the base-revision payload was replayed against the
PR 694 diff: 6 of 6 replies failed ``_parse_response``. Five were a complete object followed by an
unopened closing fence, and one was ``content: null`` with 526 completion tokens billed.

Red proof, recorded 2026-09-22 by running this file against ``origin/master``'s script at
``f2844361`` (the pre-fix ``_parse_response`` and ``_request_review``):

- ``test_object_followed_by_a_stray_closing_fence_is_parsed``: RuntimeError "did not return the
  required findings JSON object" (the defect itself).
- ``test_object_after_prose_is_parsed``: same RuntimeError.
- ``test_a_brace_in_prose_before_the_object_is_skipped``: same RuntimeError.
- ``test_parse_failure_logs_a_bounded_redacted_prefix``: AssertionError, the message carried no
  prefix of the reply.
- ``test_request_asks_for_json_mode_on_supporting_endpoints_only``: AssertionError,
  ``response_format`` was None (the first assertion; ``max_tokens`` was 2000 and was not reached).
- ``test_null_content_is_retried``: RuntimeError "non-text response" on the first reply.
- ``test_null_content_twice_names_the_provider``: AssertionError, the message was "non-text
  response" and did not say the content was empty.
- ``test_truncated_reply_is_reported_as_truncation``: AssertionError, message was the generic
  parse error rather than truncation.

The redaction assertion was additionally proven by mutating ``_redacted_prefix`` to skip
``_SECRET_PATTERN.sub``: the test failed on the token appearing in the message.
"""

from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Iterator

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / ".github" / "scripts" / "openrouter_security_scan.py"

FINDING = {
    "severity": "medium",
    "title": "Example",
    "file": "recall/example.py",
    "line": 7,
    "description": "An exploit path.",
    "recommendation": "A fix.",
}
OBJECT = json.dumps({"findings": [FINDING]}, indent=2)


@pytest.fixture()
def scan() -> Iterator[ModuleType]:
    spec = importlib.util.spec_from_file_location("openrouter_security_scan_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(spec.name, None)


def _reply(content: object, *, provider: str = "SiliconFlow", finish: str = "stop", tokens: int = 200) -> bytes:
    return json.dumps(
        {
            "provider": provider,
            "choices": [{"finish_reason": finish, "message": {"role": "assistant", "content": content}}],
            "usage": {"completion_tokens": tokens},
        }
    ).encode()


class _FakeResponse(io.BytesIO):
    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _install(monkeypatch: pytest.MonkeyPatch, scan: ModuleType, replies: list[bytes]) -> list[dict[str, Any]]:
    sent: list[dict[str, Any]] = []

    def fake_urlopen(request: Any, timeout: float) -> _FakeResponse:
        sent.append(json.loads(request.data))
        return _FakeResponse(replies[len(sent) - 1])

    monkeypatch.setattr(scan, "urlopen", fake_urlopen)
    return sent


def _review(scan: ModuleType) -> str:
    return scan._request_review("key", scan.DEFAULT_MODEL, "owner/repo", "1", "diff --git a b")


def test_object_followed_by_a_stray_closing_fence_is_parsed(scan: ModuleType) -> None:
    findings = scan._parse_response(OBJECT + "\n```")
    assert [item["title"] for item in findings] == ["Example"]


def test_object_after_prose_is_parsed(scan: ModuleType) -> None:
    findings = scan._parse_response("Here is the security review:\n\n" + OBJECT + "\nThat is all.")
    assert [item["severity"] for item in findings] == ["medium"]


def test_a_brace_in_prose_before_the_object_is_skipped(scan: ModuleType) -> None:
    content = "The diff sets {mode} and {\"other\": 1} before the answer.\n" + OBJECT
    assert len(scan._parse_response(content)) == 1


def test_parse_failure_logs_a_bounded_redacted_prefix(scan: ModuleType) -> None:
    token = "sk-or-v1-" + "a" * 40
    content = f"I cannot comply with key {token}. " + "x" * 5_000
    with pytest.raises(RuntimeError) as excinfo:
        scan._parse_response(content)
    message = str(excinfo.value)
    assert "I cannot comply with key" in message
    assert token not in message
    assert len(message) < 1_000


def test_request_asks_for_json_mode_on_supporting_endpoints_only(
    scan: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent = _install(monkeypatch, scan, [_reply(OBJECT)])
    _review(scan)
    assert sent[0].get("response_format") == {"type": "json_object"}
    assert sent[0].get("provider", {}).get("require_parameters") is True
    assert sent[0]["max_tokens"] >= 8_000


def test_null_content_is_retried(scan: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    sent = _install(monkeypatch, scan, [_reply(None, provider="Novita", tokens=526), _reply(OBJECT)])
    assert scan._parse_response(_review(scan))[0]["title"] == "Example"
    assert len(sent) == 2


def test_null_content_twice_names_the_provider(scan: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, scan, [_reply(None, provider="Novita", tokens=526)] * 2)
    with pytest.raises(RuntimeError) as excinfo:
        _review(scan)
    assert "empty content" in str(excinfo.value)
    assert "provider=Novita" in str(excinfo.value)


def test_truncated_reply_is_reported_as_truncation(scan: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, scan, [_reply(OBJECT[:40], finish="length", tokens=8_000)])
    with pytest.raises(RuntimeError) as excinfo:
        scan._parse_response(_review(scan))
    assert "truncated" in str(excinfo.value)


# Model HIGH findings are advisory by default.
#
# Invariant: a model-reported high finding is shown (annotation and step summary) but does not fail
# the job unless OPENROUTER_FAIL_ON_HIGH opts in; a review that cannot run or parse still fails.
# Measured 2026-09-22: 18 of 20 failed runs exited 1 on a HIGH the changed code contradicted.
#
# Red proof, recorded 2026-09-22:
# - test_a_model_high_is_advisory_by_default: against origin/master at c5124500, AssertionError,
#   main() returned 1.
# - test_opting_in_makes_a_model_high_blocking: mutation, _fail_on_high returning False,
#   AssertionError, main() returned 0.
# - test_an_unparseable_review_still_fails_the_job: mutation, main's except branch returning 0,
#   AssertionError, main() returned 0.

HIGH = json.dumps({"findings": [dict(FINDING, severity="high", title="Model claim")]})


def _main_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    diff = tmp_path / "pr.diff"
    diff.write_text("diff --git a/x b/x\n+changed\n", encoding="utf-8")
    summary = tmp_path / "summary.md"
    for name, value in {
        "OPENROUTER_API_KEY": "key",
        "PR_DIFF_PATH": str(diff),
        "GITHUB_REPOSITORY": "owner/repo",
        "PR_NUMBER": "1",
        "GITHUB_STEP_SUMMARY": str(summary),
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("OPENROUTER_FAIL_ON_HIGH", raising=False)
    return summary


def test_a_model_high_is_advisory_by_default(
    scan: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    summary = _main_env(monkeypatch, tmp_path)
    _install(monkeypatch, scan, [_reply(HIGH)])
    assert scan.main() == 0
    assert "::error file=recall/example.py,line=7,title=Model claim::" in capsys.readouterr().out
    assert "Model claim" in summary.read_text(encoding="utf-8")


def test_opting_in_makes_a_model_high_blocking(
    scan: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _main_env(monkeypatch, tmp_path)
    monkeypatch.setenv("OPENROUTER_FAIL_ON_HIGH", "1")
    _install(monkeypatch, scan, [_reply(HIGH)])
    assert scan.main() == 1


def test_an_unparseable_review_still_fails_the_job(
    scan: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _main_env(monkeypatch, tmp_path)
    _install(monkeypatch, scan, [_reply("no JSON here")])
    assert scan.main() == 1


def test_an_oversized_diff_passes_unreviewed_and_says_so(
    scan: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Invariant: a diff over ``MAX_DIFF_CHARS`` does not fail the job, never reaches the model,
    and is reported as NOT reviewed in the log and the job summary.

    Red proof, 2026-09-26, against ``main`` with this file unchanged: the pre-fix
    ``raise RuntimeError("... refusing to review ...")`` failed on ``scan.main() == 0`` (it
    returned 1); a mutation returning 0 without calling ``_report_not_reviewed`` (the silent
    skip) failed on the warning assertion.
    """
    summary = _main_env(monkeypatch, tmp_path)
    Path(tmp_path / "pr.diff").write_text("+x\n" * (scan.MAX_DIFF_CHARS // 3 + 1), encoding="utf-8")
    sent = _install(monkeypatch, scan, [])

    assert scan.main() == 0
    assert sent == []
    assert "::warning title=OpenRouter security review skipped::NOT REVIEWED" in capsys.readouterr().out
    assert "**Not reviewed.**" in summary.read_text(encoding="utf-8")


def test_a_diff_github_would_not_serve_passes_unreviewed_and_says_so(
    scan: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Invariant: when the fetch step recorded GitHub's refusal (HTTP 406 over 20,000 lines, PR 776),
    the job passes without calling the model and says NOT REVIEWED, naming the refusal.

    Red proof, 2026-09-26: without the ``PR_DIFF_UNAVAILABLE_PATH`` branch in ``main`` the empty diff
    reached "no diff to review" and returned 0 silently; this failed on the warning assertion.
    """
    summary = _main_env(monkeypatch, tmp_path)
    (tmp_path / "pr.diff").write_text("", encoding="utf-8")
    refusal = tmp_path / "pr.diff.unavailable"
    refusal.write_text("gh: Sorry, the diff exceeded the maximum number of lines (20000) (HTTP 406)\n", encoding="utf-8")
    monkeypatch.setenv("PR_DIFF_UNAVAILABLE_PATH", str(refusal))
    sent = _install(monkeypatch, scan, [])

    assert scan.main() == 0
    assert sent == []
    out = capsys.readouterr().out
    assert "::warning title=OpenRouter security review skipped::NOT REVIEWED: GitHub would not serve" in out
    assert "HTTP 406" in out
    assert "**Not reviewed.**" in summary.read_text(encoding="utf-8")


WORKFLOW = SCRIPT.parents[1] / "workflows" / "openrouter-security.yml"


def _fetch_step() -> str:
    """The ``run`` block of the workflow's "Fetch pull request diff" step, exactly as committed."""
    lines = WORKFLOW.read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == "- name: Fetch pull request diff")
    run = next(i for i in range(start, len(lines)) if lines[i].strip() == "run: |")
    indent = len(lines[run + 1]) - len(lines[run + 1].lstrip())
    body = []
    for line in lines[run + 1:]:
        if line.strip() and len(line) - len(line.lstrip()) < indent:
            break
        body.append(line[indent:])
    return "\n".join(body) + "\n"


def _bash() -> str:
    import shutil

    if sys.platform == "win32":
        git_bash = Path("C:/Program Files/Git/bin/bash.exe")
        if git_bash.is_file():
            return str(git_bash)
        pytest.skip("Git for Windows bash is not installed")
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is not installed on this platform")
    return bash


@pytest.mark.parametrize(
    ("fake_gh", "exit_code", "refused"),
    [
        ('echo "gh: Sorry, the diff exceeded the maximum number of lines (20000) (HTTP 406)" >&2; return 1', 0, True),
        ('echo "gh: Bad credentials (HTTP 401)" >&2; return 1', 1, False),
        ('echo "diff --git a/x b/x"', 0, False),
    ],
    ids=["406-is-recorded", "other-failure-fails", "diff-is-written"],
)
def test_the_fetch_step_records_only_a_406(tmp_path: Path, fake_gh: str, exit_code: int, refused: bool) -> None:
    """Invariant: the committed fetch step turns GitHub's 406 into a recorded refusal and exit 0, and
    every other ``gh`` failure still fails the step.

    Red proof, 2026-09-26, mutating the step in ``openrouter-security.yml`` with this file unchanged:
    the pre-fix step (``gh api ... > pr.diff`` under ``set -e``) failed ``406-is-recorded`` on the exit
    code (1, not 0); treating every failure as a refusal (``grep -q 'HTTP 406'`` replaced by ``true``)
    failed ``other-failure-fails`` on the exit code (0, not 1).
    """
    import subprocess

    script = f"gh() {{ {fake_gh}; }}\n" + _fetch_step()
    env = {"RUNNER_TEMP": tmp_path.as_posix(), "REPOSITORY": "owner/repo", "PR_NUMBER": "1",
           "PATH": __import__("os").environ["PATH"]}
    result = subprocess.run([_bash(), "-c", script], env=env, capture_output=True, text=True)

    assert result.returncode == exit_code, result.stderr
    assert (tmp_path / "pr.diff.unavailable").is_file() is refused
