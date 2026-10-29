"""Offline regressions for CCA PRs 759 through 750.

Red proof baseline: 06c1603a9, production files in .prefix_snapshot.
Each named behavior test fails its assertion (or pytest's DID NOT RAISE
assertion) on that baseline, then passes with FIX-001 through FIX-005.
Transport, SSH and privilege changes are fakes; no provider or service is called.
"""
from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

from hypothesis import given, strategies as st
import httpx
import pytest

from benchmarks.beam import aml_c9_probe as probe
from scripts import aml_add_throughput as throughput
from scripts import aml_c9_mechanism_bench as mechanism


ROOT = Path(__file__).resolve().parents[1]


def _launcher(pattern: str) -> Path:
    """Find the public launcher without duplicating its environment-specific suffix."""
    matches = list((ROOT / "scripts").glob(pattern))
    assert len(matches) == 1, matches
    return matches[0]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX launcher contract")
def test_credentials_do_not_enter_runuser_arguments(tmp_path: Path) -> None:
    """FIX-001: both original runuser commands put the canary into argv."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "runuser"
    stub.write_text(
        f"#!{sys.executable}\nimport json,os,sys\n"
        "with open(os.environ['CAPTURE'], 'a') as f:\n"
        " f.write(json.dumps({'argv':sys.argv[1:], 'aml':os.getenv('RECALL_AML_API_KEY'),"
        " 'router':os.getenv('OPENROUTER_API_KEY')})+'\\n')\n",
        encoding="utf-8",
    )
    stub.chmod(0o755)
    key = tmp_path / "key"
    key.write_text("canary-aml", encoding="utf-8")
    env_file = tmp_path / "provider.env"
    env_file.write_text("OPENROUTER_API_KEY=canary-router\n", encoding="utf-8")
    capture = tmp_path / "calls.jsonl"
    source = _launcher("aml_add_speed_*.sh").read_text(encoding="utf-8")
    arm = "run_arm() {" + source.split("run_arm() {", 1)[1].split("\n}\n", 1)[0] + "\n}"
    provider = source.split("# Measurement 2.\n", 1)[1].split("\nsha256sum", 1)[0]
    provider = provider.replace(". /home/$U/recall-tb/.env", '. "$PROVIDER_ENV"')
    preamble = 'set -euo pipefail\nstart_server() { :; }; stop_server() { :; }\n'
    environment = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}",
                   "CAPTURE": str(capture), "U": "test-user", "PY": sys.executable,
                   "CODE": str(tmp_path), "B": str(tmp_path), "OUT": str(tmp_path),
                   "RUN": "test", "URL": "http://unused", "KEY_FILE": str(key),
                   "PROVIDER_ENV": str(env_file)}
    subprocess.run(["bash", "-c", preamble + arm + "\nrun_arm 0 k8 8\n" + provider],
                   env=environment, check=True, capture_output=True, text=True)
    calls = [json.loads(line) for line in capture.read_text().splitlines()]
    assert len(calls) == 2
    assert all("canary-" not in " ".join(call["argv"]) for call in calls), calls
    assert calls[0]["aml"] == "canary-aml"
    assert calls[1]["router"] == "canary-router"


def _provider(monkeypatch: pytest.MonkeyPatch, transport) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-canary")
    monkeypatch.setattr(probe, "http_json", transport)
    monkeypatch.setattr(probe.time, "sleep", lambda _: None)


def _response():
    return 200, {"choices": [{"message": {"content": "ok"}}],
                 "usage": {"prompt_tokens": 1, "completion_tokens": 1}}, {}


def test_zero_budget_sends_no_provider_request(monkeypatch: pytest.MonkeyPatch) -> None:
    """FIX-004: baseline complete sends one request with Spend(0)."""
    calls = []
    _provider(monkeypatch, lambda *args: (calls.append(args), _response())[1])
    refused = False
    try:
        probe.complete(probe.Spend(0), [{"role": "user", "content": "x"}], 1)
    except SystemExit:
        refused = True
    assert refused and not calls


@pytest.mark.parametrize("cap", [-1.0, float("nan"), float("inf")])
def test_invalid_budget_is_rejected(cap: float) -> None:
    """FIX-004: baseline accepts negative and nonfinite allowance."""
    with pytest.raises(ValueError):
        probe.Spend(cap)


def test_competing_calls_cannot_spend_one_allowance_twice(monkeypatch: pytest.MonkeyPatch) -> None:
    """FIX-004: baseline lets both requests enter before either accounts usage."""
    entered = threading.Event()
    release = threading.Event()
    calls = []

    def transport(*args):
        calls.append(args)
        if len(calls) == 1:
            entered.set()
            assert release.wait(5)
        return _response()

    _provider(monkeypatch, transport)
    spend = probe.Spend(0.0006)
    messages = [{"role": "user", "content": "x"}]
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(probe.complete, spend, messages, 2000)
        assert entered.wait(5)
        second = executor.submit(probe.complete, spend, messages, 2000)
        try:
            with pytest.raises(SystemExit):
                second.result(timeout=5)
        finally:
            release.set()
            first.result(timeout=5)
        assert len(calls) == 1


def test_unknown_retry_expenditure_keeps_its_allowance(monkeypatch: pytest.MonkeyPatch) -> None:
    """FIX-004: baseline sends six requests despite a one-attempt allowance."""
    calls = []
    _provider(monkeypatch, lambda *args: (calls.append(args), (599, {}, {}))[1])
    try:
        probe.complete(probe.Spend(0.0006), [{"role": "user", "content": "x"}], 2000)
    except (SystemExit, RuntimeError):
        pass
    assert len(calls) == 1


def test_missing_usage_stops_paid_work(monkeypatch: pytest.MonkeyPatch) -> None:
    """FIX-004: baseline treats absent usage as zero cost and returns success."""
    _provider(monkeypatch, lambda *args: (200, {"choices": [{"message": {"content": "ok"}}]}, {}))
    refused = False
    try:
        probe.complete(probe.Spend(1), [{"role": "user", "content": "x"}], 1)
    except (SystemExit, RuntimeError):
        refused = True
    assert refused


@pytest.mark.parametrize("cleanup_failure", [False, True])
def test_failed_throughput_settles_workers_and_cleans_every_user(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cleanup_failure: bool,
) -> None:
    """FIX-002: baseline skips every deletion when one Add raises ConnectError."""
    data = tmp_path / "data.json"
    data.write_text("[]", encoding="utf-8")
    deleted = []
    active = set()

    class Client:
        async def __aenter__(self):
            self.started = asyncio.Event()
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, path):
            return httpx.Response(200, json={"variant": "test"})

        async def post(self, path, *, json):
            user = json["user_id"]
            if path == "/v1/delete":
                assert not active, "Add workers must settle before deletion"
                deleted.append(user)
                if cleanup_failure and user.endswith("u0"):
                    raise httpx.ConnectError("delete failed")
                return httpx.Response(200, request=httpx.Request("POST", "http://fake/delete"))
            if user.endswith("u1"):
                await self.started.wait()
                raise httpx.ConnectError("original Add failure")
            if json["session_id"].endswith("s0"):
                return httpx.Response(200)
            active.add(user)
            self.started.set()
            try:
                await asyncio.Event().wait()
            finally:
                active.remove(user)

    monkeypatch.setenv("RECALL_AML_API_KEY", "offline-canary")
    monkeypatch.setattr(throughput.httpx, "AsyncClient", lambda **kwargs: Client())
    monkeypatch.setattr(throughput, "arm_sessions", lambda *args: [[{"content": "x"}]] * 4)
    args = argparse.Namespace(data=data, users=2, arm=0, arm_label="k8", run_id="offline",
                              client_timeout=1, base_url="http://fake", expected_variant="test",
                              max_attempts=1, backoff=0)
    with pytest.raises(httpx.ConnectError, match="original Add failure"):
        asyncio.run(throughput.main_async(args))
    assert deleted == ["c9-speed-offline-k8-u0", "c9-speed-offline-k8-u1"]


@given(spread=st.floats(min_value=0.01, max_value=30))
def test_add_median_is_midpoint(spread: float) -> None:
    """FIX-005: baseline upper-middle percentile gives 61 for spread=1."""
    rows = [dict(seconds=s, status=200, statuses=[200], compiler_fallback=False,
                 compiled_count=0) for s in (60 - spread, 60 + spread)]
    assert mechanism._add_stats(rows, 120)["seconds_median"] == 60
    assert mechanism._add_stats([], 120)["seconds_median"] == 0


@given(spread=st.floats(min_value=0.01, max_value=30))
def test_search_median_is_midpoint(spread: float) -> None:
    """FIX-005: baseline upper-middle percentile gives 61 for spread=1."""
    rows = [dict(seconds=s, status=200, statuses=[200], key="needle", rank=1,
                 graph_relation_hits=0, graph_promoted=0, graph_invalid_relations=0,
                 atomic_fallback=False) for s in (60 - spread, 60 + spread)]
    assert mechanism._search_stats(rows, 120)["seconds_median"] == 60
    assert mechanism._search_stats([], 120)["seconds_median"] == 0


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX launcher contract")
@pytest.mark.parametrize("bench_exit,passed", [(1, False), (7, True), (0, False)])
def test_failed_mechanism_wrapper_returns_failure(
    tmp_path: Path, bench_exit: int, passed: bool,
) -> None:
    """FIX-003: baseline masks both the remote status and failed JSON report."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    report = tmp_path / "remote.json"
    remote_bench = tmp_path / "bench.py"
    remote_bench.write_text(
        "import json,os,sys\nfrom pathlib import Path\n"
        "Path(sys.argv[sys.argv.index('--out')+1]).write_text(json.dumps("
        "{'passed':os.environ['PASSED']=='1','failed_checks':[], 'summary':{}}))\n"
        "raise SystemExit(int(os.environ['BENCH_EXIT']))\n", encoding="utf-8")
    env_file = tmp_path / "server.env"
    env_file.write_text("RECALL_AML_API_KEY=canary\nRECALL_AML_PORT=1234\n", encoding="utf-8")
    stub_code = (
        f"#!{sys.executable}\nimport json,os,sys,subprocess,shutil\nfrom pathlib import Path\n"
        "name=Path(sys.argv[0]).name\na=sys.argv[1:]\n"
        "if name=='ssh':\n"
        " if 'bash' in a and '-s' in a:\n"
        "  result=subprocess.run(['bash','-s','--',os.environ['SERVER_ENV'],os.environ['BENCH'],"
        "os.environ['REPORT'],os.environ['ERR']],input=sys.stdin.read(),text=True)\n"
        "  raise SystemExit(result.returncode)\n"
        " command=' '.join(a)\n"
        " if 'grep -cE' in command: print('0')\n"
        " elif 'date -u' in command: print('2026-10-28 00:00:00')\n"
        " elif 'is-active' in command: print('active')\n"
        "elif name=='scp':\n"
        " if a[-2].startswith('fake:'): shutil.copyfile(os.environ['REPORT'],a[-1])\n"
        "elif name=='curl':\n"
        " print(json.dumps({'git_commit':'12345678','variant':'test','graph_sidecar':True,"
        "'atomic_rescue':{'mode':'test'}}) if '/version' in a[-1] else 'health 200')\n"
    )
    for name in ("ssh", "scp", "curl"):
        stub = bin_dir / name
        stub.write_text(stub_code, encoding="utf-8")
        stub.chmod(0o755)
    for name in ("python", "python3"):
        (bin_dir / name).symlink_to(sys.executable)
    environment = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}",
                   "C9_ROOT_HOST": "fake", "C9_SSH_KEY": "unused", "C9_PUBLIC_URL": "http://fake",
                   "C9_BENCH_OUT": str(tmp_path / "local.json"), "PASSED": str(int(passed)),
                   "BENCH_EXIT": str(bench_exit), "REPORT": str(report), "BENCH": str(remote_bench),
                   "SERVER_ENV": str(env_file), "ERR": str(tmp_path / "remote.err")}
    result = subprocess.run(["bash", str(_launcher("aml_c9_mechanism_bench_*.sh"))],
                            env=environment, capture_output=True, text=True)
    assert result.returncode != 0, result.stdout + result.stderr
