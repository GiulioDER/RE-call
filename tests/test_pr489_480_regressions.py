"""Behavioral proofs for the PR 489 through 480 audit.

Baseline 882bd50c2: the runner exposes credentials and loses relative paths;
worker isolation returns a shared DSN after OperationalError; probe roles collide;
the BMI counter changes units with the optional tokenizer. Each test below fails
its stated assertion on that baseline, then passes with the repair.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def official_runner(tmp_path):
    """Run the actual shell entry point with external and paid transports stubbed."""
    if shutil.which("bash") is None:
        pytest.skip("the official runner requires bash")
    repo = tmp_path / "repo"
    scripts = repo / "benchmarks" / "benchd"
    scripts.mkdir(parents=True)
    shutil.copyfile(ROOT / "benchmarks/benchd/run_official.sh", scripts / "run_official.sh")
    (scripts / "recall_adapter.py").write_text("# identical adapter fixture\n")
    (scripts / "count_tokens.py").write_text("print('{}')\n")
    harness = tmp_path / "harness"
    adapter_dir = harness / "benchd_harness" / "adapters"
    adapter_dir.mkdir(parents=True)
    shutil.copyfile(scripts / "recall_adapter.py", adapter_dir / "recall_adapter.py")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    git = bin_dir / "git"
    git.write_text('#!/bin/sh\ncase "$*" in *status*) exit 0;; *) echo fixture-sha;; esac\n')
    benchd = bin_dir / "benchd"
    benchd.write_text(
        f"#!{sys.executable}\n"
        "import json, pathlib, sys\n"
        "if sys.argv[1] == 'run':\n"
        "    out = pathlib.Path(sys.argv[sys.argv.index('--out') + 1]) / 'run_fixture'\n"
        "    out.mkdir(parents=True)\n"
        "    (out / 'manifest.signed.json').write_text(json.dumps({'traces': []}))\n"
        "print('stub transport completed')\n"
    )
    git.chmod(0o755)
    benchd.chmod(0o755)
    home = tmp_path / "home"
    dataset = home / ".cache/benchd/locomo/locomo10.json"
    dataset.parent.mkdir(parents=True)
    dataset.write_text("[]")
    env = dict(os.environ)
    env.update(
        HOME=str(home),
        PATH=str(bin_dir) + os.pathsep + env.get("PATH", ""),
        VOYAGE_API_KEY="synthetic-voyage",
        OPENROUTER_API_KEY="synthetic-openrouter",
        RECALL_BENCHD_DSN="postgresql://fixture:synthetic-password@localhost/example",
        RECALL_BENCHD_UNKNOWN_SECRET="synthetic-unknown-secret",
        RECALL_BENCHD_DRYRUN="0",
    )

    def run(*, relative=False):
        result = subprocess.run(
            ["bash", str(scripts / "run_official.sh"), "harness" if relative else str(harness),
             "locomo-v1", "1"],
            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30,
        )
        return result, harness

    return run


def test_official_record_excludes_credentials(official_runner):
    """FIX-004/SEC-001: baseline fails because the synthetic DSN is in config."""
    result, harness = official_runner()
    assert result.returncode == 0, result.stdout + result.stderr
    records = list(harness.glob("runs-official/*/run-record.json"))
    assert len(records) == 1
    text = records[0].read_text()
    config = json.loads(text)["config"]
    assert "RECALL_BENCHD_DSN" not in config
    assert "RECALL_BENCHD_UNKNOWN_SECRET" not in config
    assert "synthetic-password" not in text
    assert "synthetic-unknown-secret" not in text
    assert config["RECALL_BENCHD_TOP_K"] == "10"


def test_official_runner_accepts_relative_harness_path(official_runner):
    """BUG-002: baseline fails this success assertion after cd invalidates OUT."""
    result, harness = official_runner(relative=True)
    assert result.returncode == 0, result.stdout + result.stderr
    records = list(harness.glob("runs-official/*/run-record.json"))
    assert len(records) == 1
    assert (records[0].parent / "meter-after.json").is_file()
    assert (records[0].parent / "verify.log").is_file()


@pytest.mark.parametrize("failure", [True, False])
def test_worker_isolation_never_returns_shared_dsn(monkeypatch, failure):
    """FIX-007/STAKES-001: baseline returns the shared DSN on OperationalError.

    The failure=True node must fail the isolated-URL assertion on that baseline;
    failure=False is a successful-isolation control.
    """
    from tests import conftest
    import psycopg

    base = "postgresql://fixture:fixture@localhost/throwaway"
    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw_proof")
    monkeypatch.setenv("RECALL_TEST_DSN", base)

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

        def execute(self, *args):
            return SimpleNamespace(fetchone=lambda: (1,))

    def connect(*args, **kwargs):
        if failure:
            raise psycopg.OperationalError("synthetic transient isolation failure")
        return Connection()

    monkeypatch.setattr(psycopg, "connect", connect)
    isolated = conftest._isolate_xdist_worker(base)
    assert isolated.endswith("/throwaway_gwproof")
    assert os.environ["RECALL_TEST_DSN"] == isolated


def test_workers_use_distinct_unprivileged_roles():
    """DAT-001: baseline's two module imports select the same cluster-wide role.

    Real fixture constants are loaded under each worker's environment. No DB is
    configured, so these imports cannot create or alter a role or a database.
    """
    names = []
    for worker in ("gw0", "gw1"):
        env = dict(os.environ)
        env.pop("RECALL_TEST_DSN", None)
        env.pop("RECALL_DSN", None)
        env["PYTEST_XDIST_WORKER"] = worker
        env["PYTHONPATH"] = str(ROOT / "tests") + os.pathsep + str(ROOT)
        result = subprocess.run(
            [sys.executable, "-c", "import conftest; print(conftest.UNPRIVILEGED_ROLE)"],
            cwd=ROOT, env=env, text=True, capture_output=True, timeout=30,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        names.append(result.stdout.strip())
    assert len(set(names)) == 2
    assert all(name.startswith("recall_rls_probe_") for name in names)


def test_bmi_input_uses_harness_units_with_optional_tokenizer(monkeypatch, tmp_path, capsys):
    """NUM-001: baseline reports 99 BMI tokens for one word under this tokenizer.

    The real manifest entry point must report the harness estimator's one token;
    answer and judge cost estimates continue to use the optional tokenizer.
    """
    path = ROOT / "benchmarks/benchd/count_tokens.py"
    spec = importlib.util.spec_from_file_location("audit_token_counter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setitem(sys.modules, "tiktoken", SimpleNamespace(
        get_encoding=lambda name: SimpleNamespace(encode=lambda text: [0] * 99),
    ))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"traces": [{"raw_recall": "singleword"}]}))
    assert module.cmd_manifest(argparse.Namespace(
        path=str(manifest), price_in=0.15, price_out=0.60, project=None,
    )) == 0
    output = capsys.readouterr().out
    bmi = next(line for line in output.splitlines() if "BMI efficiency input" in line)
    assert "mean 1   median 1   p95 1" in bmi
    assert "counter: tiktoken o200k_base" in output
    assert "answerer  in        106   out       99" in output
