"""The dashboard lists the tenants a user has, never one deployment's corpus names (ENV-004, ENV-013).

Rules, owner decision 2026-11-05:
- T1 with nothing pinned, every tenant with an active generation is listed by name, the default one
  first, and none is ever shown as "not built";
- T2 `--tenants` / `RECALL_DASHBOARD_TENANTS` pins the user's own main tenants, in their order, with
  labels; only a pinned tenant that is missing is shown as "not built";
- T3 the graph's corpus switch lists every corpus up to `MAX_SWITCH_CORPORA`; past it, the main ones
  and the one shown, and says how many more there are;
- T4 the overview's schema line names RE-call's own table and counts every other migrated table.

Red proof, 2026-11-05, each failing in its own assertion:
- against the pre-change `recall/dashboard/server.py` (hardcoded `PRIMARY_TENANTS` and benchmark
  prefixes): T1 "another deployment's corpus was listed", the short T3 "the user's own corpus was
  left out of the switch", T4 "another table was named, not counted";
- `_main_tenants` ignoring the pinned tenants: T2 "a pinned tenant was not listed first";
- the switch cap removed (`if False`): the long T3 "a benchmark tenant cluttered the corpus switch";
- `_parse_tenants` without its name check: DID NOT RAISE SystemExit, for all three bad lists;
- `pinned_tenants=` not passed by `_cmd_dashboard`: "the pinned tenants did not reach the app".
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pytest

from recall.cli_commands import dashboard_cmd
from recall.dashboard import db as dbq
from recall.dashboard import server
from recall.dashboard.db import DashboardDB
from recall.dashboard.server import DashboardApp

HOST = {"Host": "127.0.0.1:8765"}
SIGNED = {**HOST, "Cookie": "recall_dashboard=tok"}
THEIRS = ("memory", "re-call-code-gen", "re-call-docs")


@pytest.fixture
def root(tmp_path: Path) -> Path:
    (tmp_path / "a.md").write_text("# A\n\nnote\n", encoding="utf-8")
    return tmp_path


def _with_tenants(monkeypatch: pytest.MonkeyPatch, *names: str) -> None:
    monkeypatch.setattr(dbq, "tenants", lambda db: [{"tenant": n, "generation": "g", "chunks": 1} for n in names])


def _picker(root: Path, **kwargs: Any) -> str:
    body = DashboardApp(root, port=8765, token="tok", db=DashboardDB("postgresql://fake"), **kwargs).handle(
        "GET", "/memo?path=a.md", SIGNED
    ).body.decode()
    return body.split("class='tenantlist'>", 1)[1].split("</div>", 1)[0]


def test_t1_with_nothing_pinned_the_tenants_listed_are_the_users_own(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _with_tenants(monkeypatch, "work", "default", "notes")
    picker = _picker(root)
    assert not any(name in picker for name in THEIRS), "another deployment's corpus was listed"
    assert "not built" not in picker
    assert picker.index("tenant=default") < picker.index("tenant=notes") < picker.index("tenant=work")


def test_t2_pinned_tenants_come_first_with_their_labels(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _with_tenants(monkeypatch, "default", "notes", "work", "bench-1")
    picker = _picker(root, pinned_tenants=(("work", "my work"), ("notes", "notes"), ("archive", "old notes")))
    order = [picker.index(f"tenant={n}") for n in ("work", "notes", "default", "bench-1")]
    assert order == sorted(order), "a pinned tenant was not listed first"
    assert "my work" in picker
    assert "old notes <span class='muted'>not built</span>" in picker, "a pinned tenant that is missing was not said to be"
    assert picker.count("not built") == 1
    assert "other tenants · 1" in picker


def _switch(root: Path, corpus: str, **kwargs: Any) -> str:
    body = DashboardApp(root, port=8765, token="tok", db=DashboardDB("postgresql://fake"), **kwargs).handle(
        "GET", f"/graph?corpus={corpus}", SIGNED
    ).body.decode()
    return body.split("class='corpus-switch'", 1)[1].split("</div>", 1)[0]


def test_t3_the_corpus_switch_lists_every_corpus_while_the_list_is_short(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _with_tenants(monkeypatch, "default", "notes", "work")
    switch = _switch(root, "notes")
    assert all(f"corpus={n}" in switch for n in ("default", "notes", "work")), "the user's own corpus was left out of the switch"
    assert "more</a>" not in switch


def test_t3_past_the_cap_the_switch_keeps_the_main_ones_and_counts_the_rest(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bench = [f"bench-{i}" for i in range(server.MAX_SWITCH_CORPORA + 2)]
    _with_tenants(monkeypatch, "default", "work", *bench)
    switch = _switch(root, "bench-3", pinned_tenants=(("work", "work"),))
    assert all(f"corpus={n}" in switch for n in ("work", "default", "bench-3"))
    assert "corpus=bench-1" not in switch, "a benchmark tenant cluttered the corpus switch"
    assert f"{len(bench) - 1} more</a>" in switch


def test_t4_the_schema_line_names_recalls_table_and_counts_the_others(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dbq, "tenants", lambda db: [])
    monkeypatch.setattr(dbq, "availability", lambda db: {
        "server_version": "17.2", "role": "reader", "read_only": True,
        "migrations": [("chunks", "0035"), ("bench_x", "0012"), ("work_chunks", "0030")],
    })
    body = DashboardApp(root, port=8765, token="tok", db=DashboardDB("postgresql://fake")).handle("GET", "/overview", SIGNED).body.decode()
    line = body.split("schema: ", 1)[1].split("</p>", 1)[0]
    assert line == "chunks 0035, and 2 other tables", f"another table was named, not counted: {line}"


def _args(tmp_path: Path, tenants: str) -> argparse.Namespace:
    return argparse.Namespace(
        root=str(tmp_path), port=8765, reports_tenant=None, db_dsn_file=None, tunnel=None,
        tunnel_ports="55433:55432", config_host="", config_command="", tenants=tenants,
    )


class _Stop(Exception):
    pass


def test_the_command_hands_the_pinned_tenants_to_the_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    built: dict[str, Any] = {}

    def app(root: Path, **kwargs: Any) -> None:
        built.update(kwargs)
        raise _Stop

    monkeypatch.setattr(server, "DashboardApp", app)
    monkeypatch.setattr(dashboard_cmd, "_database", lambda path: None)
    with pytest.raises(_Stop):
        dashboard_cmd._cmd_dashboard(_args(tmp_path, " memory , re-call-code-gen=code "))
    assert built.get("pinned_tenants") == (("memory", "memory"), ("re-call-code-gen", "code")), "the pinned tenants did not reach the app"


@pytest.mark.parametrize("spec", ["<b>x</b>", "a,a", "=label"])
def test_a_bad_tenant_list_stops_the_command(tmp_path: Path, spec: str) -> None:
    with pytest.raises(SystemExit) as stopped:
        dashboard_cmd._parse_tenants(spec)
    assert stopped.value.code == 2, "a bad tenant name was accepted"


def test_the_variable_is_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RECALL_DASHBOARD_TENANTS", "work=my work")
    parser = argparse.ArgumentParser()
    dashboard_cmd.register(parser.add_subparsers())
    assert parser.parse_args(["dashboard"]).tenants == "work=my work"
