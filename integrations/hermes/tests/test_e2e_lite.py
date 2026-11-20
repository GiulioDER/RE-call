"""One end-to-end run: the provider, a real `recall-mcp` child process, a real Lite store, fastembed.

Opt in with RECALL_HERMES_E2E=1, and name the RE-call install with RECALL_E2E_BIN (the directory
holding its `recall` and `recall-mcp`). It embeds locally with fastembed, so it is opt in. fastembed
downloads its model from HuggingFace on the first run; nothing calls a paid API. Everything lives in
pytest's temporary directory.
"""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import time
from pathlib import Path

import pytest
from conftest import wait_for

from re_call_hermes.config import PASSTHROUGH_ENV
from re_call_hermes.render import ABSTAIN_LINE, RETIRED_LEAD

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(os.environ.get("RECALL_HERMES_E2E") != "1", reason="set RECALL_HERMES_E2E=1 (runs fastembed)"),
]

TOPICS = [
    ("sourdough starter", "is fed with rye flour every Tuesday evening"),
    ("garden shed key", "hangs on the hook behind the pantry door"),
    ("dentist appointment", "is booked with Dr. Ferri at the Via Roma clinic"),
    ("bicycle", "has a seven speed hub and a dynamo front light"),
    ("passport", "expires in the spring of 2031"),
    ("violin lessons", "happen on Thursday afternoons with a teacher called Marta"),
    ("car insurance", "renews every March with a regional broker"),
    ("favourite tea", "is a smoky lapsang souchong from a shop in Bologna"),
    ("home router", "is a fibre box in the hallway cupboard"),
    ("running route", "follows the canal towpath for eight kilometres"),
    ("book club", "meets on the first Monday of each month at the library"),
    ("allergy", "is a mild reaction to penicillin"),
    ("piano", "was tuned last autumn by a technician from Modena"),
    ("houseplant", "is a fiddle leaf fig that needs water weekly"),
    ("chess rating", "is around 1650 on the online ladder"),
    ("winter coat", "is stored in the vacuum bag under the bed"),
    ("sister", "lives in Lisbon and works as an architect"),
    ("cat", "is a grey tabby called Pepe who hates the vacuum cleaner"),
    ("electricity provider", "bills quarterly by email"),
    ("hiking boots", "were resoled at the cobbler near the station"),
    ("camera", "is a mirrorless body with a 35 millimetre lens"),
    ("language course", "is intermediate Portuguese on Saturday mornings"),
    ("bank card", "was replaced after the chip stopped working"),
    ("coffee machine", "needs descaling every two months"),
    ("swimming pool", "opens at seven in the morning on weekdays"),
    ("laptop", "has a cracked hinge on the left side"),
    ("blood type", "is A positive"),
    ("balcony", "faces west and gets the evening sun"),
    ("guitar strings", "are light gauge nickel wound"),
    ("tax adviser", "is a firm on Corso Vittorio"),
    ("recipe for ragu", "simmers for four hours with milk added early"),
    ("phone plan", "includes twenty gigabytes of data a month"),
    ("parking spot", "is number fourteen in the underground garage"),
    ("favourite film", "is a 1960s Italian comedy about a heist"),
    ("smoke alarm", "chirps when its battery runs low"),
    ("gym membership", "was paused for the summer"),
    ("wedding anniversary", "falls on the ninth of June"),
    ("printer", "only accepts the XL black cartridges"),
    ("neighbour upstairs", "plays the cello late at night"),
    ("travel mug", "keeps coffee hot for about five hours"),
]


def _env(home: Path) -> dict[str, str]:
    env = {k: os.environ[k] for k in PASSTHROUGH_ENV if k in os.environ}
    env.update(PYTHONSAFEPATH="1", PYTHONIOENCODING="utf-8")
    return env


def _lite_home(tmp_path: Path) -> tuple[Path, Path]:
    """A Hermes home holding a real Lite store of TOPICS and the provider's config."""
    bin_dir = Path(os.environ["RECALL_E2E_BIN"])
    home = tmp_path / "hermes-home"
    data = home / "re-call"
    memos = data / "memos"
    memos.mkdir(parents=True)
    for i, (subject, fact) in enumerate(TOPICS):
        (memos / f"memo-{i:02d}.md").write_text(f"# My {subject}\n\nMy {subject} {fact}.\n", encoding="utf-8")
    store = data / "memory.db"
    t0 = time.monotonic()
    setup = subprocess.run(
        [str(bin_dir / "recall"), "--embedder", "fastembed", "setup", "--lite", "--memory", str(memos),
         "--store", str(store), "--no-register"],
        cwd=data, env=_env(home), capture_output=True, text=True, timeout=1200,
    )
    print("setup exit", setup.returncode, f"{time.monotonic() - t0:.1f}s")
    print(setup.stdout[-3000:], setup.stderr[-3000:])
    assert setup.returncode == 0

    (data / "config.json").write_text(json.dumps({
        "recall_command": str(bin_dir / "recall-mcp"),
        "dsn": "sqlite:///" + store.as_posix(),
        "memo_folder": str(memos),
    }), encoding="utf-8")
    return home, data


def test_supersession_and_abstention_through_a_real_lite_store(tmp_path, capsys):
    """Invariant: across the real process boundary, a corrected fact is current, the corrected one is
    never presented as current, and an unrelated question abstains.

    Red proof (2026-11-08, CCA run 20261108T171507Z, Linux host, Hermes v0.23.0 2bd378ed, `python
    tests/mutations.py E2E-SUPERSEDES-DROPPED`): in re_call_hermes/provider.py, `if v not in (None, "")}`
    became `if v not in (None, "") and k != "supersedes"}`; the test failed at `AssertionError: assert 'Acme
    Robotics' not in '## RE-call ...abs in Oslo.'`. Restored, green.
    """
    from re_call_hermes.provider import ReCallMemoryProvider

    home, data = _lite_home(tmp_path)
    provider = ReCallMemoryProvider()
    provider.initialize("e2e", hermes_home=str(home), platform="cli", agent_context="primary")
    try:
        t0 = time.monotonic()
        assert wait_for(provider._ready.is_set, timeout=600), provider._start_error
        print(f"server ready in {time.monotonic() - t0:.1f}s")

        first = json.loads(provider.handle_tool_call(
            "recall_remember", {"text": "My employer is Acme Robotics in Turin.", "valid_from": "2026-01-05"}))
        assert first.get("memo_id"), first
        second = json.loads(provider.handle_tool_call("recall_remember", {
            "text": "My employer is Borealis Labs in Oslo.", "valid_from": "2026-10-01",
            "supersedes": first["memo_id"]}))
        assert second.get("memo_id"), second

        timings = {}
        t0 = time.monotonic()
        raw = json.loads(provider.handle_tool_call("recall_search", {"query": "Who is my employer?"}))
        timings["tool_search_s"] = round(time.monotonic() - t0, 2)
        print("raw search:", json.dumps(raw)[:3000])

        t0 = time.monotonic()
        current = provider.prefetch("Who is my employer?")
        timings["prefetch_s"] = round(time.monotonic() - t0, 2)
        print("prefetch (employer):\n" + current)
        present = current.split(RETIRED_LEAD)[0]
        assert "Borealis Labs" in present
        assert "Acme Robotics" not in present
        assert provider.recall_status() is not None and provider.recall_status().count >= 1

        unrelated = provider.prefetch("What colour is the flag of the Zephyrine moon colony?")
        print("prefetch (unrelated):\n" + unrelated)
        assert ABSTAIN_LINE in unrelated
        assert provider.recall_status() is None
        print("timings", timings)
    finally:
        provider.shutdown()
        log = data / "server.log"
        if log.exists():
            print("server.log tail:\n" + log.read_text(encoding="utf-8", errors="replace")[-3000:])
        captured = capsys.readouterr()
        (Path(os.environ.get("RECALL_E2E_REPORT", str(tmp_path / "e2e-report.txt")))).write_text(
            captured.out + captured.err, encoding="utf-8")


def test_the_provider_serves_recall_mcps_own_six_tool_schemas(tmp_path):
    """Invariant: the six tools Hermes shows the model are, name for name, the description and input
    schema the running recall-mcp lists, so the provider cannot drift from RE-call's surface.

    Red proof (2026-11-09, Linux host, recall-rag 0.15.0, `python tests/mutations.py SURFACE-SCHEMA-DRIFT
    SURFACE-DESCRIPTION-DRIFT`): in re_call_hermes/surface.py, `"title": "Max Items"` became
    `"title": "Max items"`, and `Return the tenant scoped current projection` became `Return the
    current projection`; the test failed at `AssertionError: recall_evidence` and `AssertionError:
    recall_current_facts`. Restored, green. Its first run also failed for real: it compared the raw
    tools/list with the vendored (normalized) form, fixed by normalizing as RE-call does.
    """
    from re_call_hermes.provider import ReCallMemoryProvider

    home, _ = _lite_home(tmp_path)
    provider = ReCallMemoryProvider()
    provider.initialize("contract", hermes_home=str(home), platform="cli", agent_context="primary")
    try:
        assert provider._ready.is_set(), provider._start_error
        # As RE-call's own tests/test_assistant_surface.py vendors tools/list: the description
        # cleandoc'd, the input schema without its top-level title.
        served = {}
        for tool in provider._client._request("tools/list", {}, 60)["tools"]:
            schema = {k: v for k, v in tool["inputSchema"].items() if k != "title"}
            served[tool["name"]] = {"description": inspect.cleandoc(tool["description"]), "parameters": schema}
        offered = provider.get_tool_schemas()
        assert len(offered) == 6
        for tool in offered:
            assert tool["name"] in served, sorted(served)
            assert tool["description"] == served[tool["name"]]["description"], tool["name"]
            assert tool["parameters"] == served[tool["name"]]["parameters"], tool["name"]
    finally:
        provider.shutdown()
