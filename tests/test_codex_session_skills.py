"""Session startup guidance uses live trust rather than a cached calibration claim.

Red proof: this exact test failed against 32b5e7caf's session_start at the
uncalibrated assertion. The installed count does not establish the live verdict.
The separate setup discovery invariant records its production-glob mutation.
"""
import json

import recall_hooks


def test_startup_digest_does_not_invent_live_trust(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("RECALL_HOOK_CONFIG_HOME", str(tmp_path))
    (tmp_path / "recall-hook.json").write_text(json.dumps({"chunks": 12}))
    assert recall_hooks.session_start({"cwd": "fixture"}) == 0
    context = json.loads(capsys.readouterr().out)["hookSpecificOutput"]["additionalContext"]
    assert "uncalibrated" not in context, "cached count invented a live calibration verdict"
    assert "recall_stats" in context
