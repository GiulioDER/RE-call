"""BUG-002 regression at the project-scoped hook consumer boundary."""

from pathlib import Path

from recall_hooks import write_time


def test_multiedit_draft_reaches_memory_search(monkeypatch):
    """Red on 76a0b36ea: edits-only input becomes empty and never searches."""
    root = str(Path.cwd())
    config = {"dsn": "postgresql://example/test", "project_root": root, "write_time": {
        "enabled": True, "connection_mode": "cold", "min_chars": 1,
    }}
    monkeypatch.setattr(write_time, "load_config", lambda: config)
    monkeypatch.setattr(write_time, "_in_cooldown", lambda: False)
    monkeypatch.setattr(write_time, "_clear_cooldown", lambda: None)
    searches = []
    monkeypatch.setattr(write_time, "search", lambda query, *_: searches.append(query) or [])
    event = {"tool_name": "MultiEdit", "cwd": root, "tool_input": {
        "file_path": "code.py", "edits": [
            {"old_string": "a", "new_string": "first draft"},
            None, {"new_string": 42},
            {"old_string": "b", "new_string": "second draft"},
        ],
    }}
    assert write_time.context_for(event) is None
    assert searches == ["first draft\nsecond draft"]
