from __future__ import annotations

from pathlib import Path

from scripts.build_beta_queue import main


def test_build_beta_queue_prints_top_links(tmp_path, capsys, monkeypatch) -> None:
    source = tmp_path / "discussions.csv"
    source.write_text(
        "\n".join(
            [
                "platform,community,url,title,body,author_handle,posted_at,replies,reactions,tags",
                (
                    "reddit,r/LocalLLaMA,https://example.com/reddit-1,"
                    "Agent memory keeps going stale,"
                    "Our agent hallucinates after long chats and loses provenance,"
                    "user1,2026-08-10T10:00:00Z,12,35,\"agent memory,provenance\""
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "queue.csv"
    monkeypatch.chdir(tmp_path)
    main(
        [
            str(source),
            "--term",
            "agent memory",
            "--term",
            "hallucinates",
            "--out",
            str(out),
            "--preview",
            "5",
        ]
    )
    captured = capsys.readouterr().out
    assert "top links:" in captured
    assert "https://example.com/reddit-1" in captured
    assert "[public_reply]" in captured
