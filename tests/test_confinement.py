"""`recall.confinement`: the one containment check for paths under a root (M3).

Invariant: `confined` and `within` return the resolved path only when it lies under the resolved
root, decide on the link's TARGET, and refuse a UNC share (other than the root's) or a
drive-relative name before any filesystem call. The failure mode is the `access-control` shape CCA
kept finding: a containment check made after `resolve()` had already opened a UNC path, made on a
link's name, or skipped for `C:x`.

Red proof (2026-11-04), each mutation restored afterwards:

- `recall_mcp/indexing.py` restored to the pre-change `target = Path(path).resolve()` before the
  containment check: `test_index_memory_refuses_a_unc_path_without_resolving_it` failed in its
  `resolved == []` assertion, the UNC path recorded as resolved.
- `within` with its `remote_elsewhere` refusal deleted:
  `test_within_never_resolves_a_share_other_than_the_roots` and the consumer test above failed in
  the same assertion.
- `within` returning the path it was given instead of the resolved one:
  `test_within_returns_the_target_not_the_link` failed in its equality assertion.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from recall.confinement import anchored, confined, remote_elsewhere, within


@pytest.fixture
def resolved_calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every path `Path.resolve` is called on, so a test can see what reached the filesystem."""
    calls: list[str] = []
    real = Path.resolve

    def recording(self: Path, strict: bool = False) -> Path:
        calls.append(str(self))
        return real(self, strict=strict)

    monkeypatch.setattr(Path, "resolve", recording)
    return calls


@pytest.mark.parametrize(
    "text", ["/etc/passwd", "C:/x", "C:x", "\\\\host\\share\\x", "//host/share/x", "\\rooted"]
)
def test_anchored_names(text: str) -> None:
    assert anchored(text)


@pytest.mark.parametrize("text", ["notes/a.md", "a.md", "./a.md", "../a.md"])
def test_relative_names_are_not_anchored(text: str) -> None:
    assert not anchored(text)


def test_remote_elsewhere() -> None:
    assert remote_elsewhere("/srv/memos", "C:x")
    assert remote_elsewhere("/srv/memos", "//host/share/x")
    assert remote_elsewhere("//nas/memos/root", "//other/share/x")
    assert not remote_elsewhere("//nas/memos/root", "//NAS/memos/root/a.md")
    assert not remote_elsewhere("/srv/memos", "/srv/memos/a.md")


def test_confined_keeps_a_name_under_the_root(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "a.md").write_text("x", encoding="utf-8")
    assert confined(tmp_path, "sub/a.md") == (tmp_path / "sub" / "a.md").resolve()


@pytest.mark.parametrize("name", ["../outside.md", "sub/../../outside.md", "C:x", "//host/share/x", "/etc/x"])
def test_confined_refuses_a_name_that_leaves_the_root(tmp_path: Path, name: str) -> None:
    root = tmp_path / "root"
    root.mkdir()
    assert confined(root, name) is None


def test_within_never_resolves_a_share_other_than_the_roots(tmp_path: Path, resolved_calls: list[str]) -> None:
    assert within(tmp_path, "//attacker/share/x.md") is None
    assert [call for call in resolved_calls if "attacker" in call] == []


@pytest.mark.skipif(os.name == "nt", reason="creating a symlink needs a privilege on Windows")
def test_within_refuses_a_link_whose_target_leaves_the_root(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (tmp_path / "secret.md").write_text("x", encoding="utf-8")
    (root / "innocent.md").symlink_to(tmp_path / "secret.md")
    assert within(root, root / "innocent.md") is None


@pytest.mark.skipif(os.name == "nt", reason="creating a symlink needs a privilege on Windows")
def test_within_returns_the_target_not_the_link(tmp_path: Path) -> None:
    (tmp_path / "real.md").write_text("x", encoding="utf-8")
    (tmp_path / "link.md").symlink_to(tmp_path / "real.md")
    assert within(tmp_path, tmp_path / "link.md") == (tmp_path / "real.md").resolve()


def test_index_memory_refuses_a_unc_path_without_resolving_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, resolved_calls: list[str]
) -> None:
    """At the consumer: the MCP index tool refuses a UNC path before `resolve()` can open it."""
    from recall_mcp.service import index_memory

    monkeypatch.setenv("RECALL_INDEX_ROOT", str(tmp_path))
    with pytest.raises(ValueError, match="outside the directory this server is allowed to index"):
        index_memory(object(), object(), "//attacker/share/memory")  # type: ignore[arg-type]
    assert [call for call in resolved_calls if "attacker" in call] == []
