from __future__ import annotations

from pathlib import Path

from scripts.check_requirements_hashes import audit_requirements


def test_dependency_audit_requires_a_hash_for_each_exact_pin(tmp_path: Path) -> None:
    lock = tmp_path / "requirements.lock.txt"
    lock.write_text(
        """pkg-one==1.2.3 \\
    --hash=sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
pkg-two==2.0.0
""",
        encoding="utf-8",
    )

    packages, errors = audit_requirements(lock)

    assert packages == 2
    assert errors == [f"{lock}:3: pkg-two has no sha256 hash"]


def test_dependency_audit_accepts_uv_export_hash_continuations(tmp_path: Path) -> None:
    lock = tmp_path / "requirements.lock.txt"
    lock.write_text(
        """# generated
pkg-one==1.2.3 ; python_full_version >= '3.11' \\
    --hash=sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \\
    --hash=sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
    # via project
""",
        encoding="utf-8",
    )

    packages, errors = audit_requirements(lock)

    assert packages == 1
    assert errors == []


def test_dependency_audit_rejects_an_unpinned_requirement(tmp_path: Path) -> None:
    lock = tmp_path / "requirements.lock.txt"
    lock.write_text("pkg-one>=1.2\n", encoding="utf-8")

    packages, errors = audit_requirements(lock)

    assert packages == 0
    assert errors == [
        f"{lock}:1: requirement is not an exact == pin: pkg-one>=1.2",
        f"{lock}: no exact pinned requirements found",
    ]
