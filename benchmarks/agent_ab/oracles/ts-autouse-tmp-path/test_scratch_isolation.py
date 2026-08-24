"""Held out of the sandbox: does a test still own its own tmp_path?"""


def test_tmp_path_belongs_to_the_test(tmp_path):
    (tmp_path / "only.txt").write_text("x", encoding="utf-8")
    assert [p.name for p in sorted(tmp_path.iterdir())] == ["only.txt"]


def test_tmp_path_entries_are_all_files(tmp_path):
    (tmp_path / "one.txt").write_text("1", encoding="utf-8")
    for entry in sorted(tmp_path.iterdir()):
        assert entry.is_file(), f"{entry.name} is not a file"
