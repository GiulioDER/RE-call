"""Two reference solutions per task: the one the fact prevents, and the one it produces.

⛔ **A checker nobody has watched fail has not been tested.** Every task here rests on a claim with
two halves: that a competent-looking answer written without the governing fact **fails** the
checker, and that an answer written with it **passes**. Neither half is self-evident, and getting
either wrong is silent. A checker that passes both is a task worth nothing that still contributes a
row; a checker that fails both makes the task look impossible in every arm and drags the whole
estimate toward zero.

So both halves are written down here and asserted in `tests/test_agent_ab_tasksuccess.py`. The
suite is the discrimination evidence, and it runs against the real fixtures and the real checkers
on the machine the benchmark runs on, which is the only place several of these facts are even true.

`naive` is not a straw man. Each one is the implementation a capable engineer writes on this task
without the memo: the obvious `.get` default, `subprocess.run(timeout=)`, `python benchmarks/x.py`,
`git grep` with the string it was given. That is the whole point, because the ones that announce
themselves were filtered out at design time. The one place `naive` is not literally what would be
typed is `ts-separator-canary`, where the hazard happens to the file on the way in rather than in
the author's head: there `naive` is the flattened result, which is what the agent would end up with.

Three candidate tasks died on exactly this test rather than on review, and two more died at
qualification; all five are listed in `tasksuccess.DROPPED_BEFORE_MEASUREMENT`. Two of the three had
premises that no longer reproduce on this machine, which is a thing you only discover by running the
naive answer and watching it pass.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

Writer = Callable[[Path], None]


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


# --------------------------------------------------------------------------- ts-false-zero-search

def _false_zero_naive(workdir: Path) -> None:
    _write(
        workdir / "scripts/audit_paths.sh",
        "#!/usr/bin/env bash\n"
        "set -uo pipefail\n"
        'git grep -l "/home/sentiment"\n',
    )


def _false_zero_informed(workdir: Path) -> None:
    _write(
        workdir / "scripts/audit_paths.sh",
        "#!/usr/bin/env bash\n"
        "set -uo pipefail\n"
        "# Bracket the leading slash so MSYS cannot rewrite the argument into a Windows path.\n"
        'git grep -l "[/]home/sentiment"\n',
    )


# --------------------------------------------------------------------------- ts-lf-rewrite

_BUMP = '''"""Bump the version. CI calls this on every release."""

import re
import sys
from pathlib import Path

TARGET = Path(__file__).resolve().parent.parent / "recall" / "version.py"


def main() -> int:
    new = sys.argv[1]
    major, minor, patch = (int(part) for part in new.split("."))
    text = TARGET.read_text(encoding="utf-8")
    text = re.sub(r'__version__ = "[^"]+"', f'__version__ = "{new}"', text)
    text = re.sub(
        r"VERSION_INFO = \\(\\d+, \\d+, \\d+\\)",
        f"VERSION_INFO = ({major}, {minor}, {patch})",
        text,
    )
    TARGET.write_text(text, encoding="utf-8"%s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _lf_naive(workdir: Path) -> None:
    _write(workdir / "scripts/bump_version.py", _BUMP % "")


def _lf_informed(workdir: Path) -> None:
    # The single argument that keeps this tree's eol=lf intact.
    _write(workdir / "scripts/bump_version.py", _BUMP % ', newline="\\n"')


# --------------------------------------------------------------------------- ts-worktree-import

def _worktree_naive(workdir: Path) -> None:
    _write(
        workdir / "scripts/run_probe.sh",
        "#!/usr/bin/env bash\n"
        "set -uo pipefail\n"
        "python benchmarks/probe.py\n",
    )


def _worktree_informed(workdir: Path) -> None:
    _write(
        workdir / "scripts/run_probe.sh",
        "#!/usr/bin/env bash\n"
        "set -uo pipefail\n"
        "# -m, not the script path: run as a script, sys.path[0] is benchmarks/ and the worktree\n"
        "# root is nowhere on the path, so `import recall` falls through to whatever is installed.\n"
        "python -m benchmarks.probe\n",
    )


# --------------------------------------------------------------------------- ts-raise-on-missing

_SUMMARISE = '''"""Mean elapsed_ms over a run's records."""

import json
from pathlib import Path

RECORDS = Path(__file__).resolve().parent.parent / "results" / "records.jsonl"


def main() -> int:
    values = []
    for line in RECORDS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        values.append(float(%s))
    print(f"mean_elapsed_ms={sum(values) / len(values):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _missing_naive(workdir: Path) -> None:
    _write(workdir / "scripts/summarise.py", _SUMMARISE % 'row.get("elapsed_ms", 0)')


def _missing_informed(workdir: Path) -> None:
    # Subscript, so an absent measurement raises instead of becoming a fabricated zero.
    _write(workdir / "scripts/summarise.py", _SUMMARISE % 'row["elapsed_ms"]')


# --------------------------------------------------------------------------- ts-bounded-runner

def _bounded_naive(workdir: Path) -> None:
    _write(
        workdir / "scripts/bounded.py",
        '"""Run a command with a 3 second bound."""\n'
        "\n"
        "import subprocess\n"
        "import sys\n"
        "\n"
        "\n"
        "def main() -> int:\n"
        "    try:\n"
        "        done = subprocess.run(sys.argv[1:], timeout=3)\n"
        "    except subprocess.TimeoutExpired:\n"
        '        print("TIMEOUT")\n'
        "        return 124\n"
        "    return done.returncode\n"
        "\n"
        "\n"
        'if __name__ == "__main__":\n'
        "    raise SystemExit(main())\n",
    )


def _bounded_informed(workdir: Path) -> None:
    _write(
        workdir / "scripts/bounded.py",
        '"""Run a command with a 3 second bound on WALL CLOCK, not on the direct child."""\n'
        "\n"
        "import subprocess\n"
        "import sys\n"
        "\n"
        "\n"
        "def main() -> int:\n"
        "    process = subprocess.Popen(sys.argv[1:])\n"
        "    try:\n"
        "        return process.wait(timeout=3)\n"
        "    except subprocess.TimeoutExpired:\n"
        "        # The tree, not the child: a grandchild holding the pipe outlives a plain kill.\n"
        "        subprocess.run(\n"
        '            ["taskkill", "/T", "/F", "/PID", str(process.pid)],\n'
        "            capture_output=True,\n"
        "            check=False,\n"
        "        )\n"
        "        process.wait(timeout=10)\n"
        '        print("TIMEOUT")\n'
        "        return 124\n"
        "\n"
        "\n"
        'if __name__ == "__main__":\n'
        "    raise SystemExit(main())\n",
    )


# --------------------------------------------------------------------------- ts-sample-covers-tail

_SAMPLE_HEAD = '''"""Sample chunks for review."""

import argparse
import json
import random
from pathlib import Path

CORPUS = Path(__file__).resolve().parent.parent / "corpus.jsonl"
BUDGET = 4000
POOL = 200


def load():
    return [
        json.loads(line)
        for line in CORPUS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
'''

_SAMPLE_TAIL = '''

if __name__ == "__main__":
    raise SystemExit(main())
'''


def _sample_naive(workdir: Path) -> None:
    _write(
        workdir / "scripts/sample.py",
        _SAMPLE_HEAD
        + '''

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rows = load()
    rng = random.Random(args.seed)
    picked = rng.sample(range(len(rows)), POOL)
    used = 0
    # Sorted for stable corpus order, and stopped at the budget.
    for index in sorted(picked):
        row = rows[index]
        line = f"{row['chunk_id']}\\t{row['file']}\\t{row['text']}"
        if used + len(line) + 1 > BUDGET:
            break
        print(line)
        used += len(line) + 1
    return 0
'''
        + _SAMPLE_TAIL,
    )


def _sample_informed(workdir: Path) -> None:
    _write(
        workdir / "scripts/sample.py",
        _SAMPLE_HEAD
        + '''

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rows = load()
    rng = random.Random(args.seed)
    picked = rng.sample(range(len(rows)), POOL)
    # Decide membership across the WHOLE sample first, in the sample's own random order, and only
    # then sort for display. Sorting before the budget check would mean only the lowest-indexed
    # part of the sample was ever considered.
    kept = []
    used = 0
    for index in picked:
        row = rows[index]
        line = f"{row['chunk_id']}\\t{row['file']}\\t{row['text']}"
        if used + len(line) + 1 > BUDGET:
            continue
        kept.append((index, line))
        used += len(line) + 1
    for _, line in sorted(kept):
        print(line)
    return 0
'''
        + _SAMPLE_TAIL,
    )


# --------------------------------------------------------------------------- ts-separator-canary

_SEPARATOR_TEST_HEAD = "from recall.normalise import normalise\n\n\n"


def _separator_naive(workdir: Path) -> None:
    # What the agent ends up with when the pasted U+2028 is flattened on the way into the file:
    # an ordinary space, and a test that passes against code with no U+2028 handling at all.
    path = workdir / "tests/test_normalise.py"
    path.write_text(
        path.read_text(encoding="utf-8")
        + "\n\ndef test_collapses_a_line_separator():\n"
        '    assert normalise("a b") == "a b"\n',
        encoding="utf-8",
        newline="\n",
    )


def _separator_informed(workdir: Path) -> None:
    path = workdir / "tests/test_normalise.py"
    path.write_text(
        path.read_text(encoding="utf-8")
        + "\n\ndef test_collapses_a_line_separator():\n"
        "    # chr(0x2028) rather than a pasted literal: the literal is silently normalised to an\n"
        "    # ordinary space by the editing path, and the test then passes against broken code.\n"
        "    separator = chr(0x2028)\n"
        '    assert separator not in " ", "premise: the separator is not already a space"\n'
        '    assert normalise(f"a{separator}b") == "a b"\n',
        encoding="utf-8",
        newline="\n",
    )


# --------------------------------------------------------------------------- ts-autouse-tmp-path

def _autouse_naive(workdir: Path) -> None:
    _write(
        workdir / "tests/conftest.py",
        "import pytest\n"
        "\n"
        "\n"
        "@pytest.fixture(autouse=True)\n"
        "def _index_root(tmp_path, monkeypatch):\n"
        '    root = tmp_path / "recall-index-root"\n'
        "    root.mkdir(parents=True, exist_ok=True)\n"
        '    monkeypatch.setenv("RECALL_INDEX_ROOT", str(root))\n'
        "    return root\n",
    )


def _autouse_informed(workdir: Path) -> None:
    _write(
        workdir / "tests/conftest.py",
        "import uuid\n"
        "\n"
        "import pytest\n"
        "\n"
        "\n"
        "@pytest.fixture(scope=\"session\")\n"
        "def _index_root_base(tmp_path_factory):\n"
        "    # Outside every test's own tmp_path, which belongs to the test: a directory created\n"
        "    # inside it breaks any test that enumerates its own scratch space.\n"
        '    return tmp_path_factory.mktemp("recall-index-root")\n'
        "\n"
        "\n"
        "@pytest.fixture(autouse=True)\n"
        "def _index_root(_index_root_base, monkeypatch):\n"
        "    root = _index_root_base / uuid.uuid4().hex\n"
        "    root.mkdir(parents=True, exist_ok=True)\n"
        '    monkeypatch.setenv("RECALL_INDEX_ROOT", str(root))\n'
        "    return root\n",
    )


# --------------------------------------------------------------------------- controls

def _lint_naive(workdir: Path) -> None:
    """Fix the lint error, and reformat the tree while you are there."""

    import subprocess
    import sys

    _fix_probe(workdir)
    subprocess.run(  # noqa: S603 - argv list, no shell
        [sys.executable, "-m", "ruff", "format", "."], cwd=str(workdir), capture_output=True
    )


def _lint_informed(workdir: Path) -> None:
    _fix_probe(workdir)


def _fix_probe(workdir: Path) -> None:
    _write(
        workdir / "benchmarks/probe.py",
        '"""A probe that does not currently pass lint."""\n'
        "\n"
        "import json\n"
        "\n"
        "\n"
        "def load(path):\n"
        "    with open(path) as handle:\n"
        "        data = json.load(handle)\n"
        '    if data.get("mode") is None:\n'
        "        return {}\n"
        "    return data\n",
    )


def _pathspec_naive(workdir: Path) -> None:
    from .checkers._run import git

    git("add", "-A", cwd=workdir)
    git("commit", "-q", "-m", "changes", cwd=workdir)


def _pathspec_informed(workdir: Path) -> None:
    from .checkers._run import git

    git("add", "--", "recall/store.py", "recall/index.py", cwd=workdir)
    git("commit", "-q", "-m", "Tighten store lookup and dedupe index paths", cwd=workdir)


REFERENCE: dict[str, dict[str, Writer]] = {
    "ts-false-zero-search": {"naive": _false_zero_naive, "informed": _false_zero_informed},
    "ts-lf-rewrite": {"naive": _lf_naive, "informed": _lf_informed},
    "ts-worktree-import": {"naive": _worktree_naive, "informed": _worktree_informed},
    "ts-raise-on-missing": {"naive": _missing_naive, "informed": _missing_informed},
    "ts-bounded-runner": {"naive": _bounded_naive, "informed": _bounded_informed},
    "ts-sample-covers-tail": {"naive": _sample_naive, "informed": _sample_informed},
    "ts-separator-canary": {"naive": _separator_naive, "informed": _separator_informed},
    "ts-autouse-tmp-path": {"naive": _autouse_naive, "informed": _autouse_informed},
    "ctl-lint-only-check": {"naive": _lint_naive, "informed": _lint_informed},
    "ctl-stage-by-pathspec": {"naive": _pathspec_naive, "informed": _pathspec_informed},
}
