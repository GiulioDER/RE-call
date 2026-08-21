"""Generate the task-success fixtures and their held-out oracles, deterministically.

    python -m scripts.agent_ab_build_workspaces

The fixtures are committed, so this script is not run by the benchmark. It exists so that a reader
can see how each starting state was constructed, and regenerate it byte for byte, rather than
having to trust a directory of files that appeared once. Re-running it must be a no-op against a
clean tree; `tests/test_agent_ab_tasksuccess.py` asserts that.

Two rules the generator itself has to obey, both of them lessons this benchmark is measuring:

- **Every file is written with an explicit `newline="\\n"`.** One task is scored on whether the
  agent's edit preserved LF endings, and a generator that emitted CRLF would hand both arms a
  fixture that fails before anyone touches it.
- **Nothing is seeded from the clock.** The sampling corpus is generated from a fixed seed so the
  file-per-chunk distribution the checker relies on is a property of the committed fixture, not of
  the day it was built.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

WORKSPACES = REPO_ROOT / "benchmarks" / "agent_ab" / "workspaces"
ORACLES = REPO_ROOT / "benchmarks" / "agent_ab" / "oracles"

#: The planted files for ts-false-zero-search. Six of them, which is the number the checker
#: requires and the number the agent is never told.
PLANTED = {
    "scripts/sync_memory.sh": (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'REMOTE_STORE="/home/sentiment/recall-repos/memory"\n'
        'rsync -az "$REMOTE_STORE/" ./memory/\n'
    ),
    "scripts/deploy_worker.sh": (
        "#!/usr/bin/env bash\n"
        'PY="/home/sentiment/venv/bin/python"\n'
        'exec "$PY" -m worker.main "$@"\n'
    ),
    "recall/remote_config.py": (
        '"""Connection defaults for the indexing worker."""\n'
        "\n"
        'DEFAULT_CORPUS_ROOT = "/home/sentiment/recall-repos"\n'
        "DEFAULT_TIMEOUT_S = 30\n"
    ),
    "docs/OPERATIONS.md": (
        "# Operations\n"
        "\n"
        "The indexer runs on the worker host out of `/home/sentiment/recall-repos`, under the\n"
        "service account. Logs land beside it.\n"
    ),
    "tests/test_remote_paths.py": (
        "from recall.remote_config import DEFAULT_CORPUS_ROOT\n"
        "\n"
        "\n"
        "def test_default_corpus_root():\n"
        '    assert DEFAULT_CORPUS_ROOT == "/home/sentiment/recall-repos"\n'
    ),
    "benchmarks/remote_probe.py": (
        '"""Probe the worker host\'s corpus."""\n'
        "\n"
        "import subprocess\n"
        "\n"
        'CORPUS = "/home/sentiment/recall-repos/memory"\n'
        "\n"
        "\n"
        "def count_files() -> int:\n"
        '    out = subprocess.run(["ssh", "worker", f"ls {CORPUS} | wc -l"], capture_output=True)\n'
        "    return int(out.stdout or 0)\n"
    ),
}

#: Files with no such string. They exist so a search has something to reject, and so a checker
#: that accepted "every file" would fail.
DECOYS = {
    "README.md": "# worker\n\nIndexing worker for the corpus.\n",
    "recall/__init__.py": '__all__ = ["remote_config"]\n',
    "recall/index.py": (
        '"""Incremental indexing."""\n'
        "\n"
        "\n"
        "def index(paths):\n"
        "    return sorted(set(paths))\n"
    ),
    "recall/store.py": (
        '"""Chunk storage."""\n'
        "\n"
        "\n"
        "class Store:\n"
        "    def __init__(self):\n"
        "        self._rows = []\n"
        "\n"
        "    def add(self, row):\n"
        "        self._rows.append(row)\n"
    ),
    "tests/test_index.py": (
        "from recall.index import index\n"
        "\n"
        "\n"
        "def test_index_dedupes():\n"
        '    assert index(["b", "a", "b"]) == ["a", "b"]\n'
    ),
    "docs/DESIGN.md": "# Design\n\nChunks are stored per source.\n",
    "scripts/lint.sh": "#!/usr/bin/env bash\npython -m ruff check .\n",
    "notes/todo.md": "- tidy the operations doc\n- check the worker timeout\n",
}

NORMALISE_MODULE = (
    '"""Whitespace normalisation for indexed text."""\n'
    "\n"
    "import re\n"
    "\n"
    "#: Characters that separate lines and must collapse to a single space before indexing.\n"
    "SEPARATORS = (\n"
    '    "\\n",\n'
    '    "\\r",\n'
    "    chr(0x0B),\n"
    "    chr(0x0C),\n"
    "    chr(0x2028),\n"
    "    chr(0x2029),\n"
    ")\n"
    "\n"
    "\n"
    "def normalise(text: str) -> str:\n"
    '    """Collapse every run of separators and spaces into one space."""\n'
    "\n"
    "    out = text\n"
    "    for separator in SEPARATORS:\n"
    '        out = out.replace(separator, " ")\n'
    "    # Collapse runs of ASCII spaces only. A bare `out.split()` would split on every Unicode\n"
    "    # whitespace character, U+2028 included, which would make the SEPARATORS tuple above\n"
    "    # decorative: removing an entry from it would change nothing and no test could pin it.\n"
    '    return re.sub(r" +", " ", out).strip()\n'
)

#: The same module with U+2028 removed from the collapse set, and nothing else changed. Staged by
#: the checker AFTER the session, never present in the sandbox. A test whose separator was
#: flattened into an ordinary space keeps passing against this; a real one cannot.
NORMALISE_BROKEN = NORMALISE_MODULE.replace("    chr(0x2028),\n", "")

NORMALISE_TESTS = (
    "from recall.normalise import normalise\n"
    "\n"
    "\n"
    "def test_collapses_a_newline():\n"
    '    assert normalise("a\\nb") == "a b"\n'
    "\n"
    "\n"
    "def test_collapses_repeated_spaces():\n"
    '    assert normalise("a    b") == "a b"\n'
    "\n"
    "\n"
    "def test_strips_the_ends():\n"
    '    assert normalise("  a b  ") == "a b"\n'
)

PROBE_WITH_LINT_ERRORS = (
    '"""A probe that does not currently pass lint."""\n'
    "\n"
    "import json\n"
    "import os\n"
    "\n"
    "\n"
    "def load(path):\n"
    "    with open(path) as handle:\n"
    "        data = json.load(handle)\n"
    "    if data.get('mode') == None:\n"
    "        return {}\n"
    "    return data\n"
)

#: Deliberately unformatted, and deliberately lint-clean. `ruff check` has nothing to say about
#: these; `ruff format` would rewrite every one of them. That is the whole control: the tree tells
#: you which tool was run.
UNFORMATTED = {
    "benchmarks/pipeline.py": (
        "def build( rows ,limit = 10 ):\n"
        "    out=[]\n"
        "    for r in rows[:limit] :\n"
        "        out.append( r )\n"
        "    return out\n"
    ),
    "benchmarks/report.py": (
        "def render(rows):\n"
        "    return '\\n'.join( [ str(r)   for r in rows ] )\n"
    ),
    "recall/summary.py": (
        "def mean(values) :\n"
        "    if not values: return 0.0\n"
        "    return sum(values)/len(values)\n"
    ),
}


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def build_false_zero_search() -> None:
    root = WORKSPACES / "ts-false-zero-search" / "tree"
    for name, body in {**PLANTED, **DECOYS}.items():
        write(root / name, body)
    write(
        ORACLES / "ts-false-zero-search" / "expected.txt",
        "".join(f"{name}\n" for name in sorted(PLANTED)),
    )


def build_lf_rewrite() -> None:
    root = WORKSPACES / "ts-lf-rewrite" / "tree"
    write(root / ".gitattributes", "* text=auto eol=lf\n")
    write(
        root / "recall" / "version.py",
        '"""Single source of truth for the package version."""\n'
        "\n"
        "from __future__ import annotations\n"
        "\n"
        '__version__ = "0.9.7"\n'
        "\n"
        "#: Bumped by scripts/bump_version.py on every release. Keep the tuple in step: several\n"
        "#: callers compare it rather than parsing the string.\n"
        "VERSION_INFO = (0, 9, 7)\n"
        "\n"
        "\n"
        "def version_string() -> str:\n"
        '    """Return the version as it is printed by the CLI."""\n'
        "\n"
        "    return __version__\n"
        "\n"
        "\n"
        "def version_tuple() -> tuple[int, int, int]:\n"
        '    """Return the version as a comparable tuple."""\n'
        "\n"
        "    return VERSION_INFO\n"
        "\n"
        "\n"
        "def is_at_least(major: int, minor: int, patch: int = 0) -> bool:\n"
        '    """True when this build is at or beyond the given version."""\n'
        "\n"
        "    return VERSION_INFO >= (major, minor, patch)\n",
    )
    write(root / "README.md", "# recall\n\nVersion lives in `recall/version.py`.\n")
    write(root / "recall" / "__init__.py", "from .version import __version__\n")


def build_worktree_import() -> None:
    root = WORKSPACES / "ts-worktree-import" / "tree"
    write(root / "recall" / "__init__.py", '"""This checkout\'s package."""\n\nCHUNKS = 1006\n')
    write(root / "benchmarks" / "__init__.py", "")
    write(
        root / "benchmarks" / "probe.py",
        '"""Report the corpus chunk count."""\n'
        "\n"
        "import recall\n"
        "\n"
        "\n"
        "def main() -> int:\n"
        '    print(f"chunks={recall.CHUNKS}")\n'
        "    return 0\n"
        "\n"
        "\n"
        'if __name__ == "__main__":\n'
        "    raise SystemExit(main())\n",
    )
    write(
        root / "README.md",
        "# probe\n\n`benchmarks/probe.py` reports the chunk count out of the `recall` package.\n",
    )
    # The decoy stands in for the one editable install shared across every worktree on this
    # machine. It is an oracle, so it never enters the sandbox: an agent that could see a second
    # `recall` on disk would see the entire problem without knowing anything.
    write(
        ORACLES / "ts-worktree-import" / "installed" / "recall" / "__init__.py",
        '"""Another checkout\'s package, standing in for the shared editable install."""\n'
        "\n"
        "CHUNKS = 999999\n",
    )


def build_raise_on_missing() -> None:
    root = WORKSPACES / "ts-raise-on-missing" / "tree"
    rng = random.Random(20260821)
    rows = []
    for i in range(500):
        rows.append(
            {
                "task_id": f"case-{i:03d}",
                "variant": "recall_on" if i % 2 else "recall_off",
                "elapsed_ms": round(rng.uniform(120.0, 4800.0), 3),
                "tokens": rng.randint(400, 9000),
            }
        )
    write(root / "results" / "records.jsonl", "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    write(
        root / "README.md",
        "# results\n\n`results/records.jsonl` is one JSON object per benchmark session.\n",
    )

    # The oracle: the same 500 rows with the measured field ABSENT on 40 of them, which is what a
    # partial join failure produces. Never in the sandbox.
    poisoned = [dict(row) for row in rows]
    for index in rng.sample(range(500), 40):
        poisoned[index].pop("elapsed_ms")
    write(
        ORACLES / "ts-raise-on-missing" / "poisoned.jsonl",
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in poisoned),
    )
    mean = sum(r["elapsed_ms"] for r in rows) / len(rows)
    write(ORACLES / "ts-raise-on-missing" / "expected_mean.txt", f"{mean:.3f}\n")


def build_bounded_runner() -> None:
    root = WORKSPACES / "ts-bounded-runner" / "tree"
    # Python rather than a shell script, deliberately. A `bash` fixture would make this task
    # depend on resolving Git Bash, which is ts-launch-git-bash's fact, and two tasks sharing one
    # failure mode are not two measurements.
    write(
        root / "fixtures" / "slow.py",
        '"""Stands in for a command that hangs, and spawns a child that outlives it."""\n'
        "\n"
        "import subprocess\n"
        "import sys\n"
        "import time\n"
        "\n"
        "# The grandchild inherits this process's stdout pipe, which is the whole difficulty:\n"
        "# killing the direct child does not close it.\n"
        'subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])\n'
        "time.sleep(30)\n",
    )
    write(
        root / "README.md",
        "# bounded\n\n"
        "`fixtures/slow.py` stands in for a command that hangs. A pre-commit guard must not wait\n"
        "for it.\n",
    )


UPLOADS_MODULE = (
    '"""Writing an indexed upload, under a root the caller configures."""\n'
    "\n"
    "import os\n"
    "from pathlib import Path\n"
    "\n"
    "\n"
    "def index_root() -> Path:\n"
    '    """The configured root. It must already exist: creating it is the caller\'s job."""\n'
    "\n"
    '    root = Path(os.environ["RECALL_INDEX_ROOT"])\n'
    "    if not root.is_dir():\n"
    '        raise RuntimeError(f"RECALL_INDEX_ROOT does not exist: {root}")\n'
    "    return root\n"
    "\n"
    "\n"
    "def store_memo(job_id: str, text: str) -> Path:\n"
    '    """Write one memo for a job and return where it landed."""\n'
    "\n"
    '    target = index_root() / "uploads" / job_id / "memo.md"\n'
    "    target.parent.mkdir(parents=True, exist_ok=True)\n"
    '    target.write_text(text, encoding="utf-8", newline="\\n")\n'
    "    return target\n"
)

UPLOADS_TESTS = (
    "from recall.uploads import store_memo\n"
    "\n"
    "\n"
    "def test_store_memo_writes_the_text():\n"
    '    path = store_memo("job-1", "hello")\n'
    '    assert path.read_text(encoding="utf-8") == "hello"\n'
    "\n"
    "\n"
    "def test_store_memo_is_keyed_by_job():\n"
    '    a = store_memo("job-a", "a")\n'
    '    b = store_memo("job-b", "b")\n'
    "    assert a != b\n"
)

#: Held out. It asserts the property every test in a real suite implicitly assumes: that its own
#: `tmp_path` is its own. A fixture that mkdirs inside `tmp_path` fails it, and the failure names
#: the upload root rather than the fixture, which is the memo's point.
ISOLATION_TEST = (
    '"""Held out of the sandbox: does a test still own its own tmp_path?"""\n'
    "\n"
    "\n"
    "def test_tmp_path_belongs_to_the_test(tmp_path):\n"
    '    (tmp_path / "only.txt").write_text("x", encoding="utf-8")\n'
    "    assert [p.name for p in sorted(tmp_path.iterdir())] == [\"only.txt\"]\n"
    "\n"
    "\n"
    "def test_tmp_path_entries_are_all_files(tmp_path):\n"
    '    (tmp_path / "one.txt").write_text("1", encoding="utf-8")\n'
    "    for entry in sorted(tmp_path.iterdir()):\n"
    '        assert entry.is_file(), f"{entry.name} is not a file"\n'
)


def build_autouse_tmp_path() -> None:
    root = WORKSPACES / "ts-autouse-tmp-path" / "tree"
    write(root / "recall" / "__init__.py", "")
    write(root / "recall" / "uploads.py", UPLOADS_MODULE)
    write(root / "tests" / "test_uploads.py", UPLOADS_TESTS)
    write(
        root / "README.md",
        "# uploads\n\n"
        "`store_memo` writes under `RECALL_INDEX_ROOT`, which must already exist.\n",
    )
    write(ORACLES / "ts-autouse-tmp-path" / "test_scratch_isolation.py", ISOLATION_TEST)


def build_sample_covers_tail() -> None:
    root = WORKSPACES / "ts-sample-covers-tail" / "tree"
    rng = random.Random(19871787)
    files = [f"docs/{name}.md" for name in (
        "architecture", "calibration", "chunking", "cli", "contributing", "corpus", "design",
        "embeddings", "evaluation", "evidence", "faq", "frontmatter", "generations", "getting-started",
        "governance", "hnsw", "index", "ingestion", "install", "internals", "limits", "logging",
        "mcp", "memory", "migrations", "monitoring", "operations", "overview", "packaging",
        "performance", "permissions", "pipeline", "policy", "postgres", "privacy", "profiles",
        "querying", "ranking", "reasoning", "release", "reranking", "retrieval", "roadmap",
        "schema", "security", "sparse", "storage", "tenants", "testing", "trust", "validity",
    )]
    assert len(files) == 51, len(files)
    rows = []
    chunk = 0
    # 1815 chunks over 51 files, unevenly, the way a real corpus sits.
    per_file = [rng.randint(20, 51) for _ in files]
    scale = 1815 / sum(per_file)
    per_file = [max(1, round(count * scale)) for count in per_file]
    while sum(per_file) > 1815:
        per_file[per_file.index(max(per_file))] -= 1
    while sum(per_file) < 1815:
        per_file[per_file.index(min(per_file))] += 1
    for name, count in zip(files, per_file):
        for ordinal in range(count):
            words = rng.randint(45, 95)
            text = " ".join(rng.choice(LOREM) for _ in range(words))
            rows.append(
                {
                    "chunk_id": f"c{chunk:05d}",
                    "file": name,
                    "ordinal": ordinal,
                    "text": text,
                }
            )
            chunk += 1
    assert len(rows) == 1815, len(rows)
    write(root / "corpus.jsonl", "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    write(
        root / "README.md",
        "# corpus\n\n`corpus.jsonl` is one chunk per line, in corpus order.\n",
    )
    write(ORACLES / "ts-sample-covers-tail" / "file_count.txt", f"{len(files)}\n")


LOREM = (
    "retrieval corpus chunk tenant generation calibration threshold abstention evidence policy "
    "index fingerprint embedding profile reranker separability certified promotion snapshot "
    "vector store query dense sparse hybrid ingest manifest source object version digest"
).split()


def build_separator_canary() -> None:
    root = WORKSPACES / "ts-separator-canary" / "tree"
    write(root / "recall" / "__init__.py", "")
    write(root / "recall" / "normalise.py", NORMALISE_MODULE)
    write(root / "tests" / "test_normalise.py", NORMALISE_TESTS)
    write(ORACLES / "ts-separator-canary" / "normalise_broken.py", NORMALISE_BROKEN)


def build_lint_control() -> None:
    root = WORKSPACES / "ctl-lint-only-check" / "tree"
    write(root / "benchmarks" / "probe.py", PROBE_WITH_LINT_ERRORS)
    for name, body in UNFORMATTED.items():
        write(root / name, body)
    write(root / "recall" / "__init__.py", "")
    write(root / "benchmarks" / "__init__.py", "")
    write(
        root / "pyproject.toml",
        "[tool.ruff]\n"
        "line-length = 100\n"
        "target-version = \"py311\"\n"
        "\n"
        "[tool.ruff.lint]\n"
        "select = [\"E\", \"F\", \"W\"]\n",
    )
    write(root / "README.md", "# probe\n\nLint with this repository's convention.\n")


def build_pathspec_control() -> None:
    base = WORKSPACES / "ctl-stage-by-pathspec"
    write(base / "tree" / "recall" / "store.py", "def get(key):\n    return None\n")
    write(base / "tree" / "recall" / "index.py", "def index(paths):\n    return list(paths)\n")
    write(base / "tree" / "notes" / "scratch.md", "- rough notes\n")
    write(base / "tree" / "README.md", "# three files\n")
    # The overlay leaves all three modified, which is the state the task describes.
    write(
        base / "dirty" / "recall" / "store.py",
        "def get(key):\n    if not key:\n        raise KeyError(key)\n    return None\n",
    )
    write(
        base / "dirty" / "recall" / "index.py",
        "def index(paths):\n    return sorted(set(paths))\n",
    )
    write(base / "dirty" / "notes" / "scratch.md", "- rough notes\n- and more of them\n")


def main() -> int:
    build_false_zero_search()
    build_lf_rewrite()
    build_worktree_import()
    build_raise_on_missing()
    build_bounded_runner()
    build_sample_covers_tail()
    build_separator_canary()
    build_autouse_tmp_path()
    build_lint_control()
    build_pathspec_control()
    print(f"workspaces -> {WORKSPACES}")
    print(f"oracles    -> {ORACLES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
