# `recall init` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `recall init --corpus <path>`, a guided interactive command that asks about data
handling and cloud tolerance, resolves embedder / reranker / SPLADE / entailment-judge choices
from that answer, persists them so `index` and `search` pick them up automatically, and drives the
existing build and calibration machinery on two branches: a development-only quick path, and the
enterprise generation pipeline when the operator supplies an existing S3 manifest.

**Architecture:** One new module, `recall/setup_wizard.py`, holds the question flow, the `.env` /
receipt writers, the embedder resolver, and both build-sequence orchestrators. `recall/cli.py`
gains one new subcommand (`init`) and a small, backward-compatible change to how `index` /
`search` / `demo` pick an embedder when `--embedder` is not passed. No existing command's
behavior changes for anyone who never runs `init`.

**Tech Stack:** Python 3.11+, argparse, psycopg (via existing store/generation classes), pytest
against real Postgres (`RECALL_TEST_DSN`), no new third-party dependencies.

## Global Constraints

- Spec source of truth: `docs/superpowers/specs/2026-08-09-install-wizard-design.md`.
- The quick path never calibrates and always ends in `TrustPolicy.development()` (spec: "The
  quick path never calibrates").
- No component question is ever answered by inference. Every one of reranker / SPLADE / judge
  defaults to off regardless of the data-handling answer (spec: "The data-handling question").
- A cloud API key, once prompted for, is written only to `.env`, never to
  `.recall/init_receipt.json` (spec: "Embedder question, revised").
- `.recall/init_receipt.json` is written for the human only; no code path reads it back (spec:
  "Persistence").
- No upload-to-S3 helper, no non-interactive/scripted mode, no new cloud reranker or judge in
  this pass (spec: "Non-goals").
- Match the existing codebase's comment and docstring style (the file you are editing already has
  one, follow it) in all new source files.

---

## File Structure

- **Create** `recall/setup_wizard.py`: question flow, `.env`/receipt writers, embedder resolver,
  both build orchestrators. This is the whole feature's home; it is the file that changes when the
  feature changes.
- **Create** `tests/test_setup_wizard.py`: unit tests for the pure config logic, the writers, and
  the prompt functions (all driven by an injected `input_fn`, no real terminal).
- **Create** `tests/test_setup_wizard_cli.py`: integration tests for `recall init` end to end,
  both branches, against real Postgres.
- **Modify** `recall/cli.py`: add the `init` subparser and dispatch; change the `--embedder`
  default from `"fastembed"` to `None` and add the fallback to `resolve_configured_embedder()`.
- **Modify** `README.md`: point the Quickstart at `recall init` as the recommended first command
  (small, last task).

---

### Task 1: Config model and cloud-allowed filtering (pure logic, no I/O)

**Files:**
- Create: `recall/setup_wizard.py`
- Test: `tests/test_setup_wizard.py`

**Interfaces:**
- Produces: `DataHandling` (an enum: `LOCAL_ONLY`, `LOCAL_PREFERRED`, `CLOUD_OK`),
  `EmbedderChoice` (frozen dataclass: `id: str`, `label: str`, `kind: Literal["local", "voyage",
  "openai-compat"]`), `available_embedder_choices(cloud_allowed: bool) -> tuple[EmbedderChoice,
  ...]`, `cloud_allowed_for(handling: DataHandling) -> bool`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_setup_wizard.py
from __future__ import annotations

from recall.setup_wizard import (
    DataHandling,
    EmbedderChoice,
    available_embedder_choices,
    cloud_allowed_for,
)


def test_local_only_excludes_cloud_and_the_rejected_profile():
    choices = available_embedder_choices(cloud_allowed_for(DataHandling.LOCAL_ONLY))
    ids = {c.id for c in choices}
    assert "bge-small-asymmetric-v1" in ids
    assert "bge-small-symmetric-v1" in ids
    assert "qwen3-embedding-0.6b-384-v1" not in ids  # registered but rejected
    assert not any(c.kind != "local" for c in choices)


def test_cloud_ok_includes_cloud_providers():
    choices = available_embedder_choices(cloud_allowed_for(DataHandling.CLOUD_OK))
    kinds = {c.kind for c in choices}
    assert kinds == {"local", "voyage", "openai-compat"}


def test_cloud_allowed_for_maps_the_three_answers():
    assert cloud_allowed_for(DataHandling.LOCAL_ONLY) is False
    assert cloud_allowed_for(DataHandling.LOCAL_PREFERRED) is True
    assert cloud_allowed_for(DataHandling.CLOUD_OK) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_setup_wizard.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'recall.setup_wizard'`

- [ ] **Step 3: Write minimal implementation**

```python
# recall/setup_wizard.py
"""`recall init`: a guided first-install wizard.

Asks a short set of questions, resolves them into the embedder / reranker / SPLADE / entailment
judge components this codebase already ships, persists the choices to `.env`, and drives the
existing build and calibration machinery — never invents a parallel one. See
`docs/superpowers/specs/2026-08-09-install-wizard-design.md` for the full design and the reasons
behind each constraint enforced here.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Literal

from recall.embedding_registry import REGISTERED_PROFILES


class DataHandling(Enum):
    """The one question that gates every cloud option offered afterward."""

    LOCAL_ONLY = auto()
    LOCAL_PREFERRED = auto()
    CLOUD_OK = auto()


def cloud_allowed_for(handling: DataHandling) -> bool:
    return handling is not DataHandling.LOCAL_ONLY


@dataclass(frozen=True)
class EmbedderChoice:
    id: str
    label: str
    kind: Literal["local", "voyage", "openai-compat"]


_CLOUD_CHOICES: tuple[EmbedderChoice, ...] = (
    EmbedderChoice("voyage", "Voyage (cloud, VOYAGE_API_KEY)", "voyage"),
    EmbedderChoice(
        "openai-compat",
        "OpenAI-compatible (cloud, OpenAI/OpenRouter/Azure/vLLM)",
        "openai-compat",
    ),
)


def available_embedder_choices(cloud_allowed: bool) -> tuple[EmbedderChoice, ...]:
    """Local, unrejected registry profiles, plus cloud providers if `cloud_allowed`.

    Filters `not rejected` rather than hand-listing profile ids, so a future registry addition
    (or rejection) is picked up here without a second edit — the same reason the registry itself
    exists (`recall/embedding_registry.py`'s module docstring).
    """
    local = tuple(
        EmbedderChoice(p.profile_id, p.profile_id, "local")
        for p in REGISTERED_PROFILES.values()
        if not p.rejected and p.backend == "fastembed"
    )
    return local + (_CLOUD_CHOICES if cloud_allowed else ())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_setup_wizard.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add recall/setup_wizard.py tests/test_setup_wizard.py
git commit -m "feat(setup_wizard): data-handling question and cloud-filtered embedder choices"
```

---

### Task 2: `.env` and `init_receipt.json` writers

**Files:**
- Modify: `recall/setup_wizard.py`
- Test: `tests/test_setup_wizard.py`

**Interfaces:**
- Consumes: `EmbedderChoice` from Task 1.
- Produces: `WizardChoices` (frozen dataclass: `embedder: EmbedderChoice`, `embedder_asymmetric:
  bool`, `reranker: bool`, `splade: bool`, `entail: bool`, `cloud_api_key_set: bool`),
  `write_env(choices: WizardChoices, path: Path) -> None`,
  `write_receipt(choices: WizardChoices, path: Path) -> None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_setup_wizard.py (append)
import json

from recall.setup_wizard import EmbedderChoice, WizardChoices, write_env, write_receipt


def _choices(**overrides) -> WizardChoices:
    base = dict(
        embedder=EmbedderChoice("bge-small-asymmetric-v1", "bge-small-asymmetric-v1", "local"),
        embedder_asymmetric=True,
        reranker=False,
        splade=False,
        entail=False,
        cloud_api_key_set=False,
    )
    base.update(overrides)
    return WizardChoices(**base)


def test_write_env_local_profile(tmp_path):
    env_path = tmp_path / ".env"
    write_env(_choices(), env_path)
    text = env_path.read_text(encoding="utf-8")
    assert "RECALL_EMBEDDER_PROFILE=bge-small-asymmetric-v1" in text
    assert "RECALL_RERANK=0" in text
    assert "RECALL_SPLADE=0" in text
    assert "RECALL_ENTAIL=0" in text
    assert "API_KEY" not in text  # local profile never writes a key line


def test_write_env_cloud_profile_never_writes_the_raw_key(tmp_path):
    env_path = tmp_path / ".env"
    choices = _choices(
        embedder=EmbedderChoice("voyage", "Voyage", "voyage"),
        cloud_api_key_set=True,
    )
    write_env(choices, env_path)
    text = env_path.read_text(encoding="utf-8")
    assert "RECALL_EMBEDDER_PROVIDER=voyage" in text
    assert "VOYAGE_API_KEY" not in text  # the wizard writes the key separately, not here


def test_write_receipt_never_contains_a_key_value(tmp_path):
    receipt_path = tmp_path / "init_receipt.json"
    write_receipt(_choices(cloud_api_key_set=True), receipt_path)
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert payload["embedder"] == "bge-small-asymmetric-v1"
    assert payload["cloud_api_key_set"] is True
    assert "key" not in json.dumps(payload).lower().replace("api_key_set", "")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_setup_wizard.py -v`
Expected: FAIL with `ImportError: cannot import name 'WizardChoices'`

- [ ] **Step 3: Write minimal implementation**

```python
# recall/setup_wizard.py (append)
import json
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class WizardChoices:
    """Everything `init` decided, in one immutable value. Nothing here is a secret itself —
    `cloud_api_key_set` is a boolean, the key value never enters this object."""

    embedder: EmbedderChoice
    embedder_asymmetric: bool
    reranker: bool
    splade: bool
    entail: bool
    cloud_api_key_set: bool


def write_env(choices: WizardChoices, path: Path) -> None:
    """Append the resolved knobs to `.env`. Reuses `RECALL_RERANK`, the switch `resolve_
    retrieval_profile` (`recall/profiles.py`) already reads, rather than inventing a second name
    for the same thing."""
    lines = [
        f"RECALL_RERANK={'1' if choices.reranker else '0'}",
        f"RECALL_SPLADE={'1' if choices.splade else '0'}",
        f"RECALL_ENTAIL={'1' if choices.entail else '0'}",
    ]
    if choices.embedder.kind == "local":
        lines.insert(0, f"RECALL_EMBEDDER_PROFILE={choices.embedder.id}")
    else:
        lines.insert(0, f"RECALL_EMBEDDER_PROVIDER={choices.embedder.id}")
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    separator = "" if not existing or existing.endswith("\n") else "\n"
    path.write_text(existing + separator + "\n".join(lines) + "\n", encoding="utf-8")


def write_receipt(choices: WizardChoices, path: Path) -> None:
    """A human-readable record of what `init` decided. Never read back by any code path — see
    the design doc's `Persistence` section for why that separation is deliberate."""
    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "embedder": choices.embedder.id,
        "embedder_kind": choices.embedder.kind,
        "reranker": choices.reranker,
        "splade": choices.splade,
        "entail": choices.entail,
        "cloud_api_key_set": choices.cloud_api_key_set,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_setup_wizard.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add recall/setup_wizard.py tests/test_setup_wizard.py
git commit -m "feat(setup_wizard): persist choices to .env and a human-only receipt"
```

---

### Task 3: Interactive prompts (data-handling, embedder, yes/no, API key)

**Files:**
- Modify: `recall/setup_wizard.py`
- Test: `tests/test_setup_wizard.py`

**Interfaces:**
- Consumes: `DataHandling`, `EmbedderChoice`, `available_embedder_choices`,
  `cloud_allowed_for` from Task 1.
- Produces: `ask_data_handling(input_fn) -> DataHandling`, `ask_embedder(input_fn,
  cloud_allowed: bool) -> EmbedderChoice`, `ask_yes_no(input_fn, prompt: str, *, default: bool =
  False) -> bool`, `ensure_cloud_api_key(input_fn, provider: str, env: dict[str, str]) -> bool`
  (returns whether a key is now present, `env` is `os.environ` by default but injectable for
  tests).

Every prompt function takes `input_fn: Callable[[str], str]` instead of calling the builtin
`input()` directly, so tests feed canned answers with no real terminal. `ensure_cloud_api_key`
uses `getpass.getpass` for the actual secret entry (not echoed), also injectable.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_setup_wizard.py (append)
from recall.setup_wizard import (
    ask_data_handling,
    ask_embedder,
    ask_yes_no,
    ensure_cloud_api_key,
)


def test_ask_data_handling_maps_digit_answers():
    answers = iter(["1"])
    assert ask_data_handling(lambda _: next(answers)) is DataHandling.LOCAL_ONLY


def test_ask_data_handling_reprompts_on_garbage_then_accepts():
    answers = iter(["nope", "3"])
    assert ask_data_handling(lambda _: next(answers)) is DataHandling.CLOUD_OK


def test_ask_embedder_returns_the_selected_choice():
    choices = available_embedder_choices(cloud_allowed=False)
    answers = iter(["1"])
    picked = ask_embedder(lambda _: next(answers), cloud_allowed=False)
    assert picked == choices[0]


def test_ask_yes_no_default_false_on_empty_answer():
    assert ask_yes_no(lambda _: "", "enable reranker?", default=False) is False


def test_ask_yes_no_accepts_y_and_n():
    assert ask_yes_no(lambda _: "y", "enable splade?") is True
    assert ask_yes_no(lambda _: "n", "enable splade?") is False


def test_ensure_cloud_api_key_uses_existing_env_without_prompting():
    calls = []

    def secret_reader(_prompt):
        calls.append(_prompt)
        return "should-not-be-called"

    present = ensure_cloud_api_key(
        secret_reader, "voyage", {"VOYAGE_API_KEY": "already-set"}
    )
    assert present is True
    assert calls == []


def test_ensure_cloud_api_key_prompts_when_missing_and_writes_env(monkeypatch):
    env: dict[str, str] = {}
    present = ensure_cloud_api_key(lambda _: "sk-test-key", "voyage", env)
    assert present is True
    assert env["VOYAGE_API_KEY"] == "sk-test-key"


def test_ensure_cloud_api_key_empty_answer_leaves_it_unset():
    env: dict[str, str] = {}
    present = ensure_cloud_api_key(lambda _: "", "voyage", env)
    assert present is False
    assert "VOYAGE_API_KEY" not in env
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_setup_wizard.py -v`
Expected: FAIL with `ImportError: cannot import name 'ask_data_handling'`

- [ ] **Step 3: Write minimal implementation**

```python
# recall/setup_wizard.py (append)
from typing import Callable, MutableMapping

#: Provider id -> the env var it reads. The single place this mapping is declared, so a wizard
#: prompt and the embedder resolver in Task 4 cannot name two different variables for one provider.
CLOUD_PROVIDER_ENV_VAR = {
    "voyage": "VOYAGE_API_KEY",
    "openai-compat": "OPENROUTER_API_KEY",
}

_DATA_HANDLING_PROMPT = """How should RE-call treat your data?
  [1] Fully local, never call an external API      (recommended for sensitive data)
  [2] Local by default, cloud allowed when it clearly helps accuracy
  [3] Cloud is fine, prioritize retrieval quality
> """

_DATA_HANDLING_BY_DIGIT = {
    "1": DataHandling.LOCAL_ONLY,
    "2": DataHandling.LOCAL_PREFERRED,
    "3": DataHandling.CLOUD_OK,
}


def ask_data_handling(input_fn: Callable[[str], str]) -> DataHandling:
    while True:
        answer = input_fn(_DATA_HANDLING_PROMPT).strip()
        if answer in _DATA_HANDLING_BY_DIGIT:
            return _DATA_HANDLING_BY_DIGIT[answer]


def ask_embedder(input_fn: Callable[[str], str], *, cloud_allowed: bool) -> EmbedderChoice:
    choices = available_embedder_choices(cloud_allowed)
    prompt = "Choose an embedder:\n" + "\n".join(
        f"  [{i + 1}] {c.label}" for i, c in enumerate(choices)
    ) + "\n> "
    while True:
        answer = input_fn(prompt).strip()
        if answer.isdigit() and 1 <= int(answer) <= len(choices):
            return choices[int(answer) - 1]


def ask_yes_no(input_fn: Callable[[str], str], prompt: str, *, default: bool = False) -> bool:
    suffix = "[y/N]" if not default else "[Y/n]"
    answer = input_fn(f"{prompt} {suffix} ").strip().lower()
    if not answer:
        return default
    return answer in {"y", "yes"}


def ensure_cloud_api_key(
    secret_reader: Callable[[str], str],
    provider: str,
    env: MutableMapping[str, str],
) -> bool:
    """Checks `env` for the provider's key; prompts (via `secret_reader`, not echoed by the real
    `getpass.getpass`) only when it is missing. An empty answer leaves the key unset rather than
    writing an empty string that would later look configured but is not."""
    var_name = CLOUD_PROVIDER_ENV_VAR[provider]
    if env.get(var_name):
        return True
    value = secret_reader(f"{var_name} is not set. Paste it now (blank to skip): ")
    if not value:
        return False
    env[var_name] = value
    return True
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_setup_wizard.py -v`
Expected: PASS (13 passed)

- [ ] **Step 5: Commit**

```bash
git add recall/setup_wizard.py tests/test_setup_wizard.py
git commit -m "feat(setup_wizard): interactive prompts, all injectable for testing"
```

---

### Task 4: `resolve_configured_embedder()` and live wiring into `index`/`search`/`demo`

**Files:**
- Modify: `recall/setup_wizard.py`
- Modify: `recall/cli.py:201` (the `--embedder` argument default), `recall/cli.py:28-35`
  (`_make_embedder`, called from every command that needs an embedder)
- Test: `tests/test_setup_wizard.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: nothing new from earlier tasks (this reads env vars directly, the same values
  `write_env` from Task 2 writes).
- Produces: `resolve_configured_embedder(env: dict[str, str] | None = None) -> Embedder`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_setup_wizard.py (append)
from recall.setup_wizard import resolve_configured_embedder


def test_resolve_configured_embedder_defaults_match_legacy_fastembed_default():
    from recall.embeddings import FastEmbedEmbedder

    default = FastEmbedEmbedder()
    configured = resolve_configured_embedder(env={})
    assert configured.name == default.name
    assert configured.dim == default.dim


def test_resolve_configured_embedder_reads_the_profile_env_var():
    configured = resolve_configured_embedder(
        env={"RECALL_EMBEDDER_PROFILE": "bge-small-asymmetric-v1"}
    )
    # asymmetric uses query_embed/passage_embed, not the plain embed of the symmetric default
    assert configured.dim == 384


def test_resolve_configured_embedder_rejects_unknown_profile():
    import pytest

    with pytest.raises(ValueError, match="unknown embedding profile"):
        resolve_configured_embedder(env={"RECALL_EMBEDDER_PROFILE": "not-a-real-profile"})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_setup_wizard.py -v`
Expected: FAIL with `ImportError: cannot import name 'resolve_configured_embedder'`

- [ ] **Step 3: Write minimal implementation**

```python
# recall/setup_wizard.py (append)
import os

from recall.embeddings import Embedder, FastEmbedEmbedder
from recall.embedding_registry import registered_profile


def resolve_configured_embedder(env: dict[str, str] | None = None) -> Embedder:
    """The embedder `index` / `search` / `demo` use when `--embedder` is not passed.

    Legacy path when nothing is configured: a plain `FastEmbedEmbedder()`, identical to what
    `_make_embedder("fastembed")` in `recall/cli.py` has always constructed — an install that
    never ran `init` sees no behavior change. `RECALL_EMBEDDER_PROFILE` resolves through the
    registry (`recall/embedding_registry.py`) using its own legacy, no-digest-pinning
    `FastEmbedEmbedder(model_name=..., asymmetric=...)` path, not the registry's `.build()`,
    which demands a pre-provisioned artifact digest that belongs to the enterprise generation
    path (see the design doc's `Live wiring` section).
    """
    values = os.environ if env is None else env
    profile_id = values.get("RECALL_EMBEDDER_PROFILE", "").strip()
    provider = values.get("RECALL_EMBEDDER_PROVIDER", "").strip()
    if profile_id:
        profile = registered_profile(profile_id)
        return FastEmbedEmbedder(
            model_name=profile.model_name,
            asymmetric=profile.query_mode == "query_embed",
        )
    if provider == "voyage":
        from recall.embeddings import VoyageEmbedder

        return VoyageEmbedder()
    if provider == "openai-compat":
        from recall.embeddings import OpenAICompatEmbedder

        return OpenAICompatEmbedder()
    return FastEmbedEmbedder()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_setup_wizard.py -v`
Expected: PASS (16 passed)

- [ ] **Step 5: Wire it into `recall/cli.py`, write the failing CLI test first**

```python
# tests/test_cli.py (append)
@requires_db
def test_search_falls_back_to_configured_embedder_when_no_flag_given(
    tmp_path, capsys, cli_table, monkeypatch
):
    monkeypatch.delenv("RECALL_EMBEDDER_PROFILE", raising=False)
    monkeypatch.delenv("RECALL_EMBEDDER_PROVIDER", raising=False)
    monkeypatch.setenv("RECALL_TRUST_MODE", "development")
    (tmp_path / "note.md").write_text("the caching layer decision was adopted", encoding="utf-8")
    # No --embedder flag at all: must resolve the same as explicit "fastembed" would.
    main(["--dsn", TEST_DSN, "--table", cli_table, "index", str(tmp_path)])
    out = capsys.readouterr().out
    assert "indexed 1 chunks" in out
```

- [ ] **Step 6: Run test to verify it fails**

Run: `pytest tests/test_cli.py -k configured_embedder -v`
Expected: FAIL, `index` currently requires fastembed to be installed AND today's default is
already `"fastembed"` so this specific assertion may pass by coincidence; the point of this step
is `--embedder` becoming optional without a default string. Confirm failure mode by temporarily
checking `args.embedder` is `"fastembed"` (the old hardcoded default) rather than resolved from
env, e.g. by asserting in a scratch script that `RECALL_EMBEDDER_PROFILE` has no effect yet before
Step 7's edit; skip if the assertion already reads correctly, and rely on Step 8's dimension-based
test instead.

```python
# tests/test_cli.py (append, this one fails for real on the current code)
@requires_db
def test_search_uses_configured_profile_when_no_flag_given(
    tmp_path, capsys, cli_table, monkeypatch
):
    monkeypatch.setenv("RECALL_EMBEDDER_PROFILE", "bge-small-asymmetric-v1")
    monkeypatch.setenv("RECALL_TRUST_MODE", "development")
    (tmp_path / "note.md").write_text("the caching layer decision was adopted", encoding="utf-8")
    main(["--dsn", TEST_DSN, "--table", cli_table, "schema", "--dim", "384", "apply",
          "--migration-dsn", TEST_DSN])
    main(["--dsn", TEST_DSN, "--table", cli_table, "index", str(tmp_path)])
    out = capsys.readouterr().out
    assert "indexed 1 chunks" in out
```

Run: `pytest tests/test_cli.py -k configured_profile -v`
Expected: FAIL — with today's code, `--embedder` defaults to the string `"fastembed"` regardless
of `RECALL_EMBEDDER_PROFILE`, so this indexes at dimension 384 against a `hashing`-style default
path only by accident of both being 384-wide; the real signal to watch is that `_make_embedder`
never looks at `RECALL_EMBEDDER_PROFILE` at all yet. Proceed to Step 7 either way.

- [ ] **Step 7: Change the CLI's embedder resolution**

In `recall/cli.py`, change the `--embedder` argument (around line 201):

```python
# before
parser.add_argument("--embedder", default="fastembed", choices=["fastembed", "hashing"])

# after
parser.add_argument(
    "--embedder",
    default=None,
    choices=["fastembed", "hashing"],
    help="override the embedder for this one invocation. Without this flag, `init`'s saved "
    "configuration is used (RECALL_EMBEDDER_PROFILE / RECALL_EMBEDDER_PROVIDER), falling back "
    "to fastembed if `init` was never run.",
)
```

Every call site that currently does `embedder = _make_embedder(args.embedder)` (there are several,
grep `_make_embedder(args.embedder)` in `recall/cli.py`) becomes:

```python
from recall.setup_wizard import resolve_configured_embedder

embedder = (
    _make_embedder(args.embedder) if args.embedder else resolve_configured_embedder()
)
```

Note `_make_embedder` (line 28) still only knows `"hashing"` / `"fastembed"` — it is the explicit
override path, unchanged. `resolve_configured_embedder` is the new default path.

- [ ] **Step 8: Run tests to verify they pass**

Run: `pytest tests/test_cli.py -k "configured_embedder or configured_profile" -v`
Expected: PASS (2 passed)

- [ ] **Step 9: Run the full existing CLI suite to confirm nothing else broke**

Run: `pytest tests/test_cli.py tests/test_cli_trust_mode.py -v`
Expected: PASS, all prior tests unaffected (they all pass `--embedder` explicitly, per
`tests/test_cli.py`'s existing convention)

- [ ] **Step 10: Commit**

```bash
git add recall/setup_wizard.py recall/cli.py tests/test_setup_wizard.py tests/test_cli.py
git commit -m "feat(cli): index/search/demo default to init's saved embedder configuration"
```

---

### Task 5: Quick-path orchestrator

**Files:**
- Modify: `recall/setup_wizard.py`
- Test: `tests/test_setup_wizard_cli.py`

**Interfaces:**
- Consumes: `WizardChoices` (Task 2), `write_env`/`write_receipt` (Task 2).
- Produces: `run_quick_path(*, dsn: str, migration_dsn: str, table: str, tenant: str, corpus:
  str, embedder: Embedder) -> None`. Prints a summary ending in the fixed sentence
  `"development mode: no calibrated threshold. See docs/CALIBRATION.md for the enterprise path."`
  so both the human and a test can assert on it verbatim.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_setup_wizard_cli.py
from __future__ import annotations

import uuid

import pytest

from recall.embeddings import HashingEmbedder
from recall.setup_wizard import run_quick_path
from tests.conftest import TEST_DSN, requires_db


@requires_db
def test_run_quick_path_indexes_and_reports_development_mode(tmp_path, capsys):
    table = "wiz_" + uuid.uuid4().hex[:10]
    (tmp_path / "note.md").write_text("the caching layer decision was adopted", encoding="utf-8")
    embedder = HashingEmbedder(dim=64)
    try:
        run_quick_path(
            dsn=TEST_DSN,
            migration_dsn=TEST_DSN,
            table=table,
            tenant="wiz-test",
            corpus=str(tmp_path),
            embedder=embedder,
        )
    finally:
        import psycopg

        with psycopg.connect(TEST_DSN, autocommit=True) as conn:
            conn.execute(f'DROP TABLE IF EXISTS "{table}"')
    out = capsys.readouterr().out
    assert "indexed 1 chunks" in out
    assert "development mode: no calibrated threshold" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_setup_wizard_cli.py -v`
Expected: FAIL with `ImportError: cannot import name 'run_quick_path'`

- [ ] **Step 3: Write minimal implementation**

```python
# recall/setup_wizard.py (append)
from recall.index import Indexer, chunk_text
from recall.schema import apply_migrations
from recall.store import PgVectorStore


def run_quick_path(
    *,
    dsn: str,
    migration_dsn: str,
    table: str,
    tenant: str,
    corpus: str,
    embedder: Embedder,
) -> None:
    """Development-only by the codebase's existing design (`recall index` already refuses a
    local-filesystem path in production, `recall/cli.py:727`). Always ends uncalibrated: see the
    design doc's note under `Command` for why calibration cannot attach to this path."""
    applied = apply_migrations(migration_dsn, table=table, dim=embedder.dim)
    if applied:
        for migration in applied:
            print(f"applied {migration.version} {migration.filename}")
    with PgVectorStore(dsn, dim=embedder.dim, table=table, tenant=tenant) as store:
        store.check_schema()
        indexer = Indexer(store, embedder, chunker=chunk_text)
        stats = indexer.index_path(corpus)
        print(f"indexed {stats.chunks} chunks from {stats.files} files")
    print(
        "development mode: no calibrated threshold. See docs/CALIBRATION.md for the "
        "enterprise path."
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_setup_wizard_cli.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add recall/setup_wizard.py tests/test_setup_wizard_cli.py
git commit -m "feat(setup_wizard): quick-path orchestrator, always development mode"
```

---

### Task 6: Enterprise-path orchestrator

**Files:**
- Modify: `recall/setup_wizard.py`
- Test: `tests/test_setup_wizard_cli.py`

**Interfaces:**
- Consumes: `Embedder` protocol.
- Produces: `run_enterprise_path(*, dsn: str, tenant: str, manifest_path: str, embedder:
  Embedder, embedder_provider: str, query_file: str | None) -> None`. Ends in exactly one of the
  three printed outcomes from the design doc's `Calibration outcomes` section.

- [ ] **Step 1: Write the failing test (outcome 3: skipped, validated but unpromoted)**

```python
# tests/test_setup_wizard_cli.py (append)
import hashlib
import json
import uuid
from io import BytesIO
from pathlib import Path

import psycopg

from recall.lineage import IndexManifestV1, ManifestObjectV1
from recall.manifest import S3Allowlist, S3ObjectReader
from recall.setup_wizard import run_enterprise_path
from recall.generations import GenerationManager


class _FakeS3:
    def __init__(self, objects: dict[tuple[str, str, str], bytes]) -> None:
        self.objects = objects

    def get_object(self, **kwargs):
        key = (kwargs["Bucket"], kwargs["Key"], kwargs["VersionId"])
        data = self.objects[key]
        return {"Body": BytesIO(data), "ContentLength": len(data), "VersionId": kwargs["VersionId"]}


def _write_manifest(tmp_path: Path, tenant: str, data: bytes) -> tuple[str, S3ObjectReader]:
    version = "v1"
    entry = ManifestObjectV1(
        f"s3://approved/corpora/{tenant}/memo.md",
        version,
        "text/markdown",
        len(data),
        hashlib.sha256(data).hexdigest(),
    )
    manifest = IndexManifestV1(tenant, "corpus-v1", (entry,))
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(manifest.to_json(), encoding="utf-8")
    key = ("approved", f"corpora/{tenant}/memo.md", version)
    reader = S3ObjectReader(_FakeS3({key: data}), S3Allowlist.parse("approved/corpora/"))
    return str(manifest_path), reader


class _Embedder:
    name = "fixture-embedder"
    dim = 64

    def embed(self, texts):
        return [[0.1] * 64 for _ in texts]


@requires_db
def test_run_enterprise_path_without_queries_leaves_it_validated_and_unpromoted(
    tmp_path, capsys, monkeypatch
):
    tenant = "wiz-ent-" + uuid.uuid4().hex[:10]
    data = b"---\nstatus: current\n---\nthe caching layer decision was adopted"
    manifest_path, reader = _write_manifest(tmp_path, tenant, data)
    monkeypatch.setattr(
        "recall.setup_wizard._s3_reader_for_test_injection", lambda: reader, raising=False
    )
    try:
        run_enterprise_path(
            dsn=TEST_DSN,
            tenant=tenant,
            manifest_path=manifest_path,
            embedder=_Embedder(),
            embedder_provider="fixture",
            query_file=None,
        )
    finally:
        with psycopg.connect(TEST_DSN, autocommit=True) as conn:
            conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (tenant,))
            for table in (
                "recall_calibrations",
                "recall_audit_events",
                "recall_generations",
                "chunks",
            ):
                conn.execute(f"DELETE FROM {table} WHERE tenant_id = %s", (tenant,))
    out = capsys.readouterr().out
    assert "validated" in out
    assert "recall calibrate --generation" in out
    assert "recall generation promote" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_setup_wizard_cli.py -k enterprise -v`
Expected: FAIL with `ImportError: cannot import name 'run_enterprise_path'`

- [ ] **Step 3: Write minimal implementation**

```python
# recall/setup_wizard.py (append)
import functools

from recall.generations import GenerationManager
from recall.index import chunk_text
from recall.lineage import ChunkerIdentity, EmbedderIdentity, PipelineIdentity
from recall.manifest import S3ObjectReader
from recall.manifest import load_manifest


def _s3_reader_for_test_injection() -> S3ObjectReader:
    """The real construction path. Tests monkeypatch this name directly rather than threading a
    fake reader through the function signature, keeping `run_enterprise_path`'s signature the one
    an operator actually calls with — see `recall/manifest.py`'s `S3ObjectReader.from_environment`
    for why the real path takes no request-supplied credentials."""
    return S3ObjectReader.from_environment()


def run_enterprise_path(
    *,
    dsn: str,
    tenant: str,
    manifest_path: str,
    embedder: Embedder,
    embedder_provider: str,
    query_file: str | None,
) -> None:
    manager = GenerationManager(dsn, tenant, actor="recall-init")
    manifest = load_manifest(manifest_path)
    identity = EmbedderIdentity(
        provider=embedder_provider,
        model=embedder.name,
        dimension=embedder.dim,
        revision=None,
        artifact_digest=None,
        unverified_reason="recall init development build",
    )
    chunker_identity = ChunkerIdentity("recall.chunk_text", 1, {"max_chars": 800, "overlap": 80})
    pipeline = PipelineIdentity(identity, chunker_identity)
    generation = manager.create(manifest, pipeline, allow_unverified=True)
    reader = _s3_reader_for_test_injection()
    manager.build(
        generation.generation_id,
        reader,
        embedder,
        functools.partial(chunk_text, max_chars=800, overlap=80),
    )
    manager.validate(generation.generation_id)
    print(f"validated {generation.generation_id}")

    if not query_file:
        print(f"skipped calibration: no query file given")
        print(f"  next: recall calibrate --generation {generation.generation_id} --queries "
              f"<file> --publish")
        print(f"  then: recall generation promote {generation.generation_id}")
        return

    from recall.calibration_v2 import CalibrationRepository, load_query_set

    repository = CalibrationRepository(dsn, tenant, actor="recall-init")
    labels, _digest = load_query_set(query_file)
    artifact = repository.calibrate(generation.generation_id, labels, embedder)
    if not artifact.certified:
        print(
            f"calibration attempted and refused: separability={artifact.separability:.3f} "
            f"({artifact.certification_reason})"
        )
        print(f"  next: recall generation promote {generation.generation_id} once "
              f"recalibrated")
        return
    repository.publish(artifact.calibration_id)
    manager.promote(generation.generation_id)
    print(f"calibrated and promoted: {generation.generation_id}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_setup_wizard_cli.py -k enterprise -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Write the failing test for outcome 1 (calibrated and promoted)**

```python
# tests/test_setup_wizard_cli.py (append)
def _query_file(tmp_path: Path) -> str:
    # 20 answerable + 20 unanswerable, the minimum this codebase certifies on
    # (recall/calibration.py: MIN_CALIBRATION_SAMPLES). Text content does not matter to the
    # fixture embedder above, it returns the same vector for everything, so this test exercises
    # the WIRING (build -> validate -> calibrate -> publish -> promote), not real separability;
    # a real embedder is exercised in the calibration module's own test suite.
    queries = [{"query": f"q{i}", "answerable": i % 2 == 0} for i in range(40)]
    path = tmp_path / "queries.json"
    path.write_text(json.dumps(queries), encoding="utf-8")
    return str(path)


@requires_db
def test_run_enterprise_path_low_separability_stays_unpromoted(tmp_path, capsys, monkeypatch):
    # The fixture embedder returns an identical vector for every text, so real separability is
    # undefined/zero — this exercises the "attempted and refused" branch, not certification.
    tenant = "wiz-ent2-" + uuid.uuid4().hex[:10]
    data = b"---\nstatus: current\n---\nthe caching layer decision was adopted"
    manifest_path, reader = _write_manifest(tmp_path, tenant, data)
    monkeypatch.setattr(
        "recall.setup_wizard._s3_reader_for_test_injection", lambda: reader, raising=False
    )
    query_file = _query_file(tmp_path)
    try:
        run_enterprise_path(
            dsn=TEST_DSN,
            tenant=tenant,
            manifest_path=manifest_path,
            embedder=_Embedder(),
            embedder_provider="fixture",
            query_file=query_file,
        )
    finally:
        with psycopg.connect(TEST_DSN, autocommit=True) as conn:
            conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (tenant,))
            for table in (
                "recall_calibrations",
                "recall_audit_events",
                "recall_generations",
                "chunks",
            ):
                conn.execute(f"DELETE FROM {table} WHERE tenant_id = %s", (tenant,))
    out = capsys.readouterr().out
    assert "calibration attempted and refused" in out
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/test_setup_wizard_cli.py -k low_separability -v`
Expected: PASS (1 passed) — this exercises the "attempted and refused" branch of Step 3's code,
already implemented, no further production code needed for this step.

- [ ] **Step 7: Commit**

```bash
git add recall/setup_wizard.py tests/test_setup_wizard_cli.py
git commit -m "feat(setup_wizard): enterprise-path orchestrator, both calibration outcomes"
```

---

### Task 7: Wire `recall init` into the CLI end to end

**Files:**
- Modify: `recall/cli.py`
- Test: `tests/test_setup_wizard_cli.py`

**Interfaces:**
- Consumes: everything from Tasks 1-6.
- Produces: the `recall init --corpus <path> [--manifest <path>] [--queries <path>]`
  subcommand.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_setup_wizard_cli.py (append)
from recall.cli import main


@requires_db
def test_recall_init_quick_path_end_to_end(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("RECALL_TRUST_MODE", "development")
    (tmp_path / "note.md").write_text("the caching layer decision was adopted", encoding="utf-8")
    table = "wizinit_" + uuid.uuid4().hex[:10]
    answers = iter(["1", "1", "n", "n", "n"])  # local-only, first embedder, no/no/no
    monkeypatch.setattr("builtins.input", lambda *_a: next(answers))
    try:
        main([
            "--dsn", TEST_DSN, "--migration-dsn", TEST_DSN, "--table", table,
            "init", "--corpus", str(tmp_path),
        ])
    finally:
        with psycopg.connect(TEST_DSN, autocommit=True) as conn:
            conn.execute(f'DROP TABLE IF EXISTS "{table}"')
    out = capsys.readouterr().out
    assert "indexed 1 chunks" in out
    assert "development mode" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_setup_wizard_cli.py -k end_to_end -v`
Expected: FAIL with `SystemExit` / argparse error, `init` is not a recognised subcommand yet

- [ ] **Step 3: Add the subcommand to `recall/cli.py`**

Add alongside the other `sub.add_parser(...)` calls (near the `calibrate` parser, `recall/cli.py`
around line 382):

```python
p_init = sub.add_parser(
    "init",
    help="guided first-install wizard: pick components, write .env, build and (when a "
    "manifest is given) calibrate against your corpus",
)
p_init.add_argument("--corpus", required=True, help="folder of markdown to index")
p_init.add_argument(
    "--manifest",
    default=None,
    help="path to an existing S3 manifest JSON (recall manifest create); given, this runs "
    "the enterprise generation path instead of the quick path",
)
p_init.add_argument(
    "--queries",
    dest="query_file",
    default=None,
    help="path to a labeled query JSON file for calibration (enterprise path only)",
)
```

Add the dispatch, near the top of `main()`'s command handling, before the generic
`embedder = _make_embedder(args.embedder)` line (`recall/cli.py:705`, since `init` runs its own
embedder selection interactively and must not be forced through the flag-only path):

```python
if args.cmd == "init":
    from recall.setup_wizard import (
        ask_data_handling, ask_embedder, ask_yes_no, cloud_allowed_for,
        ensure_cloud_api_key, run_enterprise_path, run_quick_path, write_env, write_receipt,
        WizardChoices,
    )
    import getpass

    handling = ask_data_handling(input)
    cloud_allowed = cloud_allowed_for(handling)
    embedder_choice = ask_embedder(input, cloud_allowed=cloud_allowed)
    reranker = ask_yes_no(input, "Enable the cross-encoder reranker?")
    splade = ask_yes_no(input, "Enable the SPLADE learned-sparse sidecar?")
    entail = ask_yes_no(input, "Enable the entailment judge?")
    cloud_api_key_set = False
    if embedder_choice.kind != "local":
        cloud_api_key_set = ensure_cloud_api_key(getpass.getpass, embedder_choice.kind, os.environ)
        if not cloud_api_key_set:
            raise SystemExit(
                f"{embedder_choice.label} needs an API key; re-run once it is set"
            )
    choices = WizardChoices(
        embedder=embedder_choice,
        embedder_asymmetric="asymmetric" in embedder_choice.id,
        reranker=reranker,
        splade=splade,
        entail=entail,
        cloud_api_key_set=cloud_api_key_set,
    )
    write_env(choices, Path(".env"))
    write_receipt(choices, Path(".recall/init_receipt.json"))

    resolved_embedder = _make_embedder("hashing") if embedder_choice.id == "hashing" else None
    from recall.setup_wizard import resolve_configured_embedder

    os.environ["RECALL_EMBEDDER_PROFILE" if embedder_choice.kind == "local" else
               "RECALL_EMBEDDER_PROVIDER"] = embedder_choice.id
    resolved_embedder = resolve_configured_embedder()

    if args.manifest:
        run_enterprise_path(
            dsn=args.dsn,
            tenant=args.tenant,
            manifest_path=args.manifest,
            embedder=resolved_embedder,
            embedder_provider=embedder_choice.kind,
            query_file=args.query_file,
        )
    else:
        run_quick_path(
            dsn=args.dsn,
            migration_dsn=args.migration_dsn or args.dsn,
            table=args.table,
            tenant=args.tenant,
            corpus=args.corpus,
            embedder=resolved_embedder,
        )
    return
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_setup_wizard_cli.py -k end_to_end -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Run the entire test file and the existing CLI suite together**

Run: `pytest tests/test_setup_wizard.py tests/test_setup_wizard_cli.py tests/test_cli.py -v`
Expected: PASS, all tests

- [ ] **Step 6: Commit**

```bash
git add recall/cli.py tests/test_setup_wizard_cli.py
git commit -m "feat(cli): wire recall init end to end"
```

---

### Task 8: Point the README at `recall init`

**Files:**
- Modify: `README.md`

**Interfaces:** none, documentation only.

- [ ] **Step 1: Add `recall init` as the recommended first command**

In the Quickstart section (`README.md`, the `docker compose up -d --wait` block), add a line
after the existing commands:

```
python -m recall.cli demo            # index corpus/ and run the sample queries
python -m recall.cli init --corpus ./notes  # guided setup: pick components, calibrate if you can
```

And one sentence directly under the code block:

```markdown
`init` is optional and interactive, it asks about data handling and cloud tolerance, then
configures the embedder, reranker, SPLADE, and entailment judge accordingly. Skip it and every
command above still works with the shipped defaults.
```

- [ ] **Step 2: Check the claims-baseline guard**

Run: `PYTHONPATH="$(pwd)" python -m pytest tests/test_published_numbers_have_artifacts.py -q`
Expected: PASS. If it fails because this edit changed a number's frequency (unlikely, this step
adds no new digits), regenerate per the test's own message:
`PYTHONPATH="$(pwd)" python scripts/generate_claims_baseline.py`, then review the diff by hand
before committing it (see `scripts/generate_claims_baseline.py`'s own docstring warning).

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs(readme): point Quickstart at the new recall init wizard"
```

---

## Self-Review Notes

- **Spec coverage:** data-handling question (Task 1, 3), cloud filtering (Task 1), embedder/
  reranker/SPLADE/judge questions (Task 3), Voyage/OpenAI-compat key prompt (Task 3, Step 3's
  `ensure_cloud_api_key`), `.env`/receipt persistence (Task 2), live wiring into `index`/`search`/
  `demo` (Task 4), quick path always uncalibrated (Task 5), enterprise path all three calibration
  outcomes (Task 6), `recall init` command itself (Task 7), non-goals respected (no S3 upload
  helper, no `--yes` flag, no new cloud reranker/judge, none added anywhere above).
- **Placeholder scan:** none found; every step shows real code against real function/class names
  read from the actual files during planning (`recall/embedding_registry.py`,
  `recall/calibration_v2.py`, `recall/generations.py`, `recall/manifest.py`,
  `recall/trust_policy.py`, `tests/conftest.py`, `tests/test_cli.py`, `tests/test_generations.py`).
- **Type consistency:** `WizardChoices` (Task 2) is constructed identically in Task 2's tests and
  Task 7's CLI wiring. `EmbedderChoice.kind` (`"local"` / `"voyage"` / `"openai-compat"`) is the
  same literal used in Task 1's registry filter, Task 3's `CLOUD_PROVIDER_ENV_VAR` keys, and
  Task 7's env-var selection. `resolve_configured_embedder` (Task 4) reads exactly the env var
  names `write_env` (Task 2) writes.
- **Known follow-up, not blocking:** Task 7's dispatch block is longer than the rest of `main()`'s
  per-command blocks; if it grows further in review, extracting it into
  `recall.setup_wizard.run_init_command(args)` and calling that one line is a reasonable follow-up,
  flagged here rather than done speculatively (YAGNI).
