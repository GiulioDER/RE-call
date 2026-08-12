# Truth Extraction and Reviewed Rewrites Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate three unmerged branches into one landed feature that extracts structured truth claims from memo prose with a model and writes reviewed ones back into corpus files through a named human gate.

**Architecture:** Two `git merge` operations bring in a write path (`a7834c`) and an extractor (`ardinghelli`) that touch disjoint files. New work joins them: a relation to key resolver, the end to end chain through `promotion.py`, a verb subparser CLI, and a read only MCP surface. A third branch (`1cfc81`) contributes one ported module and is otherwise abandoned.

**Tech Stack:** Python 3.11 to 3.14, argparse, SQLite (rejection sidecar), PostgreSQL 16/17/18 (existing store only), MCP SDK 2.x, pytest, ruff 0.16.x, mypy.

**Design doc:** `docs/TRUTH_EXTRACTION_DESIGN.md`

## Global Constraints

- Worktree is `C:\Users\gde00\Documents\recall\.claude\worktrees\truth-extraction-prose-7b7e90`, branch `claude/truth-extraction-prose-7b7e90`. Never commit from another worktree.
- Database is the dedicated container only: `RECALL_TEST_DSN=postgresql://recall:recall@127.0.0.1:5434/recall`. Leave `RECALL_DSN` unset. Never use port 5432.
- `python -m ruff check .` and `mypy` must be clean. Bare `ruff` on this machine is a stale 0.6.9. **Never** run `ruff format`: 348 of 406 files fail it and CI does not run it.
- Do not inject CRLF. Use the Edit tool, or `newline="\n"` when scripting a write.
- Tests live flat in `tests/test_<area>_<subject>.py`. DB tests carry `@requires_db` and reach the DB through `make_store` / `cli_table`.
- A boundary someone could violate gets a `tests/test_*_contract.py` whose docstring enumerates the properties, one test per property, written so a plausible wrong implementation fails.
- Optional extras absent by default. Precedent: `recall/entailment.py:75`.
- Every guard is mutated and watched go red before it is claimed to work.
- Full suite takes ~12 minutes with the DB up. A run reporting ~516 SKIPPED means the DB was down; check the skip count before calling it green.
- No dashes as sentence punctuation in prose or docstrings.

---

### Task 1: Merge the write path

**Files:**
- Modify (by merge): `recall/rewrite.py`, `recall/atomic_write.py`, `recall/frontmatter.py`, `recall/fix.py`, `recall/cli.py`
- Test (by merge): `tests/test_corpus_rewrite_contract.py`, `tests/test_fix.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `recall.rewrite.plan_rewrite(root: Path, fact: PromotedFact) -> RewritePlan`, `recall.rewrite.apply_rewrite(...) -> RewriteResult`, `recall.rewrite.destination(key: str) -> Literal["frontmatter","derived"]`, `recall.rewrite.claim_key(relation, subject_id, object_id) -> str`, `recall.rewrite.RewriteRefused`, `recall.rewrite.RejectionLedger`, `recall.rewrite.default_ledger_path(root) -> Path`, `recall.rewrite.FRONTMATTER_KEYS`, `recall.rewrite.DERIVED_KEYS`, `recall.atomic_write.atomic_write_bytes(path: Path, data: bytes) -> None`, and the `recall.frontmatter` byte helpers `split_bom`, `split_lines`, `dominant_newline`, `line_terminator`, `is_fence`, `insert_frontmatter_line`.

- [ ] **Step 1: Confirm the merge is clean before starting**

```bash
git merge-tree --write-tree claude/truth-extraction-prose-a7834c claude/suspicious-ardinghelli-3d3e31
```

Expected: exit 0 and a single tree hash on stdout, with no `CONFLICT` lines. If anything else appears, stop and re-plan.

- [ ] **Step 2: Merge the branch**

```bash
git merge --no-ff claude/truth-extraction-prose-a7834c -m "Merge the reviewed rewrite write path"
```

Expected: `Merge made by the 'ort' strategy.` and no conflict markers.

- [ ] **Step 3: Run the merged branch's own contract test**

```bash
RECALL_TEST_DSN=postgresql://recall:recall@127.0.0.1:5434/recall python -m pytest tests/test_corpus_rewrite_contract.py tests/test_fix.py -q
```

Expected: all pass, 0 failed. Record the exact counts in the commit message. If anything fails, the failure belongs to the merge and must be fixed before Task 2.

- [ ] **Step 4: Lint and typecheck**

```bash
python -m ruff check . && mypy recall recall_mcp
```

Expected: `All checks passed!` and `Success: no issues found`.

- [ ] **Step 5: Commit is already made by the merge; verify it**

```bash
git log --oneline -1 && git status --short
```

Expected: the merge commit, and a clean working tree.

---

### Task 2: Merge the extractor

**Files:**
- Create (by merge): `recall/truth_extraction/{__init__,types,_engine,_prompt,_normalize,_cache,extract}.py`, `recall/reasoning_proposals/_extracted.py`
- Modify (by merge): `recall/reasoning_proposals/types.py`, `recall/reasoning_proposals/_providers.py`, `docs/INFERENCE_PROPOSALS.md`, `results/reasoning_session3_proposals.json`
- Test (by merge): `tests/test_truth_extraction_contract.py`, `tests/fakes.py`, `tests/test_reasoning_proposals.py`

**Interfaces:**
- Consumes: nothing from Task 1 (disjoint file sets).
- Produces: `recall.truth_extraction.extract.extract_file_claims(*, file, text, corpus_names, engine, cache=None) -> FileExtraction`, `extract_corpus_claims(documents, *, engine, corpus_names=None, cache=None) -> tuple[FileExtraction, ...]`, `resolve_extraction_engine(env=None) -> ExtractionEngine | None`, the `ExtractionEngine` Protocol with `engine_id`/`model_id`/`revision`/`run(prompt) -> str`, the claim dataclasses `SupersessionClaim`/`ValidityClaim`/`StatusClaim`/`IdentityClaim`, `VALIDITY_CLAIM_KEYS: tuple[Literal["valid_from","valid_until"], ...]`, `MAX_CLAIMS_PER_FILE = 12`, and `recall.reasoning_proposals._extracted.ExtractedClaimProposalProvider`. Also `recall.reasoning_proposals.types.PROPOSED_RELATIONS` and `PROPOSAL_SCHEMA_VERSION = 2`.

- [ ] **Step 1: Merge the branch**

```bash
git merge --no-ff claude/suspicious-ardinghelli-3d3e31 -m "Merge model backed truth extraction with its refusing ladder"
```

Expected: `Merge made by the 'ort' strategy.` and no conflict markers.

- [ ] **Step 2: Verify the proposal id churn landed with the schema bump**

The bump from `PROPOSAL_SCHEMA_VERSION = 1` to `2` rewrites every `ip_` id. The checked in artifact must already reflect it.

```bash
git diff HEAD~1 --stat -- results/reasoning_session3_proposals.json
```

Expected: the file shows as changed by the merge. If it does not, the artifact is stale against the new schema and every id in it is wrong; regenerate it before continuing.

- [ ] **Step 3: Run both merged test suites together**

```bash
RECALL_TEST_DSN=postgresql://recall:recall@127.0.0.1:5434/recall python -m pytest tests/test_truth_extraction_contract.py tests/test_reasoning_proposals.py tests/test_corpus_rewrite_contract.py -q
```

Expected: all pass. This is the first time these two branches' tests have run in one process.

- [ ] **Step 4: Run the full suite over the merge**

```bash
RECALL_TEST_DSN=postgresql://recall:recall@127.0.0.1:5434/recall python -m pytest -q
```

Expected: green, ~12 minutes. **Check the skip count.** If roughly 516 tests skipped, the database was down and this run is not evidence. Bring it up and re-run.

- [ ] **Step 5: Lint and typecheck**

```bash
python -m ruff check . && mypy recall recall_mcp
```

Expected: clean.

---

### Task 3: The relation to key resolver

The one seam the two merged branches do not share. `destination()` routes on a bare key; `_extracted.py` emits a relation with the key prefixed into `object_id` (`"valid_from:2026-07-14"`, `"status:deprecated"`).

**This is a live bug after Task 2, not just a missing feature.** `plan_rewrite` (`recall/rewrite.py:388-392`) already routes inline, and it calls `destination(checked.relation)` passing a **relation** where a key is expected. That works under schema v1 only by coincidence: the three v1 relation names happened to equal their key names. Schema v2 adds `declares_validity` and `declares_status`, and `destination("declares_validity")` raises `RewriteRefused: unknown_key`. It also sets `RewritePlan.key = checked.relation`, which for a validity claim is not a key at all.

So `route_relation` **replaces** that inline block. Adding it beside `plan_rewrite` and leaving the old lines in place would leave the bug in the shipping path and the new function dead.

**Files:**
- Modify: `recall/rewrite.py` (add below `destination`)
- Test: `tests/test_rewrite_routing_contract.py`

**Interfaces:**
- Consumes: `recall.rewrite.RewriteRefused`, `recall.rewrite.destination` (Task 1); `recall.truth_extraction.types.VALIDITY_CLAIM_KEYS` and `recall.reasoning_proposals.types.PROPOSED_RELATIONS` (Task 2).
- Produces: `recall.rewrite.Routed` (frozen dataclass with `key: str`, `value: str`, `edit_file: str`) and `recall.rewrite.route_relation(relation: str, subject_id: str, object_id: str) -> Routed`.

- [ ] **Step 1: Write the failing contract test**

Create `tests/test_rewrite_routing_contract.py`:

```python
"""Properties of the relation to key resolver.

1. `supersedes` is the ONLY relation whose edit lands on `object_id`.
2. Every other routable relation edits `subject_id`.
3. `declares_validity` takes its key from the `object_id` prefix.
4. A validity prefix that is not one of the two validity keys is refused, not coerced.
5. `supersedes` cannot be smuggled in through a validity prefix.
6. `declares_status` routes to the derived block, never to frontmatter.
7. `references` is refused: it has no key.
8. Every relation in `PROPOSED_RELATIONS` is either routed or explicitly refused, so adding
   a relation to the vocabulary without routing it fails this file.
"""
import pytest

from recall.reasoning_proposals.types import PROPOSED_RELATIONS
from recall.rewrite import RewriteRefused, destination, route_relation


def test_supersedes_edits_the_object_and_names_the_subject():
    routed = route_relation("supersedes", "old.md", "new.md")
    assert routed.edit_file == "new.md"
    assert routed.key == "supersedes"
    assert routed.value == "old.md"


def test_every_other_routable_relation_edits_the_subject():
    for relation, object_id in (
        ("declares_validity", "valid_from:2026-07-14"),
        ("declares_status", "status:deprecated"),
        ("contradicts", "other.md"),
        ("same_entity", "Alias"),
    ):
        routed = route_relation(relation, "subject.md", object_id)
        assert routed.edit_file == "subject.md", relation


def test_declares_validity_takes_its_key_from_the_prefix():
    routed = route_relation("declares_validity", "memo.md", "valid_until:2026-12-31")
    assert routed.key == "valid_until"
    assert routed.value == "2026-12-31"
    assert destination(routed.key) == "frontmatter"


def test_an_unknown_validity_prefix_is_refused_not_coerced():
    with pytest.raises(RewriteRefused, match="unroutable_validity"):
        route_relation("declares_validity", "memo.md", "expires_on:2026-12-31")


def test_supersedes_cannot_be_smuggled_through_a_validity_prefix():
    with pytest.raises(RewriteRefused, match="unroutable_validity"):
        route_relation("declares_validity", "memo.md", "supersedes:victim.md")


def test_declares_status_routes_to_the_derived_block():
    routed = route_relation("declares_status", "memo.md", "status:deprecated")
    assert routed.key == "status"
    assert routed.value == "deprecated"
    assert destination(routed.key) == "derived"


def test_references_is_refused_because_it_has_no_key():
    with pytest.raises(RewriteRefused, match="unroutable_relation"):
        route_relation("references", "a.md", "b.md")


def test_every_relation_in_the_vocabulary_is_routed_or_refused():
    for relation in PROPOSED_RELATIONS:
        object_id = {
            "declares_validity": "valid_from:2026-07-14",
            "declares_status": "status:active",
        }.get(relation, "other.md")
        try:
            routed = route_relation(relation, "subject.md", object_id)
        except RewriteRefused:
            continue
        destination(routed.key)  # must not raise: a routed key has a destination
```

- [ ] **Step 2: Run it to verify it fails**

```bash
python -m pytest tests/test_rewrite_routing_contract.py -q
```

Expected: FAIL, `ImportError: cannot import name 'route_relation' from 'recall.rewrite'`.

- [ ] **Step 3: Implement the resolver**

Add to `recall/rewrite.py`, immediately below `destination`, and add `from recall.truth_extraction.types import VALIDITY_CLAIM_KEYS` to the imports:

```python
@dataclass(frozen=True)
class Routed:
    """Where a proposal's relation lands: which key, what value, and in whose file."""

    key: str
    value: str
    edit_file: str


def route_relation(relation: str, subject_id: str, object_id: str) -> Routed:
    """Resolve a proposal's relation into the key, value and file a rewrite would touch.

    `_extracted.py` encodes the key as a prefix inside `object_id` for the two relations a
    document makes about itself, because neither is a relation BETWEEN two documents and
    forcing them into one would put a false relation into an audit record. Parsing that prefix
    is this function's job, and refusing an unrecognised one is the more important half: a
    prefix accepted loosely is a fourth frontmatter key invented by a malformed proposal.
    """
    if relation == "supersedes":
        # The ONLY relation whose edit lands on object_id. The schema has no `superseded_by`,
        # so the superseding document is the one that gains the key. Inverting this demotes
        # the live memo beneath the one it replaced.
        return Routed(key="supersedes", value=subject_id, edit_file=object_id)
    if relation == "declares_validity":
        key, sep, value = object_id.partition(":")
        if not sep or key not in VALIDITY_CLAIM_KEYS or not value:
            raise RewriteRefused(
                f"unroutable_validity: {object_id!r} does not name one of "
                f"{VALIDITY_CLAIM_KEYS} with a value"
            )
        return Routed(key=key, value=value, edit_file=subject_id)
    if relation == "declares_status":
        key, sep, value = object_id.partition(":")
        if key != "status" or not sep or not value:
            raise RewriteRefused(
                f"unroutable_status: {object_id!r} does not name a status value"
            )
        return Routed(key="status", value=value, edit_file=subject_id)
    if relation in ("contradicts", "same_entity"):
        return Routed(key=relation, value=object_id, edit_file=subject_id)
    raise RewriteRefused(
        f"unroutable_relation: {relation!r} has no key, so there is nowhere to write it"
    )
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
python -m pytest tests/test_rewrite_routing_contract.py -q
```

Expected: 8 passed.

- [ ] **Step 5: Make `plan_rewrite` use it, removing the inline routing**

In `recall/rewrite.py`, replace lines 386 to 397 of `plan_rewrite` (from `block = destination(...)` through the `RewritePlan(` field list) with:

```python
    routed = route_relation(checked.relation, checked.subject_id, checked.object_id)
    block = destination(routed.key)
    _refuse_unwritable_value(routed.value)
    return RewritePlan(
        edit_file=_resolve(root, routed.edit_file),
        key=routed.key,
        value=routed.value,
        block=block,
```

leaving the remaining `RewritePlan` fields (`claim`, `fact_id`, `proposal_id`, `reviewer_id`) unchanged. Delete the `if checked.relation == "supersedes":` branch: its asymmetry now lives in `route_relation`, and keeping both means two definitions of the direction rule that can drift apart.

- [ ] **Step 6: Prove the v2 relations now plan without raising**

Add to `tests/test_rewrite_routing_contract.py`:

```python
def test_plan_rewrite_handles_a_validity_relation(tmp_path, monkeypatch):
    """Before this task, `destination("declares_validity")` raised unknown_key."""
    from recall.rewrite import plan_rewrite

    (tmp_path / "memo.md").write_text("Body.\n", encoding="utf-8", newline="\n")
    fact = _promoted(relation="declares_validity", subject="memo.md", obj="valid_from:2026-07-14")
    plan = plan_rewrite(tmp_path, fact)
    assert plan.key == "valid_from"
    assert plan.value == "2026-07-14"
    assert plan.block == "frontmatter"
    assert plan.edit_file.endswith("memo.md")
```

Build `_promoted(...)` in that file with `recall.promotion.promote_accepted_proposal`, using the same reviewer, note and timestamp pattern as Task 6's chain test. Do **not** hand construct a `PromotedFact`: `_refuse_untrusted` re-checks the review fields and a hand built one would either bypass that check or fail it for the wrong reason.

```bash
python -m pytest tests/test_rewrite_routing_contract.py tests/test_corpus_rewrite_contract.py -q
```

Expected: all pass, including the 1502 line contract test merged in Task 1. If any of its tests fail here, the inline routing they pinned has changed behaviour; read the failure before changing the test, because that suite is the strongest evidence in this feature.

- [ ] **Step 7: Mutate each guard and watch it go red**

A guard that cannot fail is the recurring failure mode in this repo. Verify all three refusals really fire.

For each mutation below, apply it, run the test file, confirm the named test **fails**, then revert:

1. Change `key not in VALIDITY_CLAIM_KEYS` to `key not in VALIDITY_KEYS`. Expected red: `test_supersedes_cannot_be_smuggled_through_a_validity_prefix`.
2. Change `edit_file=object_id` in the `supersedes` branch to `edit_file=subject_id`. Expected red: `test_supersedes_edits_the_object_and_names_the_subject`.
3. Replace the final `raise RewriteRefused(...)` with `return Routed(key=relation, value=object_id, edit_file=subject_id)`. Expected red: `test_references_is_refused_because_it_has_no_key` **and** `test_every_relation_in_the_vocabulary_is_routed_or_refused`.

Record the observed failure output for each. If any mutation stays green, that guard is not testing what it claims.

- [ ] **Step 8: Lint, typecheck, commit**

```bash
python -m ruff check . && mypy recall
git add recall/rewrite.py tests/test_rewrite_routing_contract.py
git commit -m "Route a proposal's relation to a key, or refuse it"
```

---

### Task 4: The model engine behind the existing port

**Files:**
- Create: `recall/truth_extraction/_openai_engine.py`
- Modify: `recall/truth_extraction/_engine.py` (register in `_ENGINES`), `pyproject.toml` (add the `extract` extra)
- Test: `tests/test_truth_extraction_engine_openai.py`
- Reference only (do not import from it): `.claude/worktrees/truth-extraction-prose-1cfc81/recall/extraction.py` lines 450 to 530, class `_OpenAICompatChatClient` and `resolve_claim_extractor`.

**Interfaces:**
- Consumes: `ExtractionEngine` Protocol, `ExtractionPrompt`, `_ENGINES` (Task 2).
- Produces: `recall.truth_extraction._openai_engine.OpenAIExtractionEngine` with `run(prompt: ExtractionPrompt) -> str` and the identity attributes `model_id`, `revision` and `engine_id`. **As built, `engine_id` is an INSTANCE attribute**, `f"recall.truth_extraction.openai@{host}:{port}"`, because the endpoint has to be part of the audit identity: `extraction_cache_key` hashes only engine_id, model_id and revision, so two endpoints advertising the same model name would otherwise share one cache entry. Registered as `_ENGINES["openai"]`, whose factories take the resolved settings mapping so an explicit env reaches the engine instead of `os.environ`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_truth_extraction_engine_openai.py`:

```python
"""The model engine is one implementation of the existing port, held to the same ladder."""
import pytest

from recall.truth_extraction._engine import resolve_extraction_engine
from recall.truth_extraction._prompt import build_extraction_prompt


class _FakeChat:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[dict] = []

    def complete(self, messages: list[dict[str, str]], **kwargs: object) -> str:
        self.calls.append({"messages": messages, **kwargs})
        return self.reply


def test_the_engine_returns_the_model_text_unchanged():
    from recall.truth_extraction._openai_engine import OpenAIExtractionEngine

    chat = _FakeChat('{"claims": []}')
    engine = OpenAIExtractionEngine(client=chat, model_id="m", revision="r")
    prompt = build_extraction_prompt(file="a.md", human_body="body", corpus_names=("a.md",))
    assert engine.run(prompt) == '{"claims": []}'


def test_the_engine_calls_the_model_at_temperature_zero():
    from recall.truth_extraction._openai_engine import OpenAIExtractionEngine

    chat = _FakeChat('{"claims": []}')
    engine = OpenAIExtractionEngine(client=chat, model_id="m", revision="r")
    engine.run(build_extraction_prompt(file="a.md", human_body="b", corpus_names=("a.md",)))
    assert chat.calls[0]["temperature"] == 0


def test_the_engine_sends_the_rendered_system_and_user_prompts():
    from recall.truth_extraction._openai_engine import OpenAIExtractionEngine

    chat = _FakeChat('{"claims": []}')
    engine = OpenAIExtractionEngine(client=chat, model_id="m", revision="r")
    prompt = build_extraction_prompt(file="a.md", human_body="b", corpus_names=("a.md",))
    engine.run(prompt)
    roles = [m["role"] for m in chat.calls[0]["messages"]]
    assert roles == ["system", "user"]
    assert chat.calls[0]["messages"][1]["content"] == prompt.user


def test_an_unknown_engine_name_is_refused_rather_than_downgraded():
    with pytest.raises(ValueError, match="not a known engine"):
        resolve_extraction_engine(
            {"RECALL_TRUTH_EXTRACTION": "1", "RECALL_TRUTH_EXTRACTION_ENGINE": "gpt9"}
        )


def test_the_openai_engine_is_selectable_by_name():
    from recall.truth_extraction._engine import _ENGINES

    assert "openai" in _ENGINES
```

- [ ] **Step 2: Run it to verify it fails**

```bash
python -m pytest tests/test_truth_extraction_engine_openai.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'recall.truth_extraction._openai_engine'`.

- [ ] **Step 3: Implement the engine**

Create `recall/truth_extraction/_openai_engine.py`:

```python
"""An OpenAI compatible chat model as an `ExtractionEngine`.

This is one entry in `_ENGINES`, not a new architecture. Whatever the model returns still goes
through the full validation ladder in `_normalize.py`, so a model cannot skip a rung the rules
engine has to clear. That is the property that makes swapping the engine safe.

Temperature 0 is not a determinism guarantee from any hosted provider. `--recheck` exists to
MEASURE whether it holds rather than to assume it.

Requires `pip install recall[extract]`.
"""

from __future__ import annotations

import os
from typing import Protocol

from recall.truth_extraction._prompt import ExtractionPrompt

OPENAI_EXTRACTION_ENGINE_ID = "recall.truth_extraction.openai"
DEFAULT_EXTRACTION_MODEL = "anthropic/claude-sonnet-4.5"
DEFAULT_EXTRACTION_BASE_URL = "https://openrouter.ai/api/v1"


class ChatClient(Protocol):
    """The narrow slice of a chat API this engine uses."""

    def complete(self, messages: list[dict[str, str]], **kwargs: object) -> str:
        ...


class OpenAIExtractionEngine:
    """Answers an extraction prompt with a chat model. Supplies semantics, never identity."""

    engine_id = OPENAI_EXTRACTION_ENGINE_ID

    def __init__(self, *, client: ChatClient, model_id: str, revision: str) -> None:
        self._client = client
        self.model_id = model_id
        self.revision = revision

    def run(self, prompt: ExtractionPrompt) -> str:
        # `ExtractionPrompt` carries `system` and `user` as separate rendered strings, not a
        # message list. Assembling them here keeps the prompt module free of any one API's
        # message shape.
        return self._client.complete(
            [
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user},
            ],
            temperature=0,
        )


def _client_from_env(source: dict[str, str]) -> ChatClient:
    """Build the HTTP client, refusing clearly when the extra is not installed."""
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "the openai extraction engine requires: pip install recall[extract]"
        ) from exc

    key = source.get("RECALL_EXTRACTION_API_KEY", "").strip()
    if not key:
        raise ValueError(
            "RECALL_EXTRACTION_API_KEY is required for the openai extraction engine"
        )
    base_url = source.get("RECALL_EXTRACTION_BASE_URL", DEFAULT_EXTRACTION_BASE_URL)
    inner = OpenAI(api_key=key, base_url=base_url)
    model = source.get("RECALL_EXTRACTION_MODEL", DEFAULT_EXTRACTION_MODEL)

    class _Client:
        def complete(self, messages: list[dict[str, str]], **kwargs: object) -> str:
            reply = inner.chat.completions.create(
                model=model, messages=messages, **kwargs  # type: ignore[arg-type]
            )
            return reply.choices[0].message.content or ""

    return _Client()


def openai_engine_from_env(env: dict[str, str] | None = None) -> OpenAIExtractionEngine:
    """Construct the engine from environment settings."""
    source = dict(env if env is not None else os.environ)
    return OpenAIExtractionEngine(
        client=_client_from_env(source),
        model_id=source.get("RECALL_EXTRACTION_MODEL", DEFAULT_EXTRACTION_MODEL),
        revision=source.get("RECALL_EXTRACTION_REVISION", "unpinned"),
    )


__all__ = [
    "DEFAULT_EXTRACTION_BASE_URL",
    "DEFAULT_EXTRACTION_MODEL",
    "OPENAI_EXTRACTION_ENGINE_ID",
    "ChatClient",
    "OpenAIExtractionEngine",
    "openai_engine_from_env",
]
```

- [ ] **Step 4: Register it**

In `recall/truth_extraction/_engine.py`, replace the `_ENGINES` line:

```python
def _openai_factory() -> ExtractionEngine:
    from recall.truth_extraction._openai_engine import openai_engine_from_env

    return openai_engine_from_env()


#: Deferred construction: naming an engine must not import its dependency until it is chosen,
#: so `RECALL_TRUTH_EXTRACTION_ENGINE=deterministic` works with the extra absent.
_ENGINES: Mapping[str, Callable[[], ExtractionEngine]] = {
    "deterministic": DeterministicExtractionEngine,
    "openai": _openai_factory,
}
```

and change the construction line in `resolve_extraction_engine` from `engine: ExtractionEngine = _ENGINES[name]()` to keep working unchanged (the values are now callables returning an engine, so the call site is identical). Add `Callable` to the `collections.abc` import.

- [ ] **Step 5: Add the extra**

In `pyproject.toml`, under `[project.optional-dependencies]`, add:

```toml
extract = ["openai>=1.0"]
```

- [ ] **Step 6: Run the tests**

```bash
python -m pytest tests/test_truth_extraction_engine_openai.py tests/test_truth_extraction_contract.py -q
```

Expected: all pass.

- [ ] **Step 7: Verify the extra is genuinely optional**

```bash
python -c "from recall.truth_extraction import resolve_extraction_engine; print(resolve_extraction_engine({'RECALL_TRUTH_EXTRACTION': '1'}))"
```

Expected: a `DeterministicExtractionEngine` instance printed, with no import of `openai`. This proves naming the openai engine is what pulls the dependency, not importing the package.

- [ ] **Step 8: Lint, typecheck, commit**

```bash
python -m ruff check . && mypy recall
git add recall/truth_extraction/_openai_engine.py recall/truth_extraction/_engine.py pyproject.toml tests/test_truth_extraction_engine_openai.py
git commit -m "Add a model engine behind the extraction port"
```

---

### Task 5: Recheck on the merged cache

`1cfc81`'s recheck is written against its own cache. `ardinghelli`'s cache keys on engine plus prompt. Reimplement recheck onto the latter.

**Files:**
- Modify: `recall/truth_extraction/_cache.py`
- Test: `tests/test_truth_extraction_recheck.py`

**Interfaces:**
- Consumes: `ExtractionCache`, `extraction_cache_key` (Task 2); `ExtractionEngine` (Task 2).
- Produces: `recall.truth_extraction._cache.RecheckReport` (frozen dataclass with `checked: int`, `mismatched: int`, `mismatched_files: tuple[str, ...]`) and `recheck_cached_extractions(documents, *, engine, corpus_names, cache) -> RecheckReport`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_truth_extraction_recheck.py`:

```python
"""Recheck measures whether temperature 0 actually held, rather than assuming it did."""
from recall.truth_extraction._cache import RecheckReport, recheck_cached_extractions
from recall.truth_extraction._engine import DeterministicExtractionEngine
from recall.truth_extraction.extract import extract_corpus_claims


class _MemoryCache:
    def __init__(self) -> None:
        self.entries: dict[str, object] = {}

    def get(self, key: str):
        return self.entries.get(key)

    def put(self, key: str, value) -> None:
        self.entries[key] = value


class _DriftingEngine(DeterministicExtractionEngine):
    """Returns a different answer the second time it sees the same prompt."""

    def __init__(self) -> None:
        self.seen: set[str] = set()

    def run(self, prompt) -> str:
        if prompt.human_body in self.seen:
            return '{"claims": []}'
        self.seen.add(prompt.human_body)
        return super().run(prompt)


DOCS = {"a.md": "This replaces b.md entirely.", "b.md": "Older."}


def test_a_stable_engine_reports_zero_mismatches():
    cache, engine = _MemoryCache(), DeterministicExtractionEngine()
    extract_corpus_claims(DOCS, engine=engine, cache=cache)
    report = recheck_cached_extractions(
        DOCS, engine=engine, corpus_names=tuple(sorted(DOCS)), cache=cache
    )
    assert isinstance(report, RecheckReport)
    assert report.mismatched == 0
    assert report.checked == 2


def test_a_drifting_engine_is_reported_not_hidden():
    cache, engine = _MemoryCache(), _DriftingEngine()
    extract_corpus_claims(DOCS, engine=engine, cache=cache)
    report = recheck_cached_extractions(
        DOCS, engine=engine, corpus_names=tuple(sorted(DOCS)), cache=cache
    )
    assert report.mismatched >= 1
    assert "a.md" in report.mismatched_files


def test_recheck_leaves_the_cache_contents_alone():
    cache, engine = _MemoryCache(), _DriftingEngine()
    extract_corpus_claims(DOCS, engine=engine, cache=cache)
    before = dict(cache.entries)
    recheck_cached_extractions(
        DOCS, engine=engine, corpus_names=tuple(sorted(DOCS)), cache=cache
    )
    assert cache.entries == before, "recheck must measure the cache, not overwrite it"
```

- [ ] **Step 2: Run it to verify it fails**

```bash
python -m pytest tests/test_truth_extraction_recheck.py -q
```

Expected: FAIL with `ImportError: cannot import name 'RecheckReport'`.

- [ ] **Step 3: Implement recheck**

Append to `recall/truth_extraction/_cache.py`:

```python
@dataclass(frozen=True)
class RecheckReport:
    """How often a re-call disagreed with what the cache already held.

    A non zero `mismatched` means the CACHE, not the sampler, is what makes runs reproducible.
    That is worth knowing before a cache eviction silently renumbers every proposal id derived
    from it, so this is measured rather than assumed.
    """

    checked: int
    mismatched: int
    mismatched_files: tuple[str, ...]


def recheck_cached_extractions(
    documents: Mapping[str, str],
    *,
    engine: ExtractionEngine,
    corpus_names: Sequence[str],
    cache: ExtractionCache,
) -> RecheckReport:
    """Re-run the engine on already cached keys and report disagreement.

    Deliberately does NOT write back. Recheck is a measurement, and a measurement that mutates
    what it measures cannot be repeated.
    """
    from recall.truth_extraction._normalize import human_body_of, normalize_extraction
    from recall.truth_extraction._prompt import build_extraction_prompt
    from recall.truth_extraction.types import ExtractionBatchRejected

    checked = 0
    mismatched: list[str] = []
    for file, text in sorted(documents.items()):
        body = human_body_of(text)
        prompt = build_extraction_prompt(
            file=file, human_body=body, corpus_names=tuple(corpus_names)
        )
        cached = cache.get(extraction_cache_key(engine=engine, prompt=prompt))
        if cached is None:
            continue
        checked += 1
        try:
            claims, _ = normalize_extraction(
                engine.run(prompt),
                file=file,
                human_body=body,
                corpus_names=tuple(corpus_names),
            )
        except ExtractionBatchRejected:
            claims = ()
        if tuple(claims) != tuple(cached.claims):
            mismatched.append(file)
    return RecheckReport(
        checked=checked, mismatched=len(mismatched), mismatched_files=tuple(mismatched)
    )
```

Add `from dataclasses import dataclass` and `from collections.abc import Mapping, Sequence` to the module imports if absent.

- [ ] **Step 4: Run the tests**

```bash
python -m pytest tests/test_truth_extraction_recheck.py -q
```

Expected: 3 passed.

- [ ] **Step 5: Mutate the guard**

Change `if tuple(claims) != tuple(cached.claims):` to `if False:`. Run the file. Expected red: `test_a_drifting_engine_is_reported_not_hidden`. Revert.

- [ ] **Step 6: Lint, typecheck, commit**

```bash
python -m ruff check . && mypy recall
git add recall/truth_extraction/_cache.py tests/test_truth_extraction_recheck.py
git commit -m "Measure extraction determinism against the merged cache"
```

---

### Task 6: Close the chain end to end

Every link exists. No branch runs all four. This task is the reason the consolidation was worth doing.

**Files:**
- Create: `tests/test_truth_extraction_chain_contract.py`
- Modify: none expected. If a link does not fit, fix the link and note it in the commit.

**Interfaces:**
- Consumes: everything from Tasks 1 to 3.
- Produces: no new API. Proves `ExtractedClaim -> InferenceProposal -> ReviewedProposal -> PromotedFact -> file edit` runs.

- [ ] **Step 1: Write the contract test**

Create `tests/test_truth_extraction_chain_contract.py`:

```python
"""The full chain, which no single branch ever ran.

1. An extracted claim becomes an InferenceProposal with status `requires_review`.
2. A proposal cannot become a PromotedFact without a named reviewer and an audit note.
3. A PromotedFact routes to a key and rewrites the correct file.
4. A rewrite refuses a fact that did not pass promotion.
"""
from datetime import datetime, timezone
from pathlib import Path

import pytest

from recall.promotion import (
    ReviewedProposal,
    accept_reviewed_proposal,
    promote_accepted_proposal,
)
from recall.rewrite import RewriteRefused, apply_rewrite, plan_rewrite, route_relation

NOW = datetime(2026, 8, 12, tzinfo=timezone.utc)


def test_an_extracted_claim_arrives_as_requires_review(extracted_proposal):
    assert extracted_proposal.status == "requires_review"
    assert extracted_proposal.confidence is None


def test_promotion_refuses_without_a_named_reviewer(extracted_proposal):
    review = ReviewedProposal(
        proposal=extracted_proposal, state="reviewed", reviewer_id=None,
        reviewed_at=NOW, audit_note="looks right",
    )
    with pytest.raises(ValueError, match="reviewer identity is required"):
        accept_reviewed_proposal(review)


def test_promotion_refuses_without_an_audit_note(extracted_proposal):
    review = ReviewedProposal(
        proposal=extracted_proposal, state="reviewed", reviewer_id="gde",
        reviewed_at=NOW, audit_note="   ",
    )
    with pytest.raises(ValueError, match="audit note is required"):
        accept_reviewed_proposal(review)


def test_a_promoted_fact_rewrites_the_superseding_file(tmp_path, extracted_proposal):
    (tmp_path / "new.md").write_text("This replaces old.md.\n", encoding="utf-8", newline="\n")
    (tmp_path / "old.md").write_text("Older.\n", encoding="utf-8", newline="\n")
    review = ReviewedProposal(
        proposal=extracted_proposal, state="reviewed", reviewer_id="gde",
        reviewed_at=NOW, audit_note="checked the quote against the memo",
    )
    fact = promote_accepted_proposal(accept_reviewed_proposal(review), promoted_at=NOW)
    routed = route_relation(fact.relation, fact.subject_id, fact.object_id)
    assert routed.edit_file.endswith("new.md")

    plan = plan_rewrite(tmp_path, fact)
    apply_rewrite(tmp_path, fact, apply=True)

    assert "supersedes:" in (tmp_path / "new.md").read_text(encoding="utf-8")
    assert "supersedes:" not in (tmp_path / "old.md").read_text(encoding="utf-8")
    assert plan is not None


def test_a_rewrite_refuses_an_unpromoted_proposal(tmp_path, extracted_proposal):
    with pytest.raises((RewriteRefused, TypeError, ValueError)):
        apply_rewrite(tmp_path, extracted_proposal, apply=True)  # type: ignore[arg-type]
```

- [ ] **Step 2: Add the `extracted_proposal` fixture**

Append to `tests/conftest.py`:

This reuses the builders `tests/test_truth_extraction_contract.py` already proves out (`_pair_graph`, `_pair_provider`, `_extracted` at lines 560 to 586 of that file), rather than inventing a second way to construct a graph. Do not hand construct an `InferenceProposal`: a fixture that does cannot drift-detect the pipeline.

```python
@pytest.fixture
def extracted_proposal():
    """One InferenceProposal produced by the real extraction path, for chain tests.

    Built through `extract_corpus_claims`, `ExtractedClaimProposalProvider` and
    `proposal_report`, which is the same route the library takes, so this fixture goes red if
    any link changes shape.
    """
    from recall.reasoning_graph import build_reasoning_graph
    from recall.reasoning_proposals._extracted import ExtractedClaimProposalProvider
    from recall.reasoning_proposals._providers import proposal_report
    from recall.truth_extraction._engine import DeterministicExtractionEngine
    from recall.truth_extraction.extract import extract_corpus_claims
    from recall.types import Chunk

    documents = {
        "old_2026-01-01.md": "Older.\n",
        "new_2026-02-01.md": "This memo supersedes old_2026-01-01.md after review.\n",
    }
    graph = build_reasoning_graph(
        [
            Chunk(file.split("_")[0], f"/corpus/{file}", text, {"file": file, "ord": 0})
            for file, text in documents.items()
        ],
        tenant_id="acme",
        generation_id="gen_1",
        pipeline_fingerprint="pipe-a",
        include_text=True,
    )
    provider = ExtractedClaimProposalProvider(
        extract_corpus_claims(documents, engine=DeterministicExtractionEngine())
    )
    report = proposal_report(graph, model_provider=provider)
    assert report.provider_failures == (), report.provider_failures
    supersessions = [
        p
        for p in (*report.proposals, *report.rejected_proposals)
        if p.provider_id == provider.provider_id and p.proposed_relation == "supersedes"
    ]
    assert supersessions, "the deterministic engine must find the supersession"
    return supersessions[0]
```

Because the fixture now uses dated file stems, update the file names in the Task 6 test bodies to match: `new_2026-02-01.md` and `old_2026-01-01.md` in place of `new.md` and `old.md`.

- [ ] **Step 3: Run it**

```bash
python -m pytest tests/test_truth_extraction_chain_contract.py -q
```

Expected: this is the first run of the whole chain, so failures here are **information, not noise**. Fix each at its real link rather than by weakening the test. Do not add a shortcut past `promotion.py`: dropping `1cfc81`'s `promoted_prose_edge` was a deliberate decision recorded in the design doc.

- [ ] **Step 4: Lint, typecheck, commit**

```bash
python -m ruff check . && mypy recall
git add tests/test_truth_extraction_chain_contract.py tests/conftest.py
git commit -m "Run extraction through promotion into a file edit"
```

---

### Task 7: `recall extract` CLI

**Files:**
- Modify: `recall/cli.py` (parser near line 487 where `p_reasoning` is built; handler near line 921 where the `lint` filesystem command is handled)
- Test: `tests/test_cli_extract.py`

**Interfaces:**
- Consumes: `extract_corpus_claims`, `resolve_extraction_engine`, `recheck_cached_extractions` (Tasks 2, 5).
- Produces: `recall extract run <path>` and `recall extract show <file>` on the CLI.

- [ ] **Step 1: Write the failing CLI test**

Create `tests/test_cli_extract.py`:

```python
"""`recall extract` is a filesystem command: no DB, no embedder, OFF unless enabled."""
import pytest

from recall.cli import main


def _corpus(tmp_path):
    (tmp_path / "new.md").write_text("This replaces old.md.\n", encoding="utf-8", newline="\n")
    (tmp_path / "old.md").write_text("Older.\n", encoding="utf-8", newline="\n")
    return tmp_path


def test_extract_run_refuses_when_extraction_is_off(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("RECALL_TRUTH_EXTRACTION", raising=False)
    with pytest.raises(SystemExit) as exc:
        main(["extract", "run", str(_corpus(tmp_path))])
    assert exc.value.code == 2
    assert "RECALL_TRUTH_EXTRACTION" in capsys.readouterr().err


def test_extract_run_lists_claims_with_their_quotes(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("RECALL_TRUTH_EXTRACTION", "1")
    main(["extract", "run", str(_corpus(tmp_path))])
    out = capsys.readouterr().out
    assert "new.md" in out
    assert "supersession" in out
    assert "This replaces old.md." in out, "a claim must show the quote that proves it"


def test_extract_run_honours_limit(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("RECALL_TRUTH_EXTRACTION", "1")
    main(["extract", "run", str(_corpus(tmp_path)), "--limit", "1"])
    assert "1 file" in capsys.readouterr().out


def test_extract_run_help_states_it_writes_nothing(capsys):
    with pytest.raises(SystemExit):
        main(["extract", "run", "--help"])
    assert "writes nothing" in capsys.readouterr().out


def test_extract_show_prints_one_file_and_its_refusals(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("RECALL_TRUTH_EXTRACTION", "1")
    main(["extract", "show", str(_corpus(tmp_path) / "new.md")])
    out = capsys.readouterr().out
    assert "new.md" in out
```

- [ ] **Step 2: Run it to verify it fails**

```bash
python -m pytest tests/test_cli_extract.py -q
```

Expected: FAIL, `argument cmd: invalid choice: 'extract'`.

- [ ] **Step 3: Add the parser**

In `recall/cli.py`, after the `p_reasoning` block (near line 527), insert:

```python
    p_extract = sub.add_parser(
        "extract",
        help="extract structured truth claims from memo prose (no DB needed; writes nothing)",
    )
    extract_sub = p_extract.add_subparsers(dest="extract_cmd", required=True)
    p_extract_run = extract_sub.add_parser(
        "run",
        help="extract claims from a corpus. Writes nothing: use `recall rewrite` to declare "
        "an accepted claim.",
    )
    p_extract_run.add_argument("path")
    p_extract_run.add_argument("--glob", default=DEFAULT_GLOB)
    p_extract_run.add_argument(
        "--limit", type=int, default=None, help="stop after this many files"
    )
    p_extract_run.add_argument(
        "--recheck",
        action="store_true",
        help="re-call the engine on cached keys and report the mismatch rate, to measure "
        "whether determinism actually holds rather than assume it",
    )
    p_extract_run.add_argument(
        "--cache", default=None, help="path to the extraction cache (default: no cache)"
    )
    p_extract_show = extract_sub.add_parser(
        "show", help="show the claims and refusals for a single file"
    )
    p_extract_show.add_argument("file")
```

- [ ] **Step 4: Add the handler**

In `recall/cli.py`, immediately before the `if args.cmd == "check":` block (near line 1005), insert:

```python
    if args.cmd == "extract":  # pure filesystem path — no embedder, no DB
        from recall.truth_extraction import resolve_extraction_engine
        from recall.truth_extraction.extract import extract_corpus_claims

        try:
            engine = resolve_extraction_engine()
        except (ValueError, ImportError) as exc:
            print(f"recall extract: {exc}", file=sys.stderr)
            raise SystemExit(2) from exc
        if engine is None:
            print(
                "recall extract: extraction is off. Set RECALL_TRUTH_EXTRACTION=1 to enable "
                "it, and see docs/TRUTH_EXTRACTION_DESIGN.md for what it does.",
                file=sys.stderr,
            )
            raise SystemExit(2)

        root = Path(args.path if args.extract_cmd == "run" else args.file)
        if args.extract_cmd == "run":
            paths = sorted(root.glob(args.glob)) if root.is_dir() else [root]
            if args.limit is not None:
                paths = paths[: args.limit]
        else:
            paths = [root]
        documents = {
            p.name: p.read_text(encoding="utf-8-sig") for p in paths if p.is_file()
        }
        extractions = extract_corpus_claims(documents, engine=engine)
        for item in extractions:
            for claim in item.claims:
                print(f"  {item.file}: {claim.kind}")
                print(f"      quote {claim.quote!r}")
            for refusal in item.rejections:
                print(f"  SKIP {item.file}: {refusal.rung} — {refusal.reason}")
            if item.batch_rejection is not None:
                print(
                    f"  REFUSED {item.file}: {item.batch_rejection.rung} — "
                    f"{item.batch_rejection.reason}"
                )
        total = sum(len(item.claims) for item in extractions)
        print(f"\n{len(documents)} file(s) read, {total} claim(s) for review")
        print("nothing written — review with `recall rewrite plan`")
        raise SystemExit(0)
```

Note the em dash characters in the print strings above are inside **output text**, not prose punctuation, and match the existing `lint` handler's formatting. If the repo's linter objects, replace with a colon.

- [ ] **Step 5: Run the tests**

```bash
python -m pytest tests/test_cli_extract.py -q
```

Expected: 5 passed. If `--recheck` is exercised, wire it to `recheck_cached_extractions` from Task 5 before claiming the flag works.

- [ ] **Step 6: Lint, typecheck, commit**

```bash
python -m ruff check . && mypy recall
git add recall/cli.py tests/test_cli_extract.py
git commit -m "Add recall extract run and show"
```

---

### Task 8: `recall rewrite` CLI

The named human gate lives at the argument parser, before any code runs.

**Files:**
- Modify: `recall/cli.py`
- Test: `tests/test_cli_rewrite.py`

**Interfaces:**
- Consumes: `plan_rewrite`, `apply_rewrite`, `route_relation`, `RejectionLedger`, `default_ledger_path` (Tasks 1, 3).
- Produces: `recall rewrite plan|apply|reject|verify` on the CLI.

- [ ] **Step 1: Write the failing CLI test**

Create `tests/test_cli_rewrite.py`:

```python
"""The named human gate is an argparse requirement, so it fires before any code runs."""
import pytest

from recall.cli import main


def test_apply_without_a_reviewer_is_refused_by_the_parser(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["rewrite", "apply", str(tmp_path), "--proposal", "ip_x", "--note", "n"])
    assert exc.value.code == 2
    assert "--reviewer" in capsys.readouterr().err


def test_apply_without_a_note_is_refused_by_the_parser(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["rewrite", "apply", str(tmp_path), "--proposal", "ip_x", "--reviewer", "gde"])
    assert exc.value.code == 2
    assert "--note" in capsys.readouterr().err


def test_an_empty_reviewer_is_refused_even_though_argparse_accepts_it(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main([
            "rewrite", "apply", str(tmp_path), "--proposal", "ip_x",
            "--reviewer", "   ", "--note", "n",
        ])
    assert exc.value.code == 2
    assert "reviewer" in capsys.readouterr().err.lower()


def test_reject_requires_a_reviewer_and_a_note(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["rewrite", "reject", "--proposal", "ip_x"])
    assert exc.value.code == 2


def test_apply_is_a_dry_run_by_default(tmp_path, capsys):
    memo = tmp_path / "new.md"
    memo.write_text("This replaces old.md.\n", encoding="utf-8", newline="\n")
    (tmp_path / "old.md").write_text("Older.\n", encoding="utf-8", newline="\n")
    before = memo.read_text(encoding="utf-8")
    main([
        "rewrite", "plan", str(tmp_path),
    ])
    assert memo.read_text(encoding="utf-8") == before
    assert "dry run" in capsys.readouterr().out.lower()


def test_apply_help_states_the_dry_run_default(capsys):
    with pytest.raises(SystemExit):
        main(["rewrite", "apply", "--help"])
    assert "dry run" in capsys.readouterr().out.lower()
```

- [ ] **Step 2: Run it to verify it fails**

```bash
python -m pytest tests/test_cli_rewrite.py -q
```

Expected: FAIL, `argument cmd: invalid choice: 'rewrite'`.

- [ ] **Step 3: Add the parser**

In `recall/cli.py`, after the `p_extract` block:

```python
    p_rewrite = sub.add_parser(
        "rewrite",
        help="review extracted claims and declare accepted ones in corpus frontmatter",
    )
    rewrite_sub = p_rewrite.add_subparsers(dest="rewrite_cmd", required=True)

    p_rw_plan = rewrite_sub.add_parser(
        "plan", help="show what would be written, and change nothing"
    )
    p_rw_plan.add_argument("path")
    p_rw_plan.add_argument("--glob", default=DEFAULT_GLOB)

    p_rw_apply = rewrite_sub.add_parser(
        "apply",
        help="declare ONE reviewed proposal in its memo. DRY RUN by default: prints the plan "
        "and changes nothing unless --apply is given.",
    )
    p_rw_apply.add_argument("path")
    p_rw_apply.add_argument("--proposal", required=True, help="the proposal id to apply")
    p_rw_apply.add_argument(
        "--reviewer", required=True, help="the identity of the human accepting this proposal"
    )
    p_rw_apply.add_argument(
        "--note", required=True, help="why this proposal was accepted; kept in the audit record"
    )
    p_rw_apply.add_argument(
        "--apply", action="store_true", help="actually write the edit to the memo file"
    )

    p_rw_reject = rewrite_sub.add_parser(
        "reject", help="record a human's refusal so the proposal does not resurface"
    )
    p_rw_reject.add_argument("--proposal", required=True)
    p_rw_reject.add_argument("--reviewer", required=True)
    p_rw_reject.add_argument("--note", required=True)

    p_rw_verify = rewrite_sub.add_parser(
        "verify", help="check that every declared edge in a corpus still resolves"
    )
    p_rw_verify.add_argument("path")
```

- [ ] **Step 4: Add the handler with the non-empty check**

`required=True` accepts an empty string, so the gate needs a second half. In `recall/cli.py`, before the `check` block:

```python
    if args.cmd == "rewrite":  # pure filesystem path — no embedder, no DB
        from recall.rewrite import RewriteRefused

        # `required=True` is satisfied by `--reviewer ""`. A gate a caller passes by typing
        # nothing is a field, not a person, so the emptiness check is the other half of it.
        for field in ("reviewer", "note"):
            value = getattr(args, field, None)
            if value is not None and not value.strip():
                print(f"recall rewrite: --{field} must not be empty", file=sys.stderr)
                raise SystemExit(2)

        try:
            _run_rewrite(args)
        except RewriteRefused as exc:
            print(f"recall rewrite: {exc}", file=sys.stderr)
            raise SystemExit(2) from exc
        raise SystemExit(0)
```

Then add `_run_rewrite` as a module level helper in `recall/cli.py`:

```python
def _run_rewrite(args: argparse.Namespace) -> None:
    """Dispatch a rewrite verb.

    Proposals are NOT persisted anywhere, so every verb re-derives them from the corpus. That
    is affordable because extraction is cached and deterministic, and it is also the honest
    model: a proposal is a reading of the corpus as it stands now, not a stored verdict.
    """
    from datetime import datetime, timezone

    from recall.promotion import (
        ReviewedProposal,
        accept_reviewed_proposal,
        promote_accepted_proposal,
    )
    from recall.rewrite import (
        RejectionLedger,
        apply_rewrite,
        claim_key,
        default_ledger_path,
        plan_rewrite,
    )

    root = Path(args.path)
    proposals = _rewrite_proposals(root, getattr(args, "glob", DEFAULT_GLOB))

    if args.rewrite_cmd == "plan":
        with RejectionLedger(default_ledger_path(root)) as ledger:
            for proposal in proposals:
                claim = claim_key(
                    proposal.proposed_relation, proposal.subject_id, proposal.object_id
                )
                mark = "REJECTED" if ledger.is_rejected(claim) else "review"
                print(f"  {mark} {proposal.id}: {proposal.proposed_relation}")
                print(f"      {proposal.subject_id} -> {proposal.object_id}")
                print(f"      {proposal.explanation}")
        print(f"\n{len(proposals)} proposal(s)")
        print("dry run — nothing written. Use `recall rewrite apply` to declare one.")
        return

    chosen = next((p for p in proposals if p.id == args.proposal), None)
    if chosen is None:
        print(f"recall rewrite: no proposal {args.proposal!r} in {root}", file=sys.stderr)
        raise SystemExit(2)

    now = datetime.now(timezone.utc)
    review = ReviewedProposal(
        proposal=chosen, state="reviewed", reviewer_id=args.reviewer,
        reviewed_at=now, audit_note=args.note,
    )

    if args.rewrite_cmd == "reject":
        claim = claim_key(chosen.proposed_relation, chosen.subject_id, chosen.object_id)
        with RejectionLedger(default_ledger_path(root)) as ledger:
            ledger.reject(claim, reviewer_id=args.reviewer, audit_note=args.note)
        print(f"recorded rejection of {chosen.id} as claim {claim}")
        return

    fact = promote_accepted_proposal(accept_reviewed_proposal(review), promoted_at=now)
    plan = plan_rewrite(root, fact)
    print(f"  {plan.edit_file}: + {plan.key}: {plan.value}  (in {plan.block})")
    if not args.apply:
        # Dry run by DEFAULT: this edits the user's own documents, and a tool that rewrites
        # your memory the first time you try it has earned distrust.
        print("dry run — nothing written. Re-run with --apply to write this edge.")
        return
    with RejectionLedger(default_ledger_path(root)) as ledger:
        result = apply_rewrite(root, fact, ledger=ledger, apply=True)
    print("written" if result.written else f"not written: {result.refusal}")
```

`_rewrite_proposals(root, glob)` re-runs the Task 6 chain over the corpus and returns the `InferenceProposal` tuple. Build it by lifting the fixture body from Task 6 Step 2 into a function in `recall/rewrite.py` (not `cli.py`, so tests can call it without argparse), and have both the fixture and this call it.

**Deviation from the brief, stated rather than hidden.** The brief specifies `rewrite reject --proposal <id> --reviewer <id> --note "..."` with no path argument. That cannot work: the rejection ledger is keyed by *claim* (relation plus the two normalised document names), deliberately not by proposal id, because proposal ids hash in `generation_id` and would forget every rejection at the next re-index. Resolving a proposal id to a claim key requires the corpus. So `reject` takes `path` as its first positional, exactly like the other three verbs. Adjust the Step 3 parser accordingly:

```python
    p_rw_reject.add_argument("path")
```

and update the Step 1 test `test_reject_requires_a_reviewer_and_a_note` to pass a path.

- [ ] **Step 5: Run the tests**

```bash
python -m pytest tests/test_cli_rewrite.py -q
```

Expected: 6 passed.

- [ ] **Step 6: Mutate the gate and watch it go red**

Delete the emptiness loop from Step 4. Run the file. Expected red: `test_an_empty_reviewer_is_refused_even_though_argparse_accepts_it`. Restore it. If it stays green, the check is not reachable and the gate is decorative.

- [ ] **Step 7: Lint, typecheck, commit**

```bash
python -m ruff check . && mypy recall
git add recall/cli.py tests/test_cli_rewrite.py
git commit -m "Add recall rewrite with the human gate at the parser"
```

---

### Task 9: The MCP surface

Ship the read only half. Deliberately ship no apply.

**Files:**
- Modify: `recall_mcp/server.py` (near line 980, beside `recall_reasoning_proposals`), `recall_mcp/service.py` (`reasoning_proposals`, line 1206)
- Test: `tests/test_mcp_rewrite_plan.py`

**Interfaces:**
- Consumes: `plan_rewrite` (Task 1), `ExtractedClaimProposalProvider` (Task 2).
- Produces: MCP tool `recall_rewrite_plan`; new keyword `include_extracted: bool = False` on `recall_mcp.service.reasoning_proposals` and on the `recall_reasoning_proposals` tool.

- [ ] **Step 1: Write the failing test**

Create `tests/test_mcp_rewrite_plan.py`:

```python
"""The MCP surface proposes. A human applies at the CLI.

The MCP client IS the model. Letting it supply `reviewer_id` and `audit_note` would make the
named human gate a formality it satisfies by typing a string.
"""
import inspect

import recall_mcp.server as server
from recall_mcp.service import reasoning_proposals


def test_no_apply_tool_is_registered():
    source = inspect.getsource(server)
    assert "recall_rewrite_apply" not in source, "the MCP surface must not mutate documents"


def test_the_plan_tool_is_registered_and_read_only():
    source = inspect.getsource(server)
    assert "recall_rewrite_plan" in source
    index = source.index("recall_rewrite_plan")
    window = source[index : index + 400]
    assert "read_only_hint=True" in window
    assert "destructive_hint=False" in window


def test_mcp_still_makes_no_file_write_calls():
    source = inspect.getsource(server)
    for forbidden in ("write_text(", "atomic_write_bytes(", "open(", "apply_rewrite("):
        assert forbidden not in source, f"{forbidden} would make MCP mutate a user's documents"


def test_include_extracted_defaults_to_false():
    signature = inspect.signature(reasoning_proposals)
    assert signature.parameters["include_extracted"].default is False


def test_include_extracted_refuses_rather_than_returning_a_misleading_empty():
    """No extraction is persisted yet, and "0 proposals" would read as "found nothing"."""
    from recall_mcp.service import _stored_extracted_proposals

    with pytest.raises(ValueError, match="no extraction record"):
        _stored_extracted_proposals(object())
```

Add `import pytest` to the test file's imports.

- [ ] **Step 2: Run it to verify it fails**

```bash
python -m pytest tests/test_mcp_rewrite_plan.py -q
```

Expected: FAIL on `test_the_plan_tool_is_registered_and_read_only` and `test_include_extracted_defaults_to_false`.

- [ ] **Step 3: Add `include_extracted` to the service**

In `recall_mcp/service.py`, change the `reasoning_proposals` signature and body:

```python
def reasoning_proposals(
    store: PgVectorStore, *, limit: int = 100, include_extracted: bool = False
) -> ReasoningProposalResult:
    if limit < 1:
        raise ValueError("proposal limit must be positive")
    graph = project_store_graph(store, include_text=True)
    proposals = deterministic_inference_proposals(
        graph, pipeline_id=graph.pipeline_fingerprint or "legacy"
    )
    if include_extracted:
        # Extraction runs on the INGEST path. This replays a stored result and calls nothing,
        # which is what keeps `max_model_calls = 0` true on the query path.
        proposals = proposals + _stored_extracted_proposals(graph)
    ...
```

**Verified before writing this step:** `FileExtraction` is persisted nowhere the query path can read. `recall/truth_extraction/_cache.py` defines `ExtractionCache` as a Protocol with no shipped database implementation, and no module outside `recall/truth_extraction/` and `recall/reasoning_proposals/_extracted.py` references the type at all. So there is currently no store for this function to read.

That makes the obvious stub wrong. Returning `()` would make `--include-extracted` print "0 proposals", which a caller reads as *the extractor found nothing* when the truth is *nothing was ever recorded*. This repo refuses rather than returning a misleading empty, so:

```python
def _stored_extracted_proposals(
    graph: ReasoningGraphProjection,
) -> tuple[InferenceProposal, ...]:
    """Replay extractions recorded at ingest into the proposal protocol.

    Refuses when no extraction record exists for this generation. An empty tuple would be
    indistinguishable from "the extractor ran and found nothing", and the two call for
    opposite responses from whoever asked.
    """
    raise ValueError(
        "no extraction record exists for this generation. Run `recall extract run <path>` "
        "on the ingest side first; extraction never runs on the query path."
    )
```

**Never** construct an `ExtractionEngine` in this module. Persisting extractions so this can return real data is deliberately out of scope for this plan: it needs a generation scoped table and a migration, and the design doc's argument against a migration (claims are content keyed and generation independent) has to be revisited before one is added.

- [ ] **Step 4: Add the tool**

In `recall_mcp/server.py`, beside `recall_reasoning_proposals`:

```python
    @mcp.tool(
        name="recall_rewrite_plan",
        annotations=ToolAnnotations(
            title="Plan a corpus rewrite",
            read_only_hint=True,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=False,
        ),
    )
    async def recall_rewrite_plan(ctx: Context[dict, object], proposal_id: str) -> str:
        """Show what declaring a reviewed proposal would write. Writes nothing.

        There is deliberately no `recall_rewrite_apply`. Applying requires a named human at
        the CLI, because a reviewer id this tool could supply is a field, not a person.
        """
        store = _require(SCOPE_READ, ctx)
        with METRICS.timer("recall_tool_latency_ms", tool="rewrite_plan"):
            return await _to_thread(
                lambda: rewrite_plan(store, proposal_id=proposal_id).model_dump_json(indent=2)
            )
```

and update `recall_reasoning_proposals` to take and forward `include_extracted: bool = False`.

- [ ] **Step 5: Run the tests**

```bash
python -m pytest tests/test_mcp_rewrite_plan.py -q
```

Expected: 4 passed.

- [ ] **Step 6: Prove existing behaviour is byte identical by default**

```bash
RECALL_TEST_DSN=postgresql://recall:recall@127.0.0.1:5434/recall python -m pytest tests/ -k "mcp" -q
```

Expected: every pre-existing MCP test still passes unchanged. `include_extracted` defaulting to `False` is what makes that true; if any MCP test needed editing, the default is wrong.

- [ ] **Step 7: Add the matching CLI flag**

The spec lists `recall reasoning proposals --include-extracted` alongside the MCP flag. In `recall/cli.py`, replace the bare parser at line 501:

```python
    p_reasoning_proposals = reasoning_sub.add_parser(
        "proposals", help="inspect deterministic inference proposals"
    )
    p_reasoning_proposals.add_argument(
        "--include-extracted",
        action="store_true",
        help="also list proposals replayed from prose extraction recorded at ingest. Refuses "
        "if nothing was recorded: extraction never runs on the query path.",
    )
```

and forward it in the handler at line 1093:

```python
            if args.reasoning_cmd == "proposals":
                result = reasoning_proposals(
                    store, include_extracted=args.include_extracted
                )
```

- [ ] **Step 8: Test the flag both ways**

Add to `tests/test_mcp_rewrite_plan.py`:

```python
def test_the_cli_flag_defaults_to_off():
    from recall.cli import build_parser  # or the parser factory this module exposes

    args = build_parser().parse_args(["reasoning", "proposals"])
    assert args.include_extracted is False
```

If `recall/cli.py` has no separable parser factory, assert instead that `--include-extracted` appears in `recall reasoning proposals --help` output, captured the same way `tests/test_cli_extract.py` captures help text.

```bash
python -m pytest tests/test_mcp_rewrite_plan.py -q
```

Expected: all pass.

- [ ] **Step 9: Lint, typecheck, commit**

```bash
python -m ruff check . && mypy recall recall_mcp
git add recall_mcp/server.py recall_mcp/service.py recall/cli.py tests/test_mcp_rewrite_plan.py
git commit -m "Expose rewrite planning over MCP, and no way to apply"
```

---

### Task 10: Documentation and final verification

**Files:**
- Modify: `docs/API.md`, `docs/REPOSITORY_MAP.md`, `docs/README.md`, `README.md`, `CHANGELOG.md`, `docs/INFERENCE_PROPOSALS.md`

- [ ] **Step 1: Document the CLI and MCP surfaces**

Add `recall extract` and `recall rewrite` to `docs/API.md` beside the other commands, and `recall_rewrite_plan` to the MCP tool list. State in both places that there is no apply tool and why.

- [ ] **Step 2: Update the repository map**

`docs/REPOSITORY_MAP.md` and `docs/README.md` are the maps this repo expects to be updated when a package is added. Add `recall/truth_extraction/`.

- [ ] **Step 3: Add a CHANGELOG entry**

State the `PROPOSAL_SCHEMA_VERSION` bump from 1 to 2 explicitly, and that it changes every `ip_` id. That is a breaking change for anyone holding stored ids.

- [ ] **Step 4: Full suite over everything**

```bash
RECALL_TEST_DSN=postgresql://recall:recall@127.0.0.1:5434/recall python -m pytest -q
```

Expected: green, ~12 minutes. **Check the skip count**: ~516 skipped means the DB was down and the run proves nothing.

- [ ] **Step 5: Verify the commit, not the worktree**

A green suite over a tree holding uncommitted fixes is not evidence about the commit.

```bash
git status --short
```

Expected: empty. If anything is uncommitted, commit it and re-run Step 4.

- [ ] **Step 6: Lint and typecheck one final time**

```bash
python -m ruff check . && mypy recall recall_mcp
```

Expected: clean. Do not run `ruff format`.

- [ ] **Step 7: Commit**

```bash
git add docs/ README.md CHANGELOG.md
git commit -m "Document truth extraction and reviewed rewrites"
```

---

## Notes for the implementer

**The prior is a measured failure.** `recall/fix.py:10` states that the rule based version proposed zero edges on a real 792 memo corpus, and that all four candidates surviving its mechanical rules were wrong on review. Read that whole docstring before writing anything in `rewrite.py`. A non-empty proposal list is a question, not an answer.

**Three branches got here before you.** They are `claude/suspicious-ardinghelli-3d3e31`, `claude/truth-extraction-prose-a7834c` and `claude/truth-extraction-prose-1cfc81`. If something in this plan seems to be missing, check whether one of them already solved it before writing it fresh.

**Do not add a fourth frontmatter key.** `FRONTMATTER_KEYS` is `VALIDITY_KEYS`, imported rather than restated, so the set cannot drift. Anything the frontmatter has no vocabulary for goes in the derived block.
