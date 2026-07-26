# Memory Abstention Benchmark (RE-call vs Mem0) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> ⚠️ **Local-only — do NOT commit/push to the public repo until article time.** Competitive-benchmark
> strategy. The harness code + results become public when we publish; this plan stays local.
> Because of that, tasks that would normally `git commit` instead say **"stage locally (do not push)"** —
> commit to a local `bench/head-to-head` branch that is never pushed until we decide to.

**Goal:** Build a reproducible harness that runs RE-call and Mem0 through an identical LLM generator on LOCOMO, measuring adversarial abstention (the 22.5% Mem0 skips) alongside answerable accuracy.

**Architecture:** A `MemorySystem` interface (`ingest`/`retrieve`) with two adapters (RE-call, Mem0). A shared, dependency-injected pipeline runs every arm through the same OpenRouter generator + judge; only the retrieved context differs. Pure logic (abstention detection, judging, aggregation) is unit-tested offline with fake LLM/systems; real-dependency adapters get skippable integration tests.

**Tech Stack:** Python 3.11+, `openai` SDK (pointed at OpenRouter), `mem0ai`, existing `recall` package (loader, `trusted_search`, `_rate`), Postgres+pgvector (RE-call arm), a local vector store (Mem0 arm), free `fastembed` bge-small embedder.

## Global Constraints

- **Never a core/CI dependency:** `mem0ai`, `openai` live in a NEW optional extra `bench` only — never in `dev`, never installed in CI. Copy verbatim into `pyproject.toml`: `bench = ["mem0ai", "openai>=1.0"]`.
- **One LLM gateway:** all LLM calls (generator, judge, Mem0's internal LLM) go through OpenRouter (`base_url="https://openrouter.ai/api/v1"`), OpenAI model, `temperature=0`. Key from env `OPENROUTER_API_KEY`.
- **Embeddings:** controlled arms use the free local `fastembed` bge-small (`recall`'s default) for BOTH systems. OpenAI-embeddings arm is off by default (needs `OPENAI_API_KEY`, deferred).
- **Fairness invariants:** identical generator prompt + judge prompt + model across all arms; only `retrieve()` output differs. Every run dumps per-question raw records (context + answer + verdict).
- **Reporting:** always emit BOTH columns (answerable accuracy + adversarial abstention) plus answerable-false-abstain, each with Wilson 95% CI + n (reuse `recall.eval.locomo._rate`).
- **The literal abstention token is `NO_ANSWER`.** The generator is instructed to output exactly that when the memories don't contain the answer.
- **Line length 100, ruff + mypy (strict) clean, tests DB/LLM-free where marked.** Match repo conventions.

---

### Task 1: Package scaffold + abstention helper + `bench` extra

**Files:**
- Create: `benchmarks/__init__.py`
- Create: `benchmarks/pipeline.py`
- Create: `tests/test_bench_pipeline.py`
- Modify: `pyproject.toml` (add `bench` extra; add `benchmarks` to mypy `files` and hatch wheel packages ONLY if we later ship it — for now do NOT add to wheel; add to mypy files so it's type-checked locally)

**Interfaces:**
- Produces: `NO_ANSWER: str`, `is_abstention(answer: str) -> bool`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_bench_pipeline.py
from benchmarks.pipeline import NO_ANSWER, is_abstention


def test_is_abstention_exact_token() -> None:
    assert is_abstention(NO_ANSWER) is True


def test_is_abstention_is_whitespace_and_case_tolerant() -> None:
    assert is_abstention("  no_answer\n") is True
    assert is_abstention("No_Answer") is True


def test_is_abstention_false_for_real_answer() -> None:
    assert is_abstention("The limit is 500 rps.") is False
    # a real answer that merely mentions the token is not an abstention
    assert is_abstention("There is no answer key labelled NO_ANSWER here, but it is 500.") is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_bench_pipeline.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'benchmarks'`

- [ ] **Step 3: Write minimal implementation**

```python
# benchmarks/__init__.py
"""Local competitive-benchmark harness (RE-call vs Mem0). Not shipped; not in CI."""
```

```python
# benchmarks/pipeline.py
from __future__ import annotations

#: The exact token the generator must emit when the memories don't answer the question.
NO_ANSWER = "NO_ANSWER"


def is_abstention(answer: str) -> bool:
    """True iff the generated answer is exactly the abstention token (case/space-insensitive).

    Requires the WHOLE answer to be the token — an answer that merely mentions ``NO_ANSWER`` in a
    sentence is a real (if odd) answer, not an abstention.
    """
    return answer.strip().casefold() == NO_ANSWER.casefold()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_bench_pipeline.py -q`
Expected: PASS (3 tests)

- [ ] **Step 5: Add the `bench` extra**

In `pyproject.toml` under `[project.optional-dependencies]`, add (do NOT touch `dev`):

```toml
# Local competitive benchmark only (benchmarks/). NEVER added to `dev` or installed in CI — it is a
# heavy tree (mem0ai) and the benchmark is run deliberately, by hand, not on every push.
bench = ["mem0ai", "openai>=1.0"]
```

- [ ] **Step 6: Stage locally (do not push)**

```bash
git checkout -b bench/head-to-head   # first task only; reuse thereafter
git add benchmarks/__init__.py benchmarks/pipeline.py tests/test_bench_pipeline.py pyproject.toml
git commit -m "bench: package scaffold + abstention helper + bench extra"
```

---

### Task 2: OpenRouter LLM client

**Files:**
- Create: `benchmarks/llm.py`
- Create: `tests/test_bench_llm.py`

**Interfaces:**
- Produces: `LLM` protocol with `complete(system: str, user: str) -> str`; `OpenRouterLLM(model, api_key, base_url=..., temperature=0.0)`; type alias `Completer = Callable[[str, str], str]`.

- [ ] **Step 1: Write the failing test** (construction + payload only — no network)

```python
# tests/test_bench_llm.py
from benchmarks.llm import Completer, OpenRouterLLM


def test_completer_is_callable_alias() -> None:
    # a plain function satisfies the injected-LLM seam used everywhere downstream
    fn: Completer = lambda system, user: "ok"
    assert fn("s", "u") == "ok"


def test_openrouter_llm_builds_with_defaults() -> None:
    llm = OpenRouterLLM(model="openai/gpt-4o-mini", api_key="sk-test")
    assert llm.model == "openai/gpt-4o-mini"
    assert llm.base_url == "https://openrouter.ai/api/v1"
    assert llm.temperature == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_bench_llm.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'benchmarks.llm'`

- [ ] **Step 3: Write minimal implementation**

```python
# benchmarks/llm.py
from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

#: The injected-LLM seam: (system_prompt, user_prompt) -> completion text. Everything downstream
#: depends on this, not on any SDK, so the pipeline is testable with a plain function.
Completer = Callable[[str, str], str]


class LLM(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class OpenRouterLLM:
    """OpenAI-compatible chat client pointed at OpenRouter. Lazily imports the `openai` SDK so the
    module imports without the `bench` extra installed (construction is what tests exercise)."""

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        temperature: float = 0.0,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self._api_key = api_key
        self._client: object | None = None

    def complete(self, system: str, user: str) -> str:
        from openai import OpenAI  # lazy: only needed at real run time

        if self._client is None:
            self._client = OpenAI(api_key=self._api_key, base_url=self.base_url)
        resp = self._client.chat.completions.create(  # type: ignore[attr-defined]
            model=self.model,
            temperature=self.temperature,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        content = resp.choices[0].message.content
        return content or ""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_bench_llm.py -q`
Expected: PASS (2 tests)

- [ ] **Step 5: Stage locally**

```bash
git add benchmarks/llm.py tests/test_bench_llm.py
git commit -m "bench: OpenRouter LLM client + Completer seam"
```

---

### Task 3: Generator + judge (pipeline core, fully offline-testable)

**Files:**
- Modify: `benchmarks/pipeline.py`
- Modify: `tests/test_bench_pipeline.py`

**Interfaces:**
- Consumes: `Completer` (Task 2), `NO_ANSWER`/`is_abstention` (Task 1)
- Produces: `GEN_SYSTEM_PROMPT`, `JUDGE_SYSTEM_PROMPT`, `generate_answer(completer, context, question) -> str`, `judge_correct(completer, question, gold, answer) -> bool`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_bench_pipeline.py
from benchmarks.pipeline import generate_answer, judge_correct


def test_generate_answer_passes_context_and_question_to_llm() -> None:
    seen = {}
    def completer(system: str, user: str) -> str:
        seen["system"], seen["user"] = system, user
        return "500 rps"
    out = generate_answer(completer, context="rate limit is 500 rps", question="how many rps?")
    assert out == "500 rps"
    assert "NO_ANSWER" in seen["system"]              # generator is told how to abstain
    assert "rate limit is 500 rps" in seen["user"]    # context is provided
    assert "how many rps?" in seen["user"]


def test_generate_answer_empty_context_still_calls_llm() -> None:
    # RE-call abstains by returning empty context; the generator must then emit NO_ANSWER itself
    def completer(system: str, user: str) -> str:
        return "NO_ANSWER"
    assert generate_answer(completer, context="", question="q") == "NO_ANSWER"


def test_judge_correct_parses_yes_no() -> None:
    assert judge_correct(lambda s, u: "YES", "q", "500", "500 rps") is True
    assert judge_correct(lambda s, u: "no", "q", "500", "42") is False
    # judge must be robust to a verbose reply
    assert judge_correct(lambda s, u: "YES, they match.", "q", "500", "500 rps") is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_bench_pipeline.py -q`
Expected: FAIL — `ImportError: cannot import name 'generate_answer'`

- [ ] **Step 3: Write minimal implementation** (append to `benchmarks/pipeline.py`)

```python
from benchmarks.llm import Completer

GEN_SYSTEM_PROMPT = (
    "You answer questions about a conversation using ONLY the provided memories. "
    f"If the answer is not present in the memories, respond with exactly {NO_ANSWER} and nothing "
    "else. Do not use outside knowledge. Keep answers short."
)

JUDGE_SYSTEM_PROMPT = (
    "You are grading whether a predicted answer matches the gold answer to a question. "
    "Reply with exactly YES if the prediction is correct (same meaning as gold), otherwise NO."
)


def generate_answer(completer: Completer, context: str, question: str) -> str:
    user = f"Memories:\n{context}\n\nQuestion: {question}\nAnswer:"
    return completer(GEN_SYSTEM_PROMPT, user).strip()


def judge_correct(completer: Completer, question: str, gold: str, answer: str) -> bool:
    user = f"Question: {question}\nGold answer: {gold}\nPredicted answer: {answer}\nCorrect?"
    verdict = completer(JUDGE_SYSTEM_PROMPT, user).strip().casefold()
    return verdict.startswith("yes")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_bench_pipeline.py -q`
Expected: PASS (all pipeline tests)

- [ ] **Step 5: Stage locally**

```bash
git add benchmarks/pipeline.py tests/test_bench_pipeline.py
git commit -m "bench: generator + judge prompts and functions"
```

---

### Task 4: Outcome record, per-question runner, aggregation (both columns)

**Files:**
- Modify: `benchmarks/pipeline.py`
- Modify: `tests/test_bench_pipeline.py`

**Interfaces:**
- Consumes: `generate_answer`, `judge_correct`, `is_abstention` (Tasks 1/3); `recall.eval.locomo._rate`
- Produces:
  - `@dataclass(frozen=True) Outcome(question_id: str, category: str, is_adversarial: bool, context: str, answer: str, abstained: bool, correct: bool | None)`
  - `run_question(retrieve: Callable[[str], str], completer: Completer, q: dict) -> Outcome`
  - `aggregate(outcomes: list[Outcome]) -> dict` with keys `answerable_accuracy`, `adversarial_abstention`, `answerable_false_abstain` (each `{rate, n, ci95}` from `_rate`), plus `by_category`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_bench_pipeline.py
from benchmarks.pipeline import Outcome, aggregate, run_question


def _q(qid, cat, adversarial, question="q", answer="500"):
    return {"question_id": qid, "category": cat, "adversarial": adversarial,
            "question": question, "answer": answer}


def test_run_question_answerable_correct() -> None:
    retrieve = lambda _q: "rate limit is 500 rps"
    completer = lambda system, user: "YES" if "Correct?" in user else "500 rps"
    out = run_question(retrieve, completer, _q("1", "cat1", False))
    assert out.is_adversarial is False
    assert out.abstained is False
    assert out.correct is True


def test_run_question_adversarial_abstains() -> None:
    retrieve = lambda _q: ""                       # RE-call abstained: empty context
    completer = lambda system, user: "NO_ANSWER"   # generator abstains
    out = run_question(retrieve, completer, _q("2", "cat5", True))
    assert out.is_adversarial is True
    assert out.abstained is True
    assert out.correct is None                     # correctness is undefined for adversarials


def test_aggregate_reports_both_columns() -> None:
    outs = [
        Outcome("1", "cat1", False, "c", "a", abstained=False, correct=True),
        Outcome("2", "cat1", False, "c", "a", abstained=False, correct=False),
        Outcome("3", "cat5", True, "", "NO_ANSWER", abstained=True, correct=None),
        Outcome("4", "cat5", True, "c", "wrong", abstained=False, correct=None),
    ]
    agg = aggregate(outs)
    assert agg["answerable_accuracy"]["n"] == 2
    assert agg["answerable_accuracy"]["rate"] == 0.5
    assert agg["adversarial_abstention"]["n"] == 2
    assert agg["adversarial_abstention"]["rate"] == 0.5
    assert agg["answerable_false_abstain"]["n"] == 2
    assert agg["answerable_false_abstain"]["rate"] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_bench_pipeline.py -q`
Expected: FAIL — `ImportError: cannot import name 'Outcome'`

- [ ] **Step 3: Write minimal implementation** (append to `benchmarks/pipeline.py`)

```python
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from recall.eval.locomo import _rate


@dataclass(frozen=True)
class Outcome:
    question_id: str
    category: str
    is_adversarial: bool
    context: str
    answer: str
    abstained: bool
    correct: bool | None  # None for adversarials (no gold answer to be "correct" about)


def run_question(retrieve: Callable[[str], str], completer: Completer, q: dict[str, Any]) -> Outcome:
    context = retrieve(q["question"])
    answer = generate_answer(completer, context, q["question"])
    abstained = is_abstention(answer)
    adversarial = bool(q["adversarial"])
    correct = None if adversarial else (
        False if abstained else judge_correct(completer, q["question"], q["answer"], answer)
    )
    return Outcome(
        question_id=str(q["question_id"]),
        category=str(q["category"]),
        is_adversarial=adversarial,
        context=context,
        answer=answer,
        abstained=abstained,
        correct=correct,
    )


def aggregate(outcomes: list[Outcome]) -> dict[str, Any]:
    answerable = [o for o in outcomes if not o.is_adversarial]
    adversarial = [o for o in outcomes if o.is_adversarial]
    by_cat: dict[str, dict[str, Any]] = {}
    for cat in sorted({o.category for o in outcomes}):
        ans_c = [o for o in answerable if o.category == cat]
        adv_c = [o for o in adversarial if o.category == cat]
        by_cat[cat] = {
            "answerable_accuracy": _rate([bool(o.correct) for o in ans_c]),
            "adversarial_abstention": _rate([o.abstained for o in adv_c]),
        }
    return {
        "answerable_accuracy": _rate([bool(o.correct) for o in answerable]),
        "adversarial_abstention": _rate([o.abstained for o in adversarial]),
        "answerable_false_abstain": _rate([o.abstained for o in answerable]),
        "by_category": by_cat,
    }
```

> Implementer note: confirm `recall.eval.locomo._rate` returns a dict shaped `{"rate", "n", "ci95"}`.
> It does (used by the existing abstention runner). If its key names differ, adapt the asserts.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_bench_pipeline.py -q`
Expected: PASS

- [ ] **Step 5: ruff + mypy the pure core**

Run: `.venv/Scripts/python.exe -m ruff check benchmarks tests/test_bench_pipeline.py tests/test_bench_llm.py && .venv/Scripts/python.exe -m mypy benchmarks/pipeline.py benchmarks/llm.py`
Expected: clean

- [ ] **Step 6: Stage locally**

```bash
git add benchmarks/pipeline.py tests/test_bench_pipeline.py
git commit -m "bench: Outcome + run_question + both-column aggregation"
```

---

### Task 5: RE-call system adapter

**Files:**
- Create: `benchmarks/systems.py`
- Create: `tests/test_bench_systems.py`

**Interfaces:**
- Consumes: `recall` package (`PgVectorStore`, embedder factory, `trusted_search`) and `recall.eval.locomo` loader/indexer
- Produces: `MemorySystem` protocol (`name: str`, `ingest(conversation) -> None`, `retrieve(question) -> str`); `RecallSystem(dsn, embedder_name="fastembed", k=5, ...)`

- [ ] **Step 1: Read the existing LOCOMO machinery**

Read `recall/eval/locomo.py`. Identify: the conversation loader, the per-turn indexing routine (how it builds `D{i}_{j}.md` docs and the `locomo_chunks` table with one tenant per conversation), `_make_embedder`, and the `trusted_search` call. RE-call's adapter must REUSE these, not reimplement them. If the indexing is inline in `run()`, extract a reusable `index_conversation(store, embedder, conversation)` in `locomo.py` (small refactor, keep its tests green).

- [ ] **Step 2: Write the failing test** (unit-level; the protocol + a fake, no DB)

```python
# tests/test_bench_systems.py
from benchmarks.systems import MemorySystem


class _FakeSystem:
    name = "fake"
    def __init__(self) -> None:
        self.ingested: list[dict] = []
    def ingest(self, conversation: dict) -> None:
        self.ingested.append(conversation)
    def retrieve(self, question: str) -> str:
        return f"ctx for {question}"


def test_fake_satisfies_protocol() -> None:
    sys: MemorySystem = _FakeSystem()
    sys.ingest({"sample_id": "c1"})
    assert sys.retrieve("q") == "ctx for q"
    assert sys.name == "fake"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_bench_systems.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'benchmarks.systems'`

- [ ] **Step 4: Write the protocol + RecallSystem**

```python
# benchmarks/systems.py
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class MemorySystem(Protocol):
    name: str
    def ingest(self, conversation: dict[str, Any]) -> None: ...
    def retrieve(self, question: str) -> str: ...


class RecallSystem:
    """RE-call adapter: index each dialogue turn, retrieve via the trust layer. Returns an EMPTY
    context string when the trust layer abstains — that is the behaviour under test."""

    name = "recall"

    def __init__(self, dsn: str, embedder_name: str = "fastembed", k: int = 5) -> None:
        from recall.eval.locomo import _make_embedder
        self._dsn = dsn
        self._k = k
        self._embedder = _make_embedder(embedder_name)
        self._embedder_name = embedder_name
        self._tenant: str | None = None

    def ingest(self, conversation: dict[str, Any]) -> None:
        from recall.eval.locomo import index_conversation  # extracted in Step 1
        from recall.store import PgVectorStore
        self._tenant = f"bench-{conversation.get('sample_id')}"
        with PgVectorStore(self._dsn, dim=self._embedder.dim, tenant=self._tenant,
                           table="bench_locomo_chunks") as store:
            index_conversation(store, self._embedder, conversation)

    def retrieve(self, question: str) -> str:
        from recall.store import PgVectorStore
        from recall.trust import trusted_search
        assert self._tenant is not None, "ingest() must run before retrieve()"
        with PgVectorStore(self._dsn, dim=self._embedder.dim, tenant=self._tenant,
                           table="bench_locomo_chunks") as store:
            result = trusted_search(store, self._embedder, question, k=self._k)
            if result.abstained:
                return ""
            return "\n".join(h.chunk.text for h in result.hits)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_bench_systems.py -q`
Expected: PASS

- [ ] **Step 6: Add a skippable integration test** (real Postgres; self-skips without `RECALL_TEST_DSN`)

```python
# add to tests/test_bench_systems.py
import os
import pytest


@pytest.mark.skipif(not os.environ.get("RECALL_TEST_DSN"), reason="needs Postgres")
def test_recall_system_indexes_and_retrieves() -> None:
    from benchmarks.systems import RecallSystem
    conv = {"sample_id": "itest", "conversation": {
        "speaker_a": "Alice", "speaker_b": "Bob",
        "session_1": [{"speaker": "Alice", "text": "The rate limit is 500 rps."}]}}
    # shape must match what index_conversation expects — adapt to the real loader
    sys = RecallSystem(os.environ["RECALL_TEST_DSN"])
    sys.ingest(conv)
    ctx = sys.retrieve("what is the rate limit?")
    assert isinstance(ctx, str)  # content depends on the loader's turn shape; smoke-level only
```

> Implementer note: the exact `conversation` shape comes from `locomo10.json` / the loader — align
> the fixture with it during Step 1. Keep this test smoke-level (types + no crash), not exact recall.

- [ ] **Step 7: Stage locally**

```bash
git add benchmarks/systems.py tests/test_bench_systems.py recall/eval/locomo.py
git commit -m "bench: MemorySystem protocol + RE-call adapter"
```

---

### Task 6: Mem0 system adapter

**Files:**
- Modify: `benchmarks/systems.py`
- Modify: `tests/test_bench_systems.py`

**Interfaces:**
- Consumes: `mem0ai` (lazy), the OpenRouter env
- Produces: `Mem0System(openrouter_key, model, embedder="huggingface", k=5)`; `mem0_config(...)  -> dict` (pure, unit-testable config builder)

- [ ] **Step 1: Write the failing test** (config builder is pure → no mem0 install needed)

```python
# add to tests/test_bench_systems.py
from benchmarks.systems import mem0_config


def test_mem0_config_points_llm_at_openrouter_and_local_embedder() -> None:
    cfg = mem0_config(openrouter_key="sk-x", model="openai/gpt-4o-mini")
    assert cfg["llm"]["provider"] == "openai"
    assert cfg["llm"]["config"]["openai_base_url"] == "https://openrouter.ai/api/v1"
    assert cfg["llm"]["config"]["model"] == "openai/gpt-4o-mini"
    assert cfg["llm"]["config"]["api_key"] == "sk-x"
    assert cfg["embedder"]["provider"] == "huggingface"  # free local, not OpenAI
    assert "bge-small" in cfg["embedder"]["config"]["model"]


def test_mem0_config_default_arm_uses_openai_embeddings() -> None:
    cfg = mem0_config(openrouter_key="sk-x", model="openai/gpt-4o-mini",
                      embedder="openai", openai_key="sk-emb")
    assert cfg["embedder"]["provider"] == "openai"
    assert cfg["embedder"]["config"]["model"] == "text-embedding-3-small"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_bench_systems.py::test_mem0_config_points_llm_at_openrouter_and_local_embedder -q`
Expected: FAIL — `ImportError: cannot import name 'mem0_config'`

- [ ] **Step 3: Write the config builder + adapter**

```python
# append to benchmarks/systems.py
def mem0_config(
    openrouter_key: str,
    model: str,
    embedder: str = "huggingface",
    openai_key: str | None = None,
) -> dict[str, Any]:
    """Build a Mem0 config: LLM via OpenRouter (OpenAI-compatible), embedder local by default.

    embedder="huggingface" -> free local bge-small (the controlled arm, same as RE-call).
    embedder="openai"       -> text-embedding-3-small (the Mem0-default arm; needs openai_key).
    """
    llm = {
        "provider": "openai",
        "config": {
            "model": model,
            "api_key": openrouter_key,
            "openai_base_url": "https://openrouter.ai/api/v1",
        },
    }
    if embedder == "openai":
        emb = {"provider": "openai",
               "config": {"model": "text-embedding-3-small", "api_key": openai_key}}
    else:
        emb = {"provider": "huggingface",
               "config": {"model": "BAAI/bge-small-en-v1.5"}}
    return {"llm": llm, "embedder": emb}


class Mem0System:
    """Mem0 adapter. Feeds the conversation via `add`, retrieves via `search`. LLM on OpenRouter."""

    name = "mem0"

    def __init__(self, openrouter_key: str, model: str, embedder: str = "huggingface",
                 openai_key: str | None = None, k: int = 5) -> None:
        self._config = mem0_config(openrouter_key, model, embedder, openai_key)
        self._k = k
        self._user: str | None = None
        self._mem: Any = None

    def _memory(self) -> Any:
        if self._mem is None:
            from mem0 import Memory
            self._mem = Memory.from_config(self._config)
        return self._mem

    def ingest(self, conversation: dict[str, Any]) -> None:
        self._user = f"bench-{conversation.get('sample_id')}"
        messages = _conversation_to_messages(conversation)  # implementer: map turns -> [{role,content}]
        self._memory().add(messages, user_id=self._user)

    def retrieve(self, question: str) -> str:
        assert self._user is not None, "ingest() must run before retrieve()"
        res = self._memory().search(question, user_id=self._user, limit=self._k)
        results = res["results"] if isinstance(res, dict) else res
        return "\n".join(r["memory"] for r in results)
```

> Implementer note: `_conversation_to_messages` maps LOCOMO turns to Mem0's `[{"role","content"}]`
> shape — write it alongside, mirroring `index_conversation`'s turn walk so BOTH systems ingest the
> SAME turns (fairness). Confirm the `search` return shape against the pinned `mem0ai` version.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_bench_systems.py -q -k mem0_config`
Expected: PASS (2 config tests; adapter itself is covered by the skippable integration test below)

- [ ] **Step 5: Add a skippable integration test** (needs `bench` extra + `OPENROUTER_API_KEY`)

```python
# add to tests/test_bench_systems.py
@pytest.mark.skipif(
    not (os.environ.get("OPENROUTER_API_KEY") and _mem0_installed()),
    reason="needs mem0ai + OPENROUTER_API_KEY",
)
def test_mem0_system_smoke() -> None:
    from benchmarks.systems import Mem0System
    sys = Mem0System(os.environ["OPENROUTER_API_KEY"], model="openai/gpt-4o-mini")
    sys.ingest({"sample_id": "itest", "conversation": {
        "session_1": [{"speaker": "Alice", "text": "The rate limit is 500 rps."}]}})
    assert isinstance(sys.retrieve("rate limit?"), str)
```

Add the helper at the top of the test file:

```python
def _mem0_installed() -> bool:
    import importlib.util
    return importlib.util.find_spec("mem0") is not None
```

- [ ] **Step 6: Stage locally**

```bash
git add benchmarks/systems.py tests/test_bench_systems.py
git commit -m "bench: Mem0 adapter + OpenRouter/local-embedder config builder"
```

---

### Task 7: Run script — wire LOCOMO, both systems, dump results

**Files:**
- Create: `benchmarks/run.py`
- Create: `tests/test_bench_run.py`

**Interfaces:**
- Consumes: everything above + `recall.eval.locomo` loader
- Produces: `run_arm(system, completer, questions) -> tuple[list[Outcome], dict]`; a `main()` CLI with `--conversations N`, `--arm {recall,mem0,mem0-default}`, `--model`, `--out`.

- [ ] **Step 1: Write the failing test** (drives `run_arm` with fakes — no DB, no network)

```python
# tests/test_bench_run.py
from benchmarks.pipeline import Outcome
from benchmarks.run import run_arm


class _Sys:
    name = "fake"
    def ingest(self, c): ...
    def retrieve(self, q): return "" if "penguin" in q else "rate limit is 500 rps"


def test_run_arm_produces_outcomes_and_aggregate() -> None:
    completer = lambda system, user: (
        "YES" if "Correct?" in user else ("NO_ANSWER" if "Memories:\n\n" in user else "500 rps")
    )
    questions = [
        {"question_id": "1", "category": "cat1", "adversarial": False,
         "question": "rps?", "answer": "500"},
        {"question_id": "2", "category": "cat5", "adversarial": True,
         "question": "penguins on mars?", "answer": ""},
    ]
    outcomes, agg = run_arm(_Sys(), completer, questions)
    assert len(outcomes) == 2
    assert all(isinstance(o, Outcome) for o in outcomes)
    assert agg["answerable_accuracy"]["n"] == 1
    assert agg["adversarial_abstention"]["rate"] == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_bench_run.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'benchmarks.run'`

- [ ] **Step 3: Write minimal implementation**

```python
# benchmarks/run.py
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from benchmarks.llm import Completer, OpenRouterLLM
from benchmarks.pipeline import Outcome, aggregate, run_question
from benchmarks.systems import MemorySystem


def run_arm(
    system: MemorySystem, completer: Completer, questions: list[dict[str, Any]]
) -> tuple[list[Outcome], dict[str, Any]]:
    outcomes = [run_question(system.retrieve, completer, q) for q in questions]
    return outcomes, aggregate(outcomes)


def _load(conversations: int | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (conversations, flat question list) from LOCOMO via the existing loader.
    Implementer: reuse recall.eval.locomo's loader + its ANSWERABLE_CATEGORIES/ADVERSARIAL_CATEGORY
    to set each question's `adversarial` flag and `question_id`."""
    raise NotImplementedError  # wired in Step 5


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m benchmarks.run")
    p.add_argument("--arm", choices=["recall", "mem0", "mem0-default"], required=True)
    p.add_argument("--model", default="openai/gpt-4o-mini")
    p.add_argument("--conversations", type=int, default=1)
    p.add_argument("--out", type=Path, default=Path("benchmarks/results"))
    args = p.parse_args(argv)

    key = os.environ["OPENROUTER_API_KEY"]
    llm = OpenRouterLLM(model=args.model, api_key=key)
    completer: Completer = llm.complete

    convs, questions = _load(args.conversations)
    system = _build_system(args.arm, args.model)      # implementer: dsn / mem0 wiring
    for conv in convs:
        system.ingest(conv)

    outcomes, agg = run_arm(system, completer, questions)
    args.out.mkdir(parents=True, exist_ok=True)
    stamp = f"{args.arm}_{args.model.replace('/', '-')}_{len(convs)}conv"
    (args.out / f"{stamp}.json").write_text(
        json.dumps({"arm": args.arm, "model": args.model, "aggregate": agg,
                    "outcomes": [asdict(o) for o in outcomes]}, indent=2), encoding="utf-8")
    print(json.dumps(agg, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_bench_run.py -q`
Expected: PASS (`run_arm` is exercised; `_load`/`_build_system`/`main` are wired in Step 5 and covered by the manual pilot, not unit tests)

- [ ] **Step 5: Wire `_load` and `_build_system` to real LOCOMO + systems**

Read `recall/eval/locomo.py` for the loader and category constants. Implement `_load` to return the sliced conversations and a flat question list, tagging each question with `adversarial = category == ADVERSARIAL_CATEGORY` and a stable `question_id`. Implement `_build_system("recall"|"mem0"|"mem0-default", model)` using `RECALL_TEST_DSN`/`RECALL_DSN` for RE-call and `OPENROUTER_API_KEY` (+ `OPENAI_API_KEY` for `mem0-default`) for Mem0. No new unit test — verified by the pilot in Task 8.

- [ ] **Step 6: ruff + mypy + full offline tests**

Run: `.venv/Scripts/python.exe -m ruff check benchmarks tests && .venv/Scripts/python.exe -m mypy benchmarks && .venv/Scripts/python.exe -m pytest tests/test_bench_*.py -q`
Expected: clean; all offline tests pass

- [ ] **Step 7: Stage locally**

```bash
git add benchmarks/run.py tests/test_bench_run.py
git commit -m "bench: run script wiring LOCOMO + both systems + results dump"
```

---

### Task 8: Pilot run + methodology check (manual, spends a few $)

**Files:**
- Create: `benchmarks/results/` (generated)
- Create: `benchmarks/PILOT-NOTES.md`

This task has no unit test — it is the human-in-the-loop validation the spec's Phase 0 requires.

- [ ] **Step 1: Preflight** — `export OPENROUTER_API_KEY=...`; confirm Postgres up (`RECALL_TEST_DSN`); `uv pip install -e ".[bench,fastembed]"`.
- [ ] **Step 2: Run RE-call on 1 conversation** — `.venv/Scripts/python.exe -m benchmarks.run --arm recall --conversations 1 --model <pinned-openai-model>`
- [ ] **Step 3: Run Mem0-normalized on the SAME conversation** — `--arm mem0 --conversations 1 --model <same-model>`. Note the actual $ spent (Mem0's per-turn extraction dominates).
- [ ] **Step 4: Eyeball the raw dump** — open both `benchmarks/results/*.json`; manually read ~10 answerable + ~10 adversarial records. Check: does the judge's YES/NO look right? Do abstentions fire where they should? Is the SAME turn set fed to both systems?
- [ ] **Step 5: Record findings** in `PILOT-NOTES.md` — measured $/conversation, judge sanity, any prompt/judge fixes needed. Decide go/no-go for the full 10-conversation run.
- [ ] **Step 6: Stage locally** — `git add benchmarks/PILOT-NOTES.md && git commit -m "bench: pilot notes + go/no-go"` (results/ stays gitignored).

---

## Self-Review

**Spec coverage:** §1 goal → Tasks 1,3,4 (both columns) ✓. §2 arms → Task 6 config (`huggingface`/`openai` embedder) + Task 7 `--arm` ✓. §3 architecture (interface + shared pipeline) → Tasks 4,5,6 ✓. §4 metrics (both columns + Wilson + per-category) → Task 4 `aggregate` ✓. §5 fairness (same generator/judge, own retrieval defaults, same embedder, raw dump) → Tasks 3,5,6,7 ✓. §6 phases (pilot first) → Task 8 ✓. §7 lives in `benchmarks/`, `bench` extra never in CI → Task 1 ✓. §8 keys → Tasks 2,6,7 env wiring ✓. §9 risks (cost, judge variance temp=0, mem0 pin) → Tasks 2,8 ✓.

**Placeholder scan:** The two `NotImplementedError`/reuse points (`_load`, `_build_system`, `index_conversation`, `_conversation_to_messages`) are explicitly deferred to a numbered step that reads the existing loader — not vague "TODO"s, but "read this file, reuse this routine" instructions, because their exact shape depends on `locomo10.json` which the implementer must inspect. Acceptable and marked.

**Type consistency:** `Completer = Callable[[str,str],str]` used identically in Tasks 2,3,4,7. `Outcome` fields defined in Task 4 are consumed unchanged in Task 7. `MemorySystem` (Task 5) implemented by `RecallSystem`/`Mem0System` and consumed by `run_arm` (Task 7) — signatures match. `_rate` reused from `recall.eval.locomo` (Task 4), noted for verification.

**Open verification (flagged, not blocking):** exact `mem0ai` `search()` return shape and the LOCOMO turn shape — both resolved by reading pinned deps during Tasks 5/6, which is why those adapters carry skippable integration tests rather than asserting exact content offline.
