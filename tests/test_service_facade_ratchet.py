"""`recall_mcp/service.py` stays a compatibility facade: it may lose names, never gain them.

Invariant: the names `recall_mcp/service.py` binds itself at module level (functions, classes,
constants, aliases, including names bound by unpacking; imports excluded) are a subset of
`ALLOWED`; every top-level assignment binds plain names to a value that calls nothing, apart from
the statements in `ALLOWED_STATEMENTS`, which also hold its only other top-level statement, the
candidate-files provider registration; and the file is at most `MAX_LINES` long, so code cannot
regrow inside the functions it may keep.

Failure mode caught: the facade was cut to 635 lines on 2026-09-07 and was back at 5,290 lines
five days later, because the seam tests pin who owns each EXISTING name and nothing stopped a new
feature from adding its implementation here. Put new code in the module that owns its domain and,
if a name must stay importable from `recall_mcp.service`, re-export it with an import.

When a name is removed from the facade, delete it from `ALLOWED` too, so it cannot come back, and
lower `MAX_LINES` to the new length.

Red proof, 2026-10-02, against this tree with one mutation each, on a Linux test host:

* `test_the_facade_defines_no_new_names`, mutation: `def _new_feature(): return 1` appended to
  `recall_mcp/service.py`: failed at its assertion, `['_new_feature'] not allowed: ...`.
* `test_the_facade_runs_no_new_top_level_code`, mutation: `print("loaded")` appended: failed at
  its assertion, `["print('loaded')"] == []`.

Red proof, 2026-10-27, for the CCA fixes to PR 857, the same way, each mutation appended to
`recall_mcp/service.py` unless it says otherwise:

* `test_the_facade_defines_no_new_names`, mutation: `_new_feature, _unused = (lambda: 1), None`:
  failed at its assertion, `['_new_feature', '_unused'] not allowed: ...` (it passed before).
* `test_the_facade_runs_no_new_top_level_code`, mutations: `_re._execute_reasoning_query = print`
  after `import recall_mcp.reasoning_engine as _re`, and `HASHING_DIM = print('loaded') or 64`:
  each failed at its assertion with the statement listed (both passed before).
* `test_the_facade_stays_small`, mutation: a 20-line nested helper added inside `search_memory`:
  failed at its assertion, `826 lines, more than MAX_LINES = 806` (nothing failed before).
"""

from __future__ import annotations

import ast
from pathlib import Path

import recall_mcp

SERVICE = Path(recall_mcp.__file__).with_name("service.py")

ALLOWED = frozenset(
    {
        "HASHING_DIM",
        "MAX_QUERY_CONSTRUCTION_GRAPH_NODES",
        "MAX_QUERY_CONSTRUCTION_PROMPT_CHARS",
        "REASONING_BLOCKED_NOTE",
        "REASONING_SUPERSEDED_NOTE",
        "STALE_INDEX_NOTE",
        "UNCALIBRATED_NOTE",
        "_GRAPH_PROJECTIONS",
        "_GRAPH_PROJECTION_CACHE_MAX",
        "_GRAPH_PROJECTION_INFLIGHT",
        "_RERANK_FALSE",
        "_RERANK_TRUE",
        "_admission",
        "_advice_suffixes",
        "_build_reranker",
        "_cost_surface",
        "_evidence_advice",
        "_evidence_card_model",
        "_evidence_item_model",
        "_log",
        "_new_reranker",
        "_positive_env",
        "_query_construction_graph",
        "_remote_model_code_enabled",
        "_require_remote_model_code_enabled",
        "_reset_reranker_cache",
        "_retrieve_trusted",
        "_search_advice",
        "_search_hit_model",
        "_trusted_evidence_item_model",
        "_validate_quality_reranker_config",
        "evidence_memory",
        "graph_first_retrieval",
        "query_construction_challenge",
        "reasoning_audit",
        "reasoning_query",
        "resolve_reranker",
        "search_memory",
        "tenant_scopes",
    }
)

ALLOWED_STATEMENTS = frozenset(
    {
        "_set_candidate_files_provider(lambda: candidate_files)",
        "_log = get_logger('mcp.service')",
        "_RERANK_TRUE = frozenset({'1', 'true', 'yes', 'on'})",
        "_RERANK_FALSE = frozenset({'', '0', 'false', 'no', 'off'})",
    }
)

MAX_LINES = 806


def _module() -> ast.Module:
    return ast.parse(SERVICE.read_text(encoding="utf-8"))


def _bound_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names.update(
                n.id for t in targets for n in ast.walk(t)
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
            )
    return names


def test_the_facade_defines_no_new_names() -> None:
    added = _bound_names(_module()) - ALLOWED
    assert not added, (
        f"{sorted(added)} not allowed: recall_mcp/service.py is a compatibility facade. Put the "
        "code in the module that owns it and re-export the name here with an import."
    )


def test_the_facade_runs_no_new_top_level_code() -> None:
    extra = []
    for node in _module().body:
        if isinstance(
            node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            continue
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            plain = all(isinstance(t, ast.Name) for t in targets)
            calls = node.value is not None and any(isinstance(n, ast.Call) for n in ast.walk(node.value))
            if plain and not calls:
                continue  # a re-export alias or a constant
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue  # a docstring or a bare string
        statement = ast.unparse(node)
        if statement not in ALLOWED_STATEMENTS:
            extra.append(statement)
    assert extra == []


def test_the_facade_stays_small() -> None:
    lines = len(SERVICE.read_text(encoding="utf-8").splitlines())
    assert lines <= MAX_LINES, (
        f"recall_mcp/service.py is {lines} lines, more than MAX_LINES = {MAX_LINES}: the facade "
        "regrows inside the functions it keeps. Move the code to the module that owns it."
    )
