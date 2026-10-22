# Audit memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement your session's tasks. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve CCA's bug, false-positive and lesson memory from RE-call as anchored records that
round-trip without loss, so the L1 auditor brief gains anchored prior bugs, lessons and regression
edges, and repeat false positives on unchanged code are suppressed visibly.

**Architecture:** One record format (`recall_record` v1, a one-line JSON front matter key) is the
contract between the two systems. CCA computes anchors and writes records; RE-call stores them as
chunk metadata and answers exact lookups with no embedding; CCA rebuilds rows and briefs from those
answers. Code tenants carry the same anchor on code chunks, so records join to current code by key.

**Tech Stack:** Python 3.12, `ast`, JSON Schema, PostgreSQL JSONB and the lite SQLite store
(RE-call), the `cca_checks` package and its plugin prompts (CCA), FastMCP.

**Spec:** `docs/AUDIT_MEMORY_DESIGN.md`. Every session reads it in full before its own section.

## How the work is divided

Seven sessions plus the orchestrator. Each session owns one branch in one repository, touches only
the files its section lists, opens a pull request, does not merge, and reports to the orchestrator.
Sessions depend on each other only through the **contracts** Session 0 lands first, never through
each other's unmerged code.

| Session | Repository | Starts when | Owns |
|---|---|---|---|
| 0 Contracts | RE-call | plan approved (orchestrator does it) | schema, anchor reference, vectors, golden memos |
| 1 Records API | RE-call | Session 0 merged | parse, store lookup, `recall_records`, `caused_by` relation |
| 2 CCA record layer | cca-audit | Session 0 merged | anchors, record model, render and parse, `caused_by`, round-trip check |
| 3 Code anchors | RE-call | Session 0 merged | AST code chunker writing `code_anchor` |
| 4 CCA pipeline | cca-audit | Session 2 merged | FP rows, L6 acceptance, FP write path, matcher, brief |
| 5 Lessons | cca-audit | Session 4 merged | distillation, acceptance, LESSONS block |
| 6 Deploy and ops | operator tooling | Sessions 1 and 3 merged, and the owner says the AML runs are over | serving, tenants, worker check |
| 7 Recurrence experiment | RE-call research | Session 6 done | pre-registration, then the experiment |
| Final | orchestrator | Sessions 4, 5 and 6 done | live round-trip verification, integration report |

Wave 1 runs Sessions 1, 2 and 3 in parallel. Sessions 1 and 3 are both in RE-call and both edit
`recall/generations.py` (different constants); whichever merges second rebases and regenerates
`docs/ARCHITECTURE_MAP_GENERATED.md`.

**Every session's report to the orchestrator** carries: the PR link; for each new behaviour test
its node ID, the mutation used for red proof and the failing assertion; the full test command and
its pass, fail and skip counts; any departure from this plan and why.

## Global Constraints

- RE-call is public: no host names, addresses, account names or memo content in code, tests,
  fixtures or docs. `python scripts/check_public_tree.py --staged` with the deny list set must be
  clean before every RE-call commit.
- Every new behaviour test has red proof: fails in its intended assertion against a plausible
  mutation of the production line, then passes. Record node ID, mutation and failure reason.
- No test calls a paid API unless `RECALL_RUN_PAID_TESTS=1`; offline tests replace the transport.
- Test suites for both repositories run on the deployment host, bounded so they cannot compete
  with the official AML run (owner instruction, 2026-10-21; that run holds 12 GB there):
  one test run at a time host-wide (the shared test-run lock), a user scope with
  `MemoryMax=8G`, `MemorySwapMax=0`, `CPUQuota=400%`, `nice -n 15`, at most 4 xdist workers
  (3 for RE-call, its standing rule), and a test database container capped at 2 GB and one CPU.
  Never the AML service, its API, the serving checkout or anything that takes the embedding lock.
  Focused local runs of named test files only where the repository's guard allows them.
- RE-call: `python -m ruff check .` and `python -m mypy` clean; never `ruff format`; stage by
  pathspec; signed commits; squash merges.
- Canonical record JSON is `json.dumps(obj, ensure_ascii=False, sort_keys=True,
  separators=(",", ":"))`, on one line.
  **Amended 2026-10-22 (orchestrator, from cca-audit #87):** in that line U+2028, U+2029 and
  U+0085 are written as the escapes ` `, ` ` and `\u0085`, because `str.splitlines()`
  breaks on all three and `json.dumps` with `ensure_ascii=False` does not escape them, so a
  `splitlines()` reader would split one row in two. The JSON value is unchanged: RE-call's parser
  decodes the escapes and returns the record equal to the one authored
  (`tests/test_records_code_tenant.py::test_escaped_line_separators_round_trip`).
- `recall_record` is returned exactly as authored. No component renames, adds, defaults, normalises
  or drops a field.
- Serving and tenant changes on the deployment host wait for Session 6.

## Contracts (Session 0 lands these; everyone else consumes them)

**C1. `recall/schemas/recall_record.v1.json`**, a JSON Schema (draft 2020-12), shipped as package
data. Every object has `v` (const 1), `kind`, `id`, `project`, `repo`, `date` (ISO date). Per kind:

| kind | required beyond the common fields |
|---|---|
| `bug` | every `BugRow` field (`bug_id`, `tier`, `files`, `finding_id`, `category`, `severity`, `auditor`, `fix_commit`, `run_id`, `evidence`, `sources`, `summary`, `supersedes`, `ordinal`), `anchors`, `caused_by` (list of ids, may be empty) |
| `false_positive` | `run_id`, `finding_id`, `category`, `severity`, `priority`, `high_stakes`, `auditor`, `claim`, `fp_reason` (const `not_real`), `evidence_refs` (list of `file:line`), `verify_source`, `anchors` (at least one, every `code_sha` non-null) |
| `lesson` | `rule`, `pattern`, `anchors` (every `code_sha` null), `members` (at least 3 ids), `accepted_by`, `accepted_at` |
| `false_positive_retraction`, `lesson_retraction` | `target_id`, `reason`, `retracted_by` |

`additionalProperties: false` everywhere. Ids: a bug's `id` equals its `bug_id`; a false positive's
is `<project>:fp:<run_id>:<finding_id>`; a lesson's is `<project>:lesson:<first 12 hex of sha256 of
the sorted member ids joined by newlines>`; a retraction's is `<target_id>~retract`.

**C2. `recall/anchors.py`**, the reference resolver:

- `resolve_symbols(source: str) -> list[SymbolSpan]`, where `SymbolSpan` is a frozen dataclass
  `(symbol: str, start_line: int, end_line: int)` for every function and class `ast` reports,
  dotted by nesting: a method is `Pool._lease` and a nested function is `outer.inner` (never
  Python's `outer.<locals>.inner`). Decorators belong to the span. A dotted name seen a second time in one file
  (a property and its setter) becomes `name#2`, then `#3`, in source order.
- `enclosing_symbol(source: str, line: int) -> str | None`: the innermost span containing `line`.
- `code_sha(text: str) -> str`: sha256 hex of the text after CRLF and CR become LF and trailing
  whitespace is stripped from every line.
- `symbol_sha(source: str, symbol: str | None) -> str`: `code_sha` of the symbol's span, or of the
  whole file when `symbol` is None.

**C3. `tests/fixtures/audit_memory/anchor_vectors.json`**: a list of `{name, source, expect:
[{symbol, start_line, end_line, sha}], file_sha, enclosing: [{line, symbol}]}`. Cases at least:
plain function, class with methods, nested function, decorated function, property with setter,
async def, module with no symbols, the same source with CRLF line endings (same hashes), trailing
whitespace only change (same hash), body change (different hash).

**C4. `tests/fixtures/audit_memory/memos/`**: golden memo files, the exact bytes CCA must render.
Front matter order: `name`, `description`, `valid_from`, `supersedes` (only when set),
`recall_record`, then the existing `metadata` block with a fixed `modified`. Files at least: a
plain bug; a bug with `caused_by` and `supersedes`; a false positive with two anchors; a lesson; a
false-positive retraction; a lesson retraction; a bug whose summary holds Unicode, `---`, `: ` and
both quote kinds; a bug whose record exceeds 10 KB. A sibling `expected_records.json` maps each
file name to its parsed `recall_record`.

**C5. `recall_records` output envelope** (MCP tool and CLI `--json`):

```json
{"tenant_id": "...", "generation_id": "...", "truncated": false,
 "records": [{"id": "...", "kind": "bug", "recall_record": {}, "name": "...", "description": "...",
   "source": "...", "chunk_id": "...", "chunk_ids": ["..."], "valid_from": "...", "supersedes": null,
   "superseded_by": null, "valid_until": null, "caused": [],
   "anchor_matches": [{"anchor": {}, "match": "exact|symbol_changed|file_only|none"}],
   "anchor_match": "exact|symbol_changed|file_only|none"}]}
```

`anchor_match` is the best of `anchor_matches` (exact, then symbol_changed, then file_only); a
record with no anchor matching at least `file_only` is not returned for an anchor query. For an
`ids` or `--all` query every `match` is `none`. In a code tenant each code chunk is returned as a
record with `kind: "code"` and `recall_record` set to its `code_anchor` object.

**Amended 2026-10-22 (orchestrator, from Session 3's report on PR 976):** several code chunks
share one `code_anchor` (a split symbol, and every module-level chunk of a file), so a code tenant
returns ONE record per distinct anchor `(repo, file, symbol, code_sha)`, with `chunk_id` the
lowest-ordinal chunk and `chunk_ids` every chunk carrying that anchor. A memo's record carries
`chunk_ids` too: every chunk of the memo. A code anchor's `repo` may be null (a build with no
`--project`), and a null `repo` never matches a `repo` query.

---

## Session 0: Contracts (orchestrator)

Branch `feat/audit-memory-contracts` in RE-call, off `origin/master`.

### Task 0.1: Anchor reference and vectors

**Files:** Create `recall/anchors.py`, `tests/fixtures/audit_memory/anchor_vectors.json`,
`tests/test_anchors.py`.

**Interfaces:** Produces C2 and C3.

- [ ] **Step 1:** Write `tests/test_anchors.py::test_vectors_reproduce` (every vector's spans,
      shas, `file_sha` and `enclosing` equal `resolve_symbols`, `symbol_sha`, `enclosing_symbol`),
      `test_crlf_and_trailing_whitespace_do_not_change_sha`, `test_property_setter_gets_suffix`.
- [ ] **Step 2:** Run them; expected FAIL (module missing is not red proof, so red proof comes in
      Step 5).
- [ ] **Step 3:** Implement `recall/anchors.py` per C2 with `ast` only; generate the vectors file
      from hand-written sources, checking each expected value by reading it, not by trusting the
      implementation.
- [ ] **Step 4:** Run; expected PASS.
- [ ] **Step 5:** Red proof: mutate `code_sha` to skip CR normalisation, and `resolve_symbols` to
      drop the `#2` suffix; each named test fails on its assertion; restore.
- [ ] **Step 6:** Commit.

### Task 0.2: Schema and golden memos

**Files:** Create `recall/schemas/recall_record.v1.json`, `tests/fixtures/audit_memory/memos/*.md`,
`tests/fixtures/audit_memory/expected_records.json`, `tests/test_recall_record_schema.py`; modify
`pyproject.toml` (package data, and the sdist allowlist test if it pins the file list).

**Interfaces:** Produces C1 and C4.

- [ ] **Step 1:** Write `test_every_expected_record_validates`, `test_unknown_field_is_invalid`,
      `test_fp_with_null_code_sha_is_invalid`, `test_lesson_with_two_members_is_invalid`,
      `test_schema_ships_as_package_data` (loads it through `importlib.resources`).
- [ ] **Step 2:** Run; expected FAIL.
- [ ] **Step 3:** Write the schema per C1 and the golden memos per C4.
- [ ] **Step 4:** Run; expected PASS. Red proof: drop `additionalProperties: false`, then the
      `minItems` on lesson members; the matching tests fail; restore.
- [ ] **Step 5:** Commit, open the PR, and merge it with the owner's approval before Wave 1.

---

## Session 1: Records API (RE-call)

Branch `feat/audit-memory-records`. Read C1, C2 and C5.

### Task 1.1: Parse and store `recall_record`

**Files:** Modify `recall/frontmatter.py`, `recall/generations.py` (`_METADATA_RULE_VERSION`),
the lite store's metadata path; test `tests/test_frontmatter_recall_record.py`.

**Interfaces:** Produces chunk `metadata["recall_record"]` (the parsed object, unchanged) on every
chunk of a memo that declares a valid one.

- [ ] Tests: `test_golden_memos_parse_to_expected_records` (all C4 files, equality with
      `expected_records.json`); `test_invalid_record_warns_and_indexes_without_it` (warning names
      the file and the schema error); `test_unknown_version_refused_with_warning`;
      `test_metadata_rule_version_bumped` (a chunk set stored under the previous version is not
      reused).
- [ ] Run, FAIL; implement; run, PASS; red proof by mutating the parser to drop one field, and by
      reverting the version bump; commit.

### Task 1.2: The exact lookup

**Files:** Create `recall/records.py` (`lookup_records`), modify `recall/store.py` and the lite
store (one query each), `recall_mcp/server.py` (tool `recall_records`), the CLI (`recall records`,
classified for lite), `docs/API.md`; tests `tests/test_records_lookup.py`,
`tests/test_records_mcp.py`, `tests/test_records_cli.py`.

**Interfaces:**
- `lookup_records(store, *, repo: str | None, anchors: list[dict] | None, ids: list[str] | None,
  kinds: list[str] | None, all_records: bool = False, limit: int = 500) -> dict` returning C5.
- MCP `recall_records(repo, anchors, ids, kinds)`; at most 200 anchors and 500 records, refused
  past that with a message, never truncated silently (`truncated` is true only for `--all` paging).

- [ ] Tests, against a lite store indexed from the C4 memos: `test_exact_symbol_changed_file_only`
      (one record, three anchor queries, three matches); `test_multi_anchor_reports_each`;
      `test_round_trip_returns_record_unchanged` (every C4 record equals its `expected_records.json`
      entry, plus `name`, `description`, `valid_from`, `supersedes`);
      `test_superseded_and_retracted_are_marked_not_hidden`; `test_caused_reverse_edge`;
      `test_ids_and_all`; `test_over_200_anchors_refused`; `test_reads_active_generation_only`;
      the same lookup over PostgreSQL in the DB suite.
- [ ] Red proof: mutate the comparison so a hash difference counts as exact; mutate the output to
      drop `supersedes`; each named test fails; restore. Commit.

### Task 1.3: `caused_by` in `recall_related`

**Files:** Modify `recall/related.py` (`RelatedRelation` gains `"caused_by"`), its MCP wrapper;
test `tests/test_related_caused_by.py`.

- [ ] Tests: `test_caused_by_both_directions` from the C4 bug fixture; `test_unknown_relation_still_refused`.
- [ ] Red proof by mutating the direction; commit; open the PR; report.

---

## Session 2: CCA record layer (cca-audit)

Branch `feat/audit-memory-records` off `origin/master` (the local `master` is stale). Copy C1, C3
and C4 from the RE-call commit that merged Session 0 into `tests/fixtures/audit_memory/`, and pin
that commit in a constant.

### Task 2.1: Anchors

**Files:** Create `cca_checks/bugs/anchors.py`; test `tests/test_bug_anchors.py`.

**Interfaces:** `anchor_for(repo_root: Path, file: str, line: int) -> Anchor` and
`anchors_for_findings(repo_root: Path, findings: list[dict]) -> list[Anchor]`, where `Anchor` is a
frozen dataclass `(file, symbol, code_sha)` with `to_json() -> dict`. Python files use the C2
algorithm; other files anchor at file level.

- [ ] Tests: `test_vectors_reproduce` (C3, byte for byte); `test_crlf_checkout_same_sha`;
      `test_non_python_file_anchors_at_file_level`; `test_paths_are_repo_relative_forward_slash`.
- [ ] Red proof as in Task 0.1; commit.

### Task 2.2: Record model, render and parse back

**Files:** Modify `cca_checks/bugs/model.py` (`BugRow` gains `anchors`, `caused_by`, `kind`);
create `cca_checks/bugs/records.py` (`FalsePositiveRow`, `LessonRow`, `RetractionRow`,
`to_record(row) -> dict`, `from_record(record: dict) -> Row`, `render_memo(row, *, modified: str)
-> str`, `parse_memo(text: str) -> Row`); add `bugs rerender --all` to the CLI; tests
`tests/test_bug_records_roundtrip.py`.

**Interfaces:** `render_memo` produces C4 bytes exactly; `from_record(to_record(r)) == r` for
every kind.

- [ ] Tests: `test_render_matches_golden` (every C4 file, byte equality);
      `test_row_record_row_identity` (every kind and every hard case);
      `test_schema_copy_matches_pinned_recall_commit`; `test_supersedes_is_written`;
      `test_rerender_rewrites_every_memo_from_the_ledger` (ledger is the system of record; this is
      the one sanctioned overwrite of memos, and it changes no row).
- [ ] Red proof: mutate `to_record` to omit `ordinal`; mutate `render_memo` to emit
      `ensure_ascii=True`; tests fail; restore. Commit.

### Task 2.3: `caused_by` from blame

**Files:** Create `cca_checks/bugs/caused_by.py`; modify `forward.py` to carry it from the run
record; add `bugs backfill-caused-by --repo-root PATH`; test `tests/test_caused_by.py`.

**Interfaces:** `caused_by_for_fix(repo_root: Path, base_commit: str, fix_diff: str, ledger:
list[BugRow]) -> list[str]`.

- [ ] Tests against a temporary git repository built in the test: fix B modifies lines that fix A
      introduced, so B's `caused_by == [A.id]`; a line introduced by a non-CCA commit yields `[]`;
      another repo's row with the same sha does not match.
- [ ] Red proof by blaming the fix commit instead of the base; commit.

### Task 2.4: Round-trip check command

**Files:** Add `bugs roundtrip-check --dump FILE` (FILE is `recall records --all --json` output);
test `tests/test_roundtrip_check.py`.

**Interfaces:** exits 0 only when every ledger row has a record equal to `to_record(row)` and the
dump has no record without a row; otherwise prints `missing`, `changed` (with field names) and
`orphan` lists and exits 1.

- [ ] Tests: identical dump exits 0; one dropped field reports `changed: <id>: <field>`; a missing
      record; an orphan; a dump of a pre-change memo (no `recall_record`) counts as missing.
- [ ] Red proof by mutating the comparison to ignore nested fields; commit; open the PR; report.

---

## Session 3: Code anchors (RE-call)

Branch `feat/audit-memory-code-anchors`. Read C2 and C5.

### Task 3.1: AST chunking with `code_anchor`

**Files:** Modify `recall/index.py` (the `code` chunker uses `recall/anchors.py` for `.py` files),
`recall/generations.py` (chunker version in the pipeline fingerprint); test
`tests/test_code_anchor_chunks.py`.

**Interfaces:** each Python code chunk carries `metadata["code_anchor"] = {"repo", "file",
"symbol", "code_sha"}`, `repo` from the build's project identity. A symbol larger than the chunk
limit splits into several chunks that all carry the same anchor. Non-Python files are unchanged.

- [ ] Tests: `test_each_symbol_one_anchor`; `test_oversized_symbol_split_keeps_anchor`;
      `test_anchor_sha_equals_reference_symbol_sha`; `test_non_python_has_no_anchor`;
      `test_pipeline_fingerprint_changes`; and, with Session 1 merged, `recall_records` over a code
      tenant returns `kind: "code"` records (that test lands in whichever PR merges second).
- [ ] Red proof by mutating the span end line; commit; open the PR; report.

---

## Session 4: CCA pipeline (cca-audit)

Branch `feat/audit-memory-pipeline`, off `origin/master` after Session 2 merged. Builds against C5
using saved response fixtures; the live server arrives with Session 6.

### Task 4.1: Disposition rows and L6 acceptance

**Files:** Modify `cca_checks/plugin/commands/audit-fix.md` (Step 2.5b row fields; Step 6 L6
hand-off and the per-FP `ACCEPT|REJECT` table), the Codex mirror
`cca_checks/plugin/codex/cca-audit/SKILL.md`, the `pipeline record L2.5` and `L6` validators; tests
in the existing pipeline test modules.

- [ ] FP rows gain `file`, `line`, `symbol`, `code_sha`, `claim`, `evidence_refs`, `fp_reason`
      (`not_real|out_of_scope|settled`). `pipeline record L6` refuses an APPROVED verdict whose
      run has FP rows without an acceptance entry each. A REJECT routes the finding to
      `UNCERTAIN → human` in the report.
- [ ] Tests: the validator refuses a missing acceptance; refuses an unknown `fp_reason`; a REJECT
      appears under the disputed heading. Red proof per validator; commit.

### Task 4.2: FP write path at close

**Files:** Modify `cca_checks/bugs/pending.py` (no early return when L4 is empty if accepted FPs
exist), `pipeline_state.py` close hook; create `cca_checks/bugs/fp_records.py`; tests.

- [ ] Only `fp_reason: not_real`, `ACCEPT`, in a run whose final L6 verdict is APPROVED, with
      anchors whose `code_sha` is non-null, becomes a `FalsePositiveRow`; written at close, not at
      merge. Mitigation evidence locations become extra anchors.
- [ ] Tests, one per exclusion (out_of_scope, settled, REJECT, run REVISE, unresolvable anchor)
      plus an all-FP run that is captured. Red proof by removing each guard; commit.

### Task 4.3: Mechanical PRIOR_FP matcher

**Files:** Create `cca_checks/bugs/match_fp.py` and `bugs match-fp --findings L2.json --records
FILE`; modify audit-fix.md to add the step between L2 and L2.5 and the `PRIOR_FP` verdict
everywhere verdicts are enumerated (disposition ledger, tallies, report headings); tests.

**Interfaces:** `match(findings: list[dict], response: dict) -> list[Match]` where `Match` holds
`finding_id`, `record_id`, `action` (`suppress|attach`).

- [ ] Suppress only when the record is current, every anchor is `exact`, `category` is equal, the
      finding is not high stakes and its priority is not higher than the record's; otherwise
      `attach` (to fp-check). `CCA_PRIOR_FP=off` disables suppression.
- [ ] Tests: one per condition flipping suppress to attach; a renamed file never suppresses; a
      retracted record never suppresses; the switch. Red proof by deleting each condition; commit.

### Task 4.4: The brief

**Files:** Modify audit-fix.md Step 1 and the Codex mirror (drop the three similar-fix queries and
the leave-out sentence; call `recall_records` with `bugs anchors --diff` output; save the
response), `cca_checks/bugs/prior.py` (`--records FILE` replaces the ledger join and `--similar`),
the run record (block counts by `anchor_match`); tests.

- [ ] The block lists `exact` and `symbol_changed` before `file_only`, labels each, marks a bug
      whose fix later caused a regression (`caused` non-empty), and never instructs leaving
      anything out. Retraction of an FP record: `bugs retract-fp <id> --reason TEXT`.
- [ ] **Code join.** When the project has a code tenant, Step 1 also calls that tenant's
      `recall_records` with the records' anchors and saves it; `prior.py --code FILE` labels each
      record `unchanged` (a code chunk with the same file, symbol and sha), `changed` (same file
      and symbol, other sha) or `gone` (no chunk with that file and symbol). Without a code tenant
      no label is shown, never a guessed one.
- [ ] Tests: ordering; regression mark; the old sentence is absent from the rendered prompt; block
      counts land in the run record; retraction writes a valid `false_positive_retraction`; the
      three code labels and the no-tenant case. Red proof per behaviour; commit; open the PR;
      report.

---

## Session 5: Lessons (cca-audit)

Branch `feat/audit-memory-lessons`, after Session 4 merged.

### Task 5.1: Distillation proposals

**Files:** Create `cca_checks/bugs/distill.py`, `bugs distill [--repo R]`, proposals under
`<bugs dir>/lessons/proposed/` (outside the ledger); tests with the model transport replaced.

**Interfaces:** `candidate_groups(rows: list[BugRow]) -> list[Group]` (repo, category, file or
symbol, following `caused_by`; three or more members); `draft(group, client) -> Proposal`.

- [ ] Tests: groups of fewer than three are dropped; a `caused_by` chain joins a group; a draft
      stores rule, pattern, members and the model id; nothing is written to the ledger or memos.
      A live model call is skipped unless `RECALL_RUN_PAID_TESTS=1` and says what it spends.
- [ ] Red proof; commit.

### Task 5.2: Acceptance, retraction, brief block

**Files:** `bugs lesson accept <proposal> --by NAME`, `bugs lesson retract <id> --reason TEXT`;
audit-fix.md Step 1 and the Codex mirror (LESSONS block); `prior.py`; tests.

- [ ] Accept writes a `LessonRow` and memo valid against C1; retract writes `lesson_retraction`.
      The block shows at most three current lessons for the changed files, each with member
      citations, worded as patterns to check, with no "do not flag" wording.
- [ ] Tests for each; red proof; commit; open the PR; report.

---

## Session 6: Deploy and ops (operator tooling)

Starts only when Sessions 1 and 3 are merged **and the owner says the AML official runs are over**.
Works in the operator's private tooling repository and on the deployment host; its brief, not this
public plan, names hosts and paths.

- [ ] Sync the serving checkout to master through the normal gated sync (handshake and parity).
- [ ] Rebuild the `cca` tenant after `bugs rerender --all` ships every memo with `recall_record`;
      confirm the metadata rule version forced a re-parse.
- [ ] Switch the RE-call code tenant to the anchored chunker; add a code tenant for cca-audit and
      its MCP registration.
- [ ] The refresh worker runs `recall records --all --json` and `bugs roundtrip-check` after every
      rebuild and fails the tick on any difference.
- [ ] Report: generation ids, counts, and the first round-trip result.

## Session 7: Recurrence experiment

After Session 6. Write and commit a pre-registration in the private research archive first
(question, prediction, falsifier, arms), then build the query from a bug record's fixed code
against the code tenant. Nothing reaches a brief from this session.

## Final: orchestrator

- [ ] Run the live round trip: every ledger row against RE-call's dump, zero `missing`, `changed`
      and `orphan`; one record of each kind fetched by `recall_records` and compared by hand.
- [ ] One DIFF-mode audit run end to end on a real change: anchors computed, records fetched,
      block rendered with counts, matcher run, records written at close.
- [ ] Report to the owner: what shipped, what is measured, what waits for the measurement trigger.

## Review Focus

- **A CRLF checkout** (this workstation): anchors computed on Windows must equal anchors computed
  on Linux. Covered by C3's CRLF vector, Task 2.1 and Task 3.1.
- **A malformed `recall_record` line** (truncated JSON): indexed without it, warned, and reported
  as `missing` by the round-trip check rather than passing. Task 1.1 and Task 2.4.
- **A renamed or moved file**: old-path anchors never suppress and the brief shows no stale match.
  Task 4.3.
- **A property and its setter**, or two methods of one name in one file: distinct anchors through
  the `#2` suffix. C3 and Task 2.1.
- **The 691 memos written before this change**: they round-trip only after `bugs rerender --all`;
  until then the check counts them as missing, never as passing. Task 2.2 and Task 2.4.
