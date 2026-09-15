# Guarded spare slot shadow implementation

Measured 2026-09-14.

## Verdict

`READY_FOR_FRESH_VALIDATION`

The production shaped shadow now starts from the unchanged alpha `0.08` selection, fills only
unused positions under the frozen dual leg and lexical dominant guards, and exposes only private
hash and aggregate diagnostics. The default policy remains `alpha008`, and the guarded policy is
available only inside the existing sampled shadow mode.

This is an implementation result, not a retrieval quality result. The 72 row trace was already
inspected while deriving the rule, so it establishes parity only.

## Frozen trace parity

The production helper matched the immutable replay on all 72 rows. Every base prefix was
preserved. The candidate differed on two queries and appended five items. The generated artifact
SHA256 was
`2EC1E795788D0613F7FA90664DC7298A9DED9D19B7D0CEE99AFA1F13DAAA387D`, identical to the earlier
frozen replay artifact.

Reproduce from the repository root:

```powershell
$guardedReplay = Join-Path $env:TEMP 'recall-guarded-spare-slot-parity-20260914.json'
python -u scripts/replay_source_admission_spare_slots.py --trace docs/results/2026-09-13-live-source-admission-trace-capture.json --artifact docs/results/2026-09-13-source-conditioning-model.json --output $guardedReplay
Get-FileHash -Algorithm SHA256 $guardedReplay
```

## Service boundary proof

The focused suite passed 59 tests. It includes deliberate red then green receipts for the item
budget, diagnostic privacy, policy validation, remote launch policy, public evidence parity,
timing receipt, and zero extra embedding, dense, sparse, reranking, or trust work.

```powershell
python -m pytest tests/test_guarded_spare_slot_shadow.py tests/test_replay_source_admission_spare_slots.py tests/test_source_conditioning.py tests/test_live_source_conditioned_admission.py tests/test_reasoning_embedding_reuse.py tests/test_doc_citations.py -q
```

The changed production modules passed the type checker, and the two changed runner modules passed
with their existing script import boundary ignored:

```powershell
.mypy-venv\Scripts\python.exe -m mypy --explicit-package-bases recall/source_conditioning.py recall_mcp/service.py recall_mcp/settings.py
.mypy-venv\Scripts\python.exe -m mypy --explicit-package-bases --ignore-missing-imports scripts/replay_source_admission_spare_slots.py scripts/run_live_tty_graph_precision.py
```

The repository wide type command was also run. It reported two pre-existing errors in
`benchmarks/atm_bench.py`, which is byte identical to `origin/master`; it reported no error in a
changed file.

```powershell
.mypy-venv\Scripts\python.exe -m mypy
git diff --exit-code origin/master -- benchmarks/atm_bench.py
```

Lint, architecture, citation, compilation, and whitespace checks:

```powershell
python -m ruff check recall/source_conditioning.py recall_mcp/service.py recall_mcp/settings.py scripts/replay_source_admission_spare_slots.py scripts/run_live_tty_graph_precision.py tests/test_guarded_spare_slot_shadow.py
python tools/architecture_map.py --check
python -m pytest tests/test_doc_citations.py -q
python -m compileall -q recall/source_conditioning.py recall_mcp/service.py recall_mcp/settings.py scripts/replay_source_admission_spare_slots.py tests/test_guarded_spare_slot_shadow.py
git diff --check
```

## Next gate

The next measurement is the preregistered fresh blinded validation. No promotion claim is allowed
until the triggered cohort contains at least 20 answerable and 20 unanswerable queries, a human
review is recorded before arm identity is revealed, and every safety and quality gate passes.
