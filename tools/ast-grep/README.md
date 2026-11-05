# Structural lint rules (ast-grep)

Rules that match the syntax tree rather than text, for invariants ruff cannot express. CI runs them
in the `architecture` job; `sgconfig.yml` at the repository root points here.

```bash
pip install --require-hashes -r .github/requirements/ast-grep.txt
ast-grep test          # every rule against its valid and invalid cases
ast-grep scan          # the tree; exits 1 on an error-level match
```

| Rule | Invariant |
|---|---|
| `no-secret-valued-cli-flag` | no CLI flag takes a key, token, secret or password as its value; name the environment variable instead (`--api-key-env`) |
| `benchmark-indexer-without-cache` | no `Indexer` in `benchmarks/` gets the embedding cache unless a suppression says why |
| `python-module-run-with-safe-path` | in the production packages, a `python -m` launch passes `-P` or `-I`, so the working directory cannot shadow the installed package |

## Adding a rule

1. Write `rules/<id>.yml` and `rule-tests/<id>-test.yml`. The `invalid:` cases are the red proof
   the project requires: each must be a shape the rule exists to catch.
2. Mutate the rule (drop an alternative, widen a constraint) and watch `ast-grep test` report
   `[Missing]` or `[Noisy]`, then restore it.
3. `ast-grep test -U` writes the snapshots under `rule-tests/__snapshots__/`; commit them.
4. `ast-grep scan` must be clean. A real match is either fixed or suppressed with
   `# ast-grep-ignore: <id>` on the line above, beside a comment that says why.

ast-grep sees syntax only: no types, no data flow, nothing across files. A rule that needs those
belongs in a Python checker under `scripts/`.

## Red proof for the first two rules (2026-11-03, ast-grep 0.51.0)

- `no-secret-valued-cli-flag`: dropping `token` from the regex failed two invalid cases
  (`[Missing]`); renaming `--api-key-env` to `--api-key` in
  `scripts/run_openrouter_answer_batch.py` made `scan` exit 1.
- `benchmark-indexer-without-cache`: dropping the `cache=None` exemption failed the valid case
  (`[Noisy]`); deleting the suppression in `benchmarks/ladder/systems/recall_system.py` made
  `scan` exit 1.

## Red proof for `python-module-run-with-safe-path` (2026-11-05, ast-grep 0.51.0)

- Narrowing the safe flag from `-[PI]` to `-P` made the `-I` valid case fail (`[Noisy]`).
- Removing `-P` from `recall/desktop/runtime.py`'s MCP server launch made `scan` exit 1 there.
