# Convention: every experiment module states its prior-work search

**One line, in the module docstring, before anything else:**

```
Prior work: [[memo-name]] — <what it already established>
```

or, when nothing was found:

```
Prior work: none found (docs_search "<query>", source_type=memory)
```

For changed or newly added modules, run the repository check with the affected paths:

```text
python -m tools.prior_art check-experiments benchmarks/my_probe.py
```

## Why this exists rather than a rule someone remembers

On 2026-07-28 an abstention investigation was run that had already been run on 2026-07-24, and
reached the same conclusion — cosine AUC 0.78 against the earlier 0.753, entailment 0.59 against
0.648. Cost: roughly four hours and several hours of VPS CPU, including 777 queries through a
cross-encoder, to re-derive something already written down.

The rule to search memory first already existed in CLAUDE.md. It was skipped, and **nothing about
the skipping was visible** — a reader of the resulting probe could not tell "searched, found
nothing" from "never searched". That is the actual defect: not the omission, but that the omission
left no trace.

A rule whose compliance produces an artifact gets followed, because its absence is conspicuous.
The pre-registration convention works for exactly this reason — a prediction committed before a
run is checkable afterwards. This is the same mechanism applied to prior work.

## Scope

Any new module under `benchmarks/` or `results/` that measures something. A `PreToolUse` hook in
`.claude/settings.json` prompts at the moment of creation; this file is what the prompt is asking
you to satisfy.

Not required for: edits to an existing probe (the hypothesis was already declared), harness
plumbing, or scripts that only reshape data already measured.

## What a reviewer checks

Open the file. If the first paragraph does not say what was searched and what was found, the
search cannot be assumed to have happened — treat the experiment as potentially duplicating work,
and search before spending on it.
