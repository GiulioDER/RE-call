"""RE-call inside somebody else's benchmark harness. Repo-only; not part of the `recall` wheel.

The one thing this package is for
---------------------------------
Mem0 publishes LOCOMO / LongMemEval / BEAM numbers produced by their own harness
(github.com/mem0ai/memory-benchmarks). Every number RE-call has published so far was produced by
RE-call's harness. Those two sets of numbers are **not comparable**, and no amount of care in the
prose makes them comparable: different answerer prompt, different judge, different retrieval
budget, different scored subset.

The only way to put RE-call in a column next to their 92.5 is to run RE-call **through their
harness**, with their prompts, judge, budgets and scoring left byte-for-byte untouched. That is
what `RecallBackend` is: a façade with the exact shape of their `Mem0Client`, so their runner can
be pointed at RE-call by swapping one constructor.

The package is named `recall_interop` rather than `benchmarks` **deliberately**: their repo has a
top-level `benchmarks` package of its own, and a same-named package on `sys.path` would shadow one
of them — silently, and differently depending on which directory the run started from.

See `docs/their-harness-parity.md` for the mechanical patch their repo needs and the reproduce
commands. `[tool.hatch.build.targets.wheel].packages` deliberately does not list this package: it
is benchmark scaffolding, not library API.
"""

from recall_interop.memory_benchmarks import RecallBackend

__all__ = ["RecallBackend"]
