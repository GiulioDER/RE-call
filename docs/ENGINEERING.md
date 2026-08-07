# Engineering

**1,300+ tests.** The database-touching ones run against a real pgvector container — no mock
DB. CI runs `ruff`, `mypy`, the suite against PostgreSQL under coverage, the suite *again* at the
declared dependency floor, and `pip-audit` over a checked-in `uv.lock` — each as a gate rather than
a report.

Type checking arrived late and is worth being specific about, because "we added mypy" is usually a
non-event. 81% of functions here already carried a return annotation and **nothing verified any of
them**. Running the checker over that found two things a green test suite had not:
`RECALL_TRANSPORT` was an unvalidated environment string flowing into a `Literal`-typed SDK
parameter — a typo reached `mcp.run()` as an arbitrary value after startup had already opened a
store and read the token file — and `ensure_schema` indexed a `None` row when pointed at an
existing table that was not a recall table. Both now fail early and by name. The gate is
`disallow_untyped_defs`, not a permissive baseline: a partially-checked package stops checking
wherever an annotation is missing, so a lenient gate passes while its coverage shrinks.

Tests are written to fail for the right reason. A representative sample:

- the RLS tests connect as a role that **cannot bypass RLS**, because as a superuser they would pass
  while testing nothing;
- the cross-tenant test asserts the other tenant's row **exists** before checking it is invisible,
  so a silently failed write cannot make it green;
- the supersession-cache test counts real table scans, so a "fix" that quietly became *rescan every
  search* would be caught;
- the metrics test asserts the counters move on the **real retrieval path** — instrumentation that
  is never wired up reports zero forever and reads as "nothing is going wrong".

Several defects were found only by running the library against a real corpus and a real server, and
each has a regression test quoting the input that caused it: a single NUL byte in one file aborting a
792-file index; every declared supersession edge failing on reference *formatting*; five tests that
encoded the developer's own environment and failed on a correctly-configured host.
