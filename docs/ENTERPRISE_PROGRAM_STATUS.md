# Enterprise retrieval program status

Rolling handoff between sessions of the enterprise retrieval program described in
[ENTERPRISE_RETRIEVAL.md](ENTERPRISE_RETRIEVAL.md). Newest session first. Each entry records what
landed, what was measured, and what is blocked, so the next session can start without
re-deriving state.

`docs/*.md` is deliberately outside `claim_gate.py`'s `GATED_DOCS`, so figures here carry no
evidence markers. If this file is ever promoted into the gate, every number below needs a marker
before that change can go green.

---

## 2026-08-05, the paired comparison: stopped before it ran, because its two arms are the same system

Branch cut fresh from `origin/master` at `ca8ccd8`, in its own worktree (`…/RE-call-paired`). The
primary clone was clean and on `master` for once; it was still left alone, because three
consecutive entries below record finding somebody else's uncommitted work in it.

### Session ledger

| # | Item | Outcome |
|---|---|---|
| 1 | Verify the three stated preconditions | done. **One holds, two do not**; evidence per precondition below |
| 2 | Predict the outcome and the invariants before running | done, and the prediction is what stopped the run |
| 3 | Run the paired comparison, both retrieval profiles, five corpora | **NOT DONE, deliberately.** The two named arms cannot produce different vectors. Measured, with controls |
| 4 | Machine-readable decision, negative result retained, archived | done, as a **precondition audit** rather than a counterfeit `PromotionDecision` |
| 5 | Decide which candidate the campaign should actually compare | **not mine to decide.** Two registered candidates are real; the choice is below and needs the operator |

### The headline: the candidate is not a different system

`bge-small-symmetric-v1` and `bge-small-asymmetric-v1` are registered in
`recall/embedding_registry.py` with the same `model_name`, `dimension`, `context_mode`,
`normalization`, `instruction_version`, `chunker_version` and `backend`. They differ in exactly two
fields, `query_mode` and `passage_mode`, and `/opt/recall-enterprise/manifest.json` records them as
the active and shadow profiles of **one** provisioned artifact tree
(`9a443d711e06…c919c`, recomputed this session and equal to the recorded value).

So the entire retrieval difference between the two arms is whatever the backend's `query_embed` and
`passage_embed` do differently from `embed`. Measured on VPS2 against that tree, offline: **nothing.
Six probes, three query-shaped and three passage-shaped, all three encoders byte-identical on every
one, cosine exactly 1.0.**

A paired comparison of the two is therefore a system compared with itself. Every question's `hit@5`
delta is zero by construction, the stratified paired bootstrap interval is `[0, 0]`, and the gate
refuses for a reason that describes neither profile. **That artifact already exists on master**:
`results/promotion/decision.null-difference.json`, produced by the previous session deliberately as
the harness's own null control. Running the campaign as briefed would have published the control a
second time under a candidate's name, and called it the first real paired comparison.

⚠️ **Worse than uninformative.** The two profiles have different fingerprints, so they index into
two different physical tables. Ties among equal scores can break on row order, so two provably
identical systems can still yield a handful of nonzero per-question deltas. A gate fed that noise
can report a corpus improvement with a Holm-corrected p-value attached to it. A null that is
*exactly* zero refuses cleanly; a null contaminated by tie-break noise is the shape that promotes
something.

**This was predicted in writing before it was measured**, from the registry alone, and the
measurement is the confirmation rather than the discovery. Session 8 had already recorded the same
fact (`docs_search`, cosine 0.81, no gap warning) and this run is an independent confirmation with
different probe texts, made because a fact that decides whether a whole campaign is worth running is
one to confirm rather than inherit.

### The preconditions: one holds, two do not

The brief said to stop and say so rather than improvise if any was missing. Two are missing.

| Precondition | Verdict | Evidence |
|---|---|---|
| The prior session's promotion harness is merged | **HOLDS** | `recall/eval/promotion/` is on `origin/master` at `ca8ccd8` (PR #208, merged as `c6c0197`), and `results/promotion/decision.null-difference.json` proves the producer has been driven end to end. ⚠️ PR **#203** (eval question-id validation, shared scoring flag, nearest-rank percentile) is still **OPEN**; it touches `benchmarks/` scoring rather than this package |
| Frozen manifests exist | **FAILS, for five of the six corpora** | the only frozen manifest anywhere is `results/promotion/labelled.manifest.jsonl`: 25 questions, digest `cad3281c…`, corpus `labelled`. `find / -xdev -maxdepth 8 -name '*.manifest.jsonl'` on VPS2 returns nothing. None exists for LOCOMO, the PEPs, the ladder, LongMemEval or MT-RAG |
| Profile-scoped calibration artifacts exist for both profiles | **FAILS, for both** | the mechanism exists (`save_for_profile` / `load_for_profile`, `PROFILE_FINGERPRINT_KEY`, covered by `tests/test_calibration_profile_scope.py`). No artifact does. The only two RE-call calibration artifacts on VPS2 are `/opt/recall-beam/calibration.json` (`voyage-4-large`) and `/opt/recall-beam/beam_calibration.json` (`openai:openai/text-embedding-3-small`), **both `certified: false`**. Relation `recall_calibrations` **does not exist** in `recall_enterprise` or `recall_bench`, and `/opt/recall-enterprise/manifest.json` records `core_schema_current: 0010` while calibration binding is migration `0011` |

**The corpus DATA is not the problem.** LOCOMO (`/opt/membench-run/locomo10.json` and two more
copies), LongMemEval (`/opt/longmemeval/`, the `oracle` and two `cleaned` variants), MT-RAG
(`/var/tmp/re_call_mtrag_20260803/mt-rag-benchmark`) and the PEPs (`/opt/peps-corpus`) are all
provisioned, and the ladder's manifest ships in the repository. What is absent is the frozen
question set, which is the artifact whose entire value is that it predates any candidate result,
and the calibration without which `superseded_trust_rate` is `null` rather than a number.

That last one is not a formality. `recall.eval.promotion run --trust-policy strict` **refuses**
without a generation-bound certified calibration, and `--trust-policy development` records every
verdict as `unverified`, under which the gate's "superseded trust rate exactly zero" criterion
cannot be evaluated at all. The existing null-difference decision says exactly that in its
`failures` list.

### What was measured

`benchmarks/check_profile_encoder_distinctness.py`, run on VPS2 by root SSH. qwen-mcp's file roots
are `/opt/sentiment_agent`, `/var/lib/qwen_agent`, `/var/log/qwen_agent` and
`/etc/systemd/system`; this program lives in `/opt/recall-enterprise`, outside all four, so root SSH
is the documented fallback and is stated here rather than left to be inferred.

The script does not import the `recall` package. The subject is what the BACKEND does, and asking
the code under test whether its own two configurations differ is the self-comparison this program
keeps recording. Two controls, both blocking:

| Control | Result |
|---|---|
| Positive: the same comparator on two texts that must not agree | fired. Bytes differ, cosine `0.654948234558` |
| Determinism: the same call twice | byte-identical, so an "identical" reading is not luck |
| **The positive control can refuse** | proven by mutation: one line changed on a copy so the control compares a text with itself. **Exit 2, zero bytes of stdout, no verdict produced at all.** The unmutated script was verified unchanged by SHA256 afterwards |

The third row is the point. A control that runs is not a control that fires, and this program has
spent two sessions on guards that read as protection and could not.

**No timing is reported and none is citable.** VPS2 showed a load average of 9.37 / 8.79 / 9.46 on
12 cores while this ran, from unrelated live production. The only numbers taken from VPS2 here are
digests, byte comparisons and cosines.

### Archived

`/var/lib/recall-benchmarks/2026-08-05-embedding-profile-distinctness/`, six files, with
`MANIFEST.sha256` covering all of them and `sha256sum -c` passing. The archived script is
byte-identical to the committed one (`2b04d6af93e2…12cf0`), which is the check the MT-RAG salvage
entry below had to do the hard way after a runner survived only as untracked files in `/var/tmp`.

`precondition_audit.json` (schema `recall-precondition-audit-v1`) is the machine-readable
deliverable. It is deliberately **not** a `PromotionDecision`: no arm was scored, so emitting that
schema would have described a gate evaluation that never happened. A refusal to run and a run that
refused must not read the same.

### What was deliberately not done

1. **No arm was scored, no manifest was frozen, no calibration was authored.** Each of those would
   have been improvising a missing precondition into existence, which the brief forbade and which
   would have produced a manifest that postdates the candidate — the one property a frozen manifest
   exists to deny.
2. **The MT-RAG pool-100 preregistration was not touched, relabelled or extended.** It is a valid
   preregistered baseline and it cannot certify a pool-20 quality profile.
3. **No new profile was registered.** Giving BGE a query instruction would make the asymmetric
   profile genuinely asymmetric, and that is a new model candidate, which this program's standing
   scope puts behind a separately registered experiment.

### The decision the next session needs from the operator

The campaign's machinery is sound; only its candidate is empty. Two registered candidates would
make it a real experiment, and choosing between them is a decision rather than a fix:

1. **The context-mode profiles** — `bge-small-context-{document,section,neighbor}-v1` against
   `bge-small-symmetric-v1`. These genuinely differ: the same encoder over different passage text,
   which is the one axis the byte-identity finding above leaves intact. The context-modes entry
   below deliberately did not measure which mode wins and named it the next campaign. **This is the
   paired embedding comparison that is actually available today.**
2. **The retrieval profiles** — `fast` against `quality` (pinned MiniLM, pool 20) on one embedding
   profile. Also a real difference, and it is the arm the MT-RAG pool-100 results cannot certify.
   But it varies the retrieval profile rather than the embedding profile, so the gate's arms mean
   something different and the brief's framing would have to move with it.

Either way the two failing preconditions have to be met first: freeze a manifest per corpus, and
author a generation-bound certified calibration (which needs migration `0011` applied on the host,
where the schema is currently at `0010`).

### Gates run

| Gate | Result |
|---|---|
| `ruff check .` | see the commit; no source behaviour changed |
| `mypy` | as above |
| `pytest` | the suite was not re-run for a diff of one new benchmark script and two documents; the claim-gate documents were not touched |
| Positive control on the new measurement | fired, and **proven able to refuse** by mutation |

### Standing blockers

| Blocker | Kind | Effect | Change |
|---|---|---|---|
| **No latency reference host.** VPS2 showed load average 9.37 on 12 cores during this session. | External dependency. Do not work around it. | Latency is **PENDING**; promotion blocked on latency grounds. Quality and safety gates still run. | unchanged, restated from observation |
| **No frozen manifest outside `labelled`.** | Open, **new to this list** | Four of the five corpora in the campaign brief cannot be scored reproducibly. | new |
| **No certified calibration for any BGE profile, and `recall_calibrations` is not deployed.** | Open, **new to this list** | Under `strict` the harness refuses; under `development` every trust verdict is `unverified` and the superseded-trust gate cannot be evaluated. | new |
| **No production corpus.** | Open | Nothing may be claimed about enterprise-corpus behaviour. | unchanged |
| **No approved local generator confirmed.** | Open | The generator-neutral evidence path stays unexercised end to end. | unchanged |

### What the next session should start with

1. **The operator's answer to the candidate question above.** Everything else is downstream of it.
2. **Freeze a manifest per corpus**, once the candidate is chosen: `python -m recall.eval.promotion
   freeze --corpus …`. The adapters for all six exist; the data is provisioned on VPS2; nothing has
   been frozen but `labelled`.
3. **Apply migration `0011` on the host and author a calibration bound to a generation.** Until
   then no decision can be green, and the `superseded_trust_rate` gate is not merely failing but
   unevaluable.
4. **PR #203** is still open and touches the scoring path. Land it before any campaign quotes a
   percentile.
5. The dual-write skip defect (`recall/index.py:426`/`:459`) named two entries below is still open
   and is still the only item that can silently half-populate a shadow generation.

---

## 2026-08-05, retrieval profiles: a budget that does something, bounded cost, a complete result surface

### Session ledger

| # | Item | Outcome |
|---|---|---|
| 1 | Profiles behave as specified, selection is process level, conflicts refuse **startup** | done; the conflict used to refuse the first *search*, not startup |
| 2 | Decide and implement what `latency_budget_ms` means at request time | done, admission deadline + reported overrun; see below |
| 3 | Resource bounds: one reranker per worker, thread limits, bounded queues, rejection **before** embedding, separate fast/quality concurrency | done; the rejection ordering was already right and is now proven, the rest was not |
| 4 | Result surface: profile, generation, pool size, rerank flag, and all seven stage timings | done, `evidence_assembly` was the missing bracket |
| 5 | Safety: dense cosine preserved, no query or corpus text in logs | done, with a positive control on the detector |
| 6 | Prove every new test can fail | done, **54 of 54 mutations killed**, all as clean assertion failures |
| 7 | CCA audit at DEEP, plus the anti-regression and architect gates | done; **the audit invalidated the first version of item 3**, and the two gates then found six more, one of them inside the audit fix. Reported first, below |

This is backlog session 9 (items 25 to 28) plus the parts of areas 4, 5 and 6 the gap matrix marked
untested. Backlog item 29 (the reranker's `local_files_only` / `artifact_sha256` offline path and
the symlink-escape refusal) is **not** done: it needs the `rerank` extra installed to exercise, and
this session added the pin rather than the loader test.

**Correction, made on merge.** This entry was written saying "session 3, the outbox drain, is still
first in the backlog and still untouched". That was true when the branch was cut from `98f2a85` and
is not true now: a concurrent session landed it as #198 and #201 while this branch ran, and its
entry sits directly below this one. An earlier draft of this paragraph also said `origin/master`
did not move during the session; it moved **28 commits**, across two merges: 23 before the PR
was opened and 5 more in the minutes after, which GitHub reported as CONFLICTING.

The branch was **merged rather than rebased**, because four commits would each have had to resolve
the same two shared-document conflicts. Both documents were resolved by **reconstruction with
assertions**: upstream's copy is carried through byte for byte and this entry inserted before it,
verified by length and by content rather than by reading the diff. Upstream's new
`tests/test_env_example_parity.py` — which is, pleasingly, the gate this entry recommends below as
future work — passes against this branch's `.env.example`.

⚠️ One of those assertions was itself too wide, which is worth recording because it is the same
shape as the defects this session spent its day on. The "no conflict markers survived" check was a
plain substring test, and it fired on a **historical CHANGELOG entry that describes** a release
which shipped raw markers. Anchoring the pattern to line starts, which is what git actually writes,
is the difference between a guard and a text search.

### The audit found the guard could not fire, and that is the headline

The work above went through the tiered CCA pipeline at DEEP (forced: the diff trips the numeric
path). Ten auditors, 46 raw findings. **The most important one invalidates the first version of
this session's central claim**, so it is recorded before the claim rather than after it.

**`queue_full` could never fire through the server, and the budget did not bound the client's
wait.** Every MCP tool body runs inside `anyio.to_thread.run_sync`, so a request parked in
`RetrievalAdmission` is holding a worker thread. anyio's default limiter is **40** tokens. Fast's
`8 + 32` is also **40**. The 41st concurrent search therefore never reached `__enter__` at all: it
waited in anyio's limiter, which has no timeout, no budget and no counter. A saturated process
would have queued unboundedly while `recall_retrieval_rejected_total` read zero, which is exactly
the scenario the budget was written to prevent.

Three independent auditors found it and each measured `total_tokens == 40` rather than asserting
it; I re-measured before acting. **A guard that reads as protection and cannot fire** is this
project's standing lesson, and the first version of this session's work was an instance of it. The
existing test proved the mechanism on a hand-built 1+1 profile and never through the server:
**a positive control validates the MECHANISM, not the SCOPE.**

Fixed by sizing the pool from the profile (`worker_thread_budget` = admission capacity plus eight
reserved threads, raised at startup, never lowered), with the invariant asserted for every profile
and the application of it tested separately, because asserting an invariant is not enforcing it.

### What the rest of the audit changed

| Finding | Verdict | Outcome |
|---|---|---|
| Admission capacity equals the anyio worker pool, so `queue_full` is unreachable | CONFIRMED, measured | fixed, above |
| CHANGELOG mis-stated the one genuinely new startup refusal | CONFIRMED | fixed; see below |
| `_validate_quality_reranker_config` compared a normalised digest and returned the raw one | CONFIRMED, measured | fixed, normalise once and return that |
| Budget charged twice (admission timeout AND end-to-end deadline) | CONFIRMED | fixed, `budget_exceeded` now on served work; `admission_wait` is a stage |
| `QUALITY_PROFILE != resolve_retrieval_profile("quality")`, so they minted two admission queues | CONFIRMED, measured | fixed both ways: constant matches its resolver, and the queue is keyed on `queue_identity` |
| `_admission` still on `lru_cache`, the defect this diff fixed for the reranker | CONFIRMED | fixed, same lock |
| A failed reranker construction was retried on every request, re-hashing the model tree | CONFIRMED | fixed, failures are cached |
| `recall_retrieval_total_ms` was success-only | CONFIRMED | fixed, observed on every exit plus `recall_retrieval_failed_total` |
| Legacy's 24-day sentinel shipped to clients as `latency_budget_ms` | CONFIRMED | fixed, `null` when no budget is enforced |
| Running-slot leak in the window between acquire and its store | CONFIRMED | **NOT closed.** An ownership flag was added and the anti-regression gate then showed the branch cannot fire: CPython's exception table makes the acquire call the only interruptible point, where the flag is still false, and the real window is inside `Semaphore.acquire`. The guard is kept (correct, free) and the claim is withdrawn |
| `RECALL_RERANK_THREADS` documented as general; only read on quality | CONFIRMED | doc fixed, behaviour unchanged |
| Docs claimed model+revision are pinned; only the digest is enforced | CONFIRMED | doc fixed, and the tree-vs-model limit stated |
| `.env.example` blank keys break a rollback to the previous parser | CONFIRMED | keys commented out; rollback note in the CHANGELOG |
| Two `### Added` blocks in `[Unreleased]` | **FALSE POSITIVE** | `HEAD~1` already had six such groups; it is the file's per-session convention, not something this diff introduced |
| Per-request metric overhead (~87 us, 0.035% of the fast budget) | CONFIRMED, measured | no change; the auditor's own verdict was "not material" |

**The CHANGELOG correction is the one worth naming.** I had written that a contradictory pair, a
missing reranker path, or a non-pinned digest "used to produce a server that came up clean and
failed on its first client request". That is true of four of the five newly refusing
configurations, and **false of the fifth**: before this change the operator's digest was passed
straight to `verify_artifact`, so a quality deployment whose digest correctly described its *own*
reranker tree started **and served every request**. It is now a breaking change for that
deployment, and the CHANGELOG says so.

### The two remaining gates then found six more, including one in the audit fix itself

The anti-regression gate (`differential-review` over the fix diff) and the architect gate both
returned **REVISE**. Running them mattered: three of their six findings are in code the audit
itself never saw, which is this program's own lesson that **a fix can promote a dormant defect, so
the audit goes after as well as before**.

| Finding | Verdict | Outcome |
|---|---|---|
| **The `k` clamp had no test.** It is the entire mechanism behind "a client cannot request a more expensive profile", newly advertised in three documents, and deleting it left all 2345 tests green | CONFIRMED | fixed: a test with a legacy arm, so it discriminates the clamp rather than a small store |
| **A shed request was counted as a failure and injected its budget-length wait into the served-latency histogram.** Measured by the reviewer, reproduced here | CONFIRMED | fixed: `RetrievalOverloaded` is matched before the general handler, so a shed appears only in `recall_retrieval_rejected_total`. It would otherwise have contaminated the p95 this program is blocked on, in exactly the overload regime that matters |
| **Sizing the pool from the profile removed anyio's 40 as an accidental thread ceiling**, making `RECALL_SEARCH_QUEUE` an unvalidated thread-count knob (`=5000` would ask for 5008 threads) | CONFIRMED | fixed: `MAX_ADMISSION_CAPACITY = 256`, refused at resolution |
| **Re-raising one cached exception instance grows its traceback per call**, and each retained frame pins its locals, which on this path include the query text | CONFIRMED, measured (4 → 7 → 10 → 13 → 16 frames) | fixed: cache `(type, args)` and raise a fresh instance. This was also a safety defect, not only a leak |
| **`except BaseException` cached a `KeyboardInterrupt`**, turning a transient event into a process-lifetime outage | CONFIRMED | fixed: narrowed to `Exception` |
| **`inference_threads` was plumbed but never shown to reach the reranker** | CONFIRMED | fixed: a recording stub asserts the kwarg, without needing the `rerank` extra |
| `.env.example` rollback premise ("no version parses these keys strictly") | **REFUTED** | `git show 98f2a85:recall/profiles.py` parses them strictly, and the deployed VPS2 wheel is built from an ancestor. The fix stands; the CHANGELOG wording now says which builds are affected rather than implying released ones |

**And one claim of mine was withdrawn rather than softened.** The ownership flag added for the
running-permit leak **cannot fire**: CPython's exception table makes the acquire call the only
interruptible point in that block, where the flag is still false, and the real window lives inside
`Semaphore.acquire`. I had recorded that leak as fixed. It is not. The guard is kept because it is
correct and free, and the claim is gone from the code comment, this document and the table above.

### What the audit surfaced and this session did NOT fix

Recorded rather than acted on, with the reason:

1. **No tenant dimension in admission.** One tenant can fill the queue, and the new shedding turns
   that from queueing into active rejection for every other tenant. The rate limiter bounds call
   RATE, not concurrency. This is a design decision about fairness, beyond the scope given.
2. **The reranker pin is a tree digest of one provisioned directory**, not a portable model
   identity, and no shipped command reproduces that tree. Documented as a limit; deciding between
   "pin the tree" and "pin a portable identity" is a decision, not a bug fix.
3. **`total_ms` includes the queue wait, so it is a weak cross-tenant load signal.** Low severity
   and the field is genuinely useful; noted.
4. **`RetrievalOverloaded`'s message discloses the process's admission parameters** to a client.
5. **`retry_after_seconds` is a backoff hint derived from the budget, not a computed time to
   success** the way `RateLimited`'s is. Honest documentation was chosen over inventing a drain
   estimate this program cannot measure.
6. **The legacy `RECALL_RERANK` path is still not validated at startup.** The claim is now scoped
   to the profile path rather than the claim being widened.
7. **`_RERANKERS` is keyed on the profile name, not the full reranker identity.** Unreachable with
   a static process environment, which is the documented deployment model.

⚠️ **Two auditors died mid-run on API errors** (`code-auditor`, `env-validator`), so general code
quality and the full `.env.example` round trip were **not** covered. A skipped gate and a passing
gate must not read the same, so: those two dimensions are unaudited for this change.

⚠️ **Deterministic coverage for this run was NONE.** `cca_checks` is not installed in this venv, so
no static backend was available and every verdict above rests on LLM adjudication plus whatever I
re-executed myself. Everything marked "measured" is a command I ran.

### What `latency_budget_ms` means now

It was declared on every profile, validated in `__post_init__`, and read by nothing:
`git grep` returned three hits, all inside `recall/profiles.py`. The promotion gate's budget is a
separate caller-supplied float. It now means exactly two things, both observable and both tested.

**It bounds the admission wait.** `RetrievalAdmission.__enter__` takes the queue slot
non-blocking, then the running slot with the budget as its timeout. A request that cannot start
within the budget is shed with `RetrievalOverloaded(reason="budget_exhausted")` *before the query
is embedded*. Previously the running-slot acquisition blocked with no timeout, so `queue_capacity`
bounded how many threads could be parked and said nothing about how long any of them waited.

**It labels an overrun.** A request that finishes over budget still returns its answer and reports
`total_ms`, `latency_budget_ms` and `budget_exceeded`, plus
`recall_retrieval_budget_exceeded_total{profile}` and a warning carrying numbers only.

**The budget is charged once.** `budget_exceeded` is computed on the work the request did
(`total_ms` minus the `admission_wait` stage), not on end-to-end latency. Since the budget is
already spent as the admission timeout, charging it again would label a fast retrieval slow
because another request was ahead of it, and would saturate the counter under any queueing.
The legacy profile enforces no budget and reports `latency_budget_ms` as `null` rather than the
24-day sentinel used internally.

**A mid-flight abort was rejected, deliberately.** There is no cancellation point inside a blocking
cross-encoder `predict`. Aborting would pay the whole cost and then discard the answer, which turns
a latency regression into an availability incident. Shedding happens at the door, where it is free.

⚠️ This makes the budget *enforced*. It does not make it *validated*: no measurement here says 250
ms or 1500 ms is the right number, and none can be taken until the latency blocker below is
resolved.

### A latent bug the new timeout would have made live

`RetrievalAdmission.__enter__` acquired the queue slot and then the running slot. `__exit__` does
not run when `__enter__` raises, so any failure between the two lost that queue slot permanently.
Once every one of the `max_concurrency + queue_capacity` permits has leaked the process refuses
every request forever while reporting itself merely busy; partial leakage degrades proportionally.
(An earlier draft of this entry said `queue_capacity` leaks were enough. The audit caught it: after
32 leaks a fast process still serves 8 concurrent requests, with no queue depth left.) It was
unreachable before (nothing could fail between the two acquisitions) and would have become
reachable the moment the timeout was added. The release is now explicit on every failure path,
including `BaseException`.

⚠️ **The running-permit half of this is NOT fixed, and the anti-regression gate is what caught the
overclaim.** An ownership flag was added so the handler could release the running permit too. The
gate then showed the branch cannot fire: CPython's exception table makes the acquire call the only
interruptible point in that block, where the flag is still false, and the actual window lives
inside `Semaphore.acquire` between its counter decrement and its return, which is not reachable
from calling code. The flag is kept because it is correct and free, but **a guard that reads as
protection and cannot fire is exactly what this program refuses to count**, so the claim is
withdrawn rather than softened. The queue-permit leak, which was the reachable one, is closed.

The test that pins it discriminates on the *reason* of a second rejection, not on a count: a leaked
slot makes the next attempt `queue_full` without waiting, and only a returned slot lets it reach the
budget wait again.

### What else landed

**Separate concurrency budgets.** Both profiles inherited `max_concurrency=4` / `queue_capacity=16`;
legacy still does. Fast is now 8 + 32 and quality 2 + 8. Quality's per-request budget is six times
fast's, so an equal queue depth would make its clients wait roughly six times as long; the new
values hold `queue_capacity x latency_budget_ms` within one order of magnitude (fast 8000
slot-milliseconds, quality 12000). **Slot-milliseconds, not CPU-seconds:** an earlier draft of this
entry called the quantity CPU-seconds, which is wrong by a factor of 1000 and names a factor
(`max_concurrency`) the argument is not about. `latency_budget_ms` bounds a WAIT, not a service
time, so nothing here is a claim about CPU consumed.

**These are a policy choice and are labelled as one in the code, the doc and `.env.example`.** They
are not tuned to measured throughput and cannot be until there is a reference host.

**One reranker per worker, under a lock.** The shared instance was memoised with `lru_cache`, which
is a cache lookup and not a construction lock: on a cold start under load, every concurrent first
request missed and loaded its own copy of a cross-encoder. The test uses a factory that sleeps, so
the race is deterministic rather than incidental; unlocked, eight threads build eight models.

**Startup refuses a bad cost profile.** `startup_retrieval_profile` resolves the profile and
validates the quality reranker configuration, and it is the first thing `_lifespan` does, ahead of
any I/O. It imports no torch and loads no model: a configuration check that needed the extra
installed would not be a startup check. The test asserts the refusal is the *profile* one, so the
check cannot be silently moved below the store setup.

**The quality reranker is pinned by digest.** `RECALL_RERANK_SHA256` used to be whatever the
operator typed, which made the `local_files_only` verification self-referential: it proved the tree
hashes to its own hash, which every tree does. `recall/rerank.py` now pins artifact `db6ad879…` and
the environment must equal it.

Two limits the audit made explicit, both now in the docs. The model name and revision recorded
beside the digest are **provenance, not a runtime check** (nothing reads them; the quality profile
loads locally, where the Hub revision is unused). And the digest hashes a whole provisioned
**tree**, path names included, so it identifies one directory rather than the model in general;
no shipped command reproduces that tree elsewhere.

**`admission_wait` and `evidence_assembly` timings.** Five stages were timed; queueing and the
assembly of the client-facing evidence were not. All eight are now on `stage_ms` and observed into
`METRICS` as `recall_retrieval_stage_ms{profile,stage}` alongside `recall_retrieval_total_ms`
(backlog item 28), the latter on **every** exit including failures. Every label is
library-authored; no corpus-derived string can reach one.

**`RetrievalOverloaded` is now a retryable refusal** carrying `reason` and `retry_after_seconds`,
following `recall_mcp.limits.RateLimited` rather than inventing a second convention (backlog item
25). The retry hint is capped at 5 s so the legacy profile's 24-day sentinel budget cannot be handed
to a client.

### What was measured

**The reranker pin, on VPS2.** Root SSH, not qwen-mcp: this program lives in
`/opt/recall-enterprise`, outside qwen-mcp's four file roots. `artifact_tree_sha256` was
**reimplemented in a standalone script** and run over the provisioned trees, deliberately not
importing the deployed wheel, so the result is an independent recomputation rather than the code
checking itself.

| Tree | Recomputed digest | Agrees with |
|---|---|---|
| `models/ms-marco-MiniLM-L-6-v2` | `db6ad87969c7…2ab2a` | `manifest.json`, recorded 2026-08-03 |
| `models/bge-fastembed-cache` | `9a443d711e06…c919c` | `manifest.json`, and the 2026-08-05 embedding session's own run |

The second row is the positive control: it reproduces a digest two independent tools already agree
on, so a recomputation that matched the first row by accident would have had to match this one too.
Nothing under `/opt/recall-enterprise` was modified; the script lives at
`/var/tmp/recall-reranker-pin-check.py`.

**The latency blocker, restated from observation rather than memory.** VPS2 reported a load average
of 9.78 / 9.44 / 8.84 on 12 cores during this session, from unrelated live production. Unchanged and
not worked around. **No timing in this session is cited for any promotion decision**; the only
numbers taken from VPS2 here are checksums.

### Gates run

| Gate | Result |
|---|---|
| `ruff check .` | clean |
| `mypy` | clean, 139 source files |
| `pytest -q` | **2316 passed, 35 skipped, 0 failed** (7 m 48 s) on the branch; **2533 passed, 36 skipped, 0 failed** (10 m 04 s) after merging the 28 upstream commits. Throwaway pgvector container on port 5437 |
| Mutation sweep | **54 of 54 killed** (two repairs along the way; see below) |
| CCA audit | DEEP tier, 10 auditors, **2 died mid-run**; anti-regression and architect gates both REVISE, then satisfied |

The suite ran against a container created for this session rather than the shared dev database on
5432, following the previous entry's finding that a test database provisioned before #196 must be
recreated. Two other containers were up on this machine and were left alone.

### A hazard this session introduced, and caught

Emptying `RECALL_SEARCH_CONCURRENCY` / `RECALL_SEARCH_QUEUE` in `.env.example` (so they default to
the selected profile rather than to a shared `4` / `16`) would have made the shipped example refuse
startup: a dotenv load puts an empty **string** in the environment rather than omitting the key, and
`_positive` read that as a malformed integer. Empty now means unset, with a test asserting that the
empty and absent resolutions are equal objects rather than merely both non-raising.

Worth naming because the defect was created by a documentation edit and would have been found by an
operator, not by the suite: nothing tests `.env.example` against the parser.

### Proving the tests can fail

54 mutations, one narrow change each, across `profiles.py`, `service.py`, `server.py`, `rerank.py`,
`retriever.py` and one against the test file's own leak detector. The harness aborts the entire run
if a search string is absent or occurs more than once, and it restores by **bytes** rather than
`write_text`, which is what left ten files spuriously dirty last session. A final pass asserts every
touched file is byte-identical to its starting content; it was.

**The sweep found a test of mine that could not discriminate**, which is the reason to run it at
all. `the budget is charged twice, wait included` survived: the fixture queued a request for 200 ms
and then ran a microsecond search, so even with the wait charged the total stayed inside the 250 ms
budget and a double-charging implementation passed. The fixture now queues ~150 ms **and** does
~150 ms of work, so wait plus work crosses the budget while the work alone does not, and it asserts
both halves of that so a future edit cannot quietly return it to the vacuous shape.

**And one "kill" was not a kill.** `an interrupt during a cold build poisons the cache` reported
red, but the run said `no tests ran`: the mutated code let a `KeyboardInterrupt` escape the test,
which aborts the pytest session rather than failing an assertion. **A session that never finished
is not evidence a test can fail** — the same class as a guard that is silent for an unrelated
reason. The test now catches it explicitly and calls `pytest.fail`, so the mutation produces a red
test. All 54 kills are clean assertion failures.

Three others are worth naming because they separate a real guard from a description:

* **"the query is embedded before admission is taken"** inserts one `timed.embed([query])` above the
  admission block. The rejection still raises and the process still looks like it has working
  overload control; only `embedder.calls == 0` fails. That assertion is the ordering requirement,
  and nothing else in the test would have noticed.
* **"hits carry the fused rank instead of the dense cosine"** replaces the dense score with the RRF
  value in `recall/retriever.py`. The fixture uses two chunks at cosine 1.0 and 0.6, so a single
  wrong-but-plausible score cannot satisfy both assertions.
* **"the log-leak detector reads only the rendered message"** mutates the *test's* own helper. Two
  of the three planted leaks ride the `extra=` channel and the exception text, which a detector
  reading `getMessage()` alone would miss, so this proves the detector rather than the code.

The one search-string mismatch this run produced (a wrong indentation) aborted before any test ran,
which is the harness behaving as designed.

### Decisions a reader should be able to reverse

1. **The budget sheds at the door and never aborts in flight.** The alternative is a hard deadline
   with a cancellable retrieval, which would need cancellation points the cross-encoder does not
   offer. Revisit only with a real cancellation mechanism, not with a timer.
2. **The concurrency numbers (8/32 and 2/8) are unmeasured policy.** They are the first thing to
   re-derive once a reference host exists. Nothing depends on their exact values.
3. **`RECALL_SEARCH_CONCURRENCY` / `RECALL_SEARCH_QUEUE` now default to the profile's value**
   rather than a shared 4/16, and `.env.example` no longer ships those two numbers pre-filled. An
   operator who had copied the example and relied on the literal `4` will now get the profile
   default instead. That is the intended direction; it is a behaviour change on an opt-in path.
4. **A reranker digest that is not the pin refuses.** Fail-closed, and consistent with how
   `embedding_registry` treats artifacts, but it means an operator with a legitimately different
   local copy must register an experiment rather than set a variable.
5. **`RetrievalOverloaded` propagates as an exception with structured fields**, matching
   `RateLimited`, rather than being mapped to a distinct MCP error type. If the MCP layer ever
   grows a real status taxonomy, both should move together.
6. **The stage-timing metric has no tenant dimension.** Profile and stage only. Adding a tenant
   label would make the series cardinality tenant-controlled, and the gap matrix's note about
   `MetricsRegistry` accepting unconstrained labels is still open.

### Standing blockers

| Blocker | Kind | Effect | Change |
|---|---|---|---|
| **No latency reference host.** VPS2 showed load average 9.78 on 12 cores during this session. | External dependency. Do not work around it. | Latency is **PENDING**; promotion blocked on latency grounds. Quality and safety gates still run. | unchanged. The budget is now enforced but its VALUE is unvalidated |
| **No production corpus.** | Open | Nothing may be claimed about enterprise-corpus behaviour. | unchanged |
| **No approved local generator confirmed.** | Open | The generator-neutral evidence path stays unexercised end to end. | unchanged |

### What the next session should start with

1. ~~Session 3 of the backlog, the migration outbox drain.~~ **Landed by a concurrent session while
   this branch ran** (#198, #201); see the entry directly below. Read that one before planning,
   because this list was written against a backlog that has since moved.
2. Backlog item 29, deliberately left here: cover the reranker's offline loader path
   (`local_files_only`, `artifact_sha256`, `inference_threads`) and the symlink-escape refusal.
   This session pinned the digest but never loaded a model; the pin and the loader are two
   different guards and only one of them has been shown to fire.
3. Decide whether `MetricsRegistry.increment` / `observe` should take an allowlist of label values.
   Every call site is library-authored today and one new careless call would end that, which is the
   same shape as the `advice` injection this codebase already fixed once.
4. Consider a test that parses `.env.example` through the resolvers that read those keys. This
   session created a startup-refusing example with a one-line documentation edit and caught it by
   reasoning, not by a gate.
5. **Decide the tenant-fairness question in admission** (audit item 1 above). The shedding this
   session added turns one tenant's saturation into every other tenant's rejection, and the rate
   limiter bounds call rate rather than concurrency. This became a live question because of this
   change, so it should not wait behind the whole backlog.
6. **Re-run the two auditors that died**, `code-auditor` and `env-validator`. General code quality
   and the `.env.example` round trip are unaudited for this change.

---



## 2026-08-05, context modes: three modes given a rule each, and two rules that were not enforced

Backlog session 9. Branch cut fresh from `origin/master` at `fa673e5`, in a separate worktree
(`…/RE-call-ctxmodes`) because the primary clone was mid-flight on someone else's work — see below.

### Session ledger

| # | Item | Outcome |
|---|---|---|
| 1 | Three deterministic context modes that never alter stored chunk text | done; `chunk_text()` and the stored row are byte-identical under every mode |
| 2 | One test per rule, eight rules | done, 69 tests across two files |
| 3 | The load-bearing invariant, per mode, over five corpus shapes | done at the function level and again against real PostgreSQL rows |
| 4 | Prove every test can fail by mutation | done, **38 of 38 mutations killed, 0 unapplied** |
| 5 | Decide which mode wins | **deliberately not done.** That is session 10's campaign; running it here would have measured the harness |

`ruff check .` clean on 0.15.22 (inside CI's `>=0.5,<0.16` pin), `mypy` clean across 139 source
files.

### The primary clone was someone else's, again

`C:/…/eva/work/RE-call` was on `codex/retrieval-profiles-bounded-cost` with **620 uncommitted
insertions** across eight files, including `recall/profiles.py`, `recall_mcp/service.py` and two
untracked test files — and `recall/profiles.py` is a file this session's brief also named. Nothing
of theirs was branched over, committed, stashed or reverted; this session ran in its own worktree
with its own `.venv` (the clone's venv is an editable install pinned to the clone's absolute path,
so running from the worktree would have imported THEIR code while reporting mine).

`origin/master` also moved twice mid-session, `fa673e5` to `c02645b` (#202, OIDC subject binding).
Only `CHANGELOG.md` overlaps.

**This is the third consecutive session to record the same thing.** Check `git worktree list` and
`git status` in the primary clone before branching there; the answer has been "someone else is in
it" every time.

Two smaller races, both new, both worth the next session's attention:

* **A subagent overwrote this session's mutation harness.** The architect gate wrote its own
  one-off mutation script to `…/scratchpad/mutate.py`, the exact path the sweep runner lived at,
  and the next sweep silently ran the subagent's single mutation instead of thirty-nine. It was
  caught only because the output carried a `-x` flag and a `RESTORED` line the harness never
  prints. The scratchpad is shared exactly like the clone is; the runner is now
  `sweep_runner.py`, and a session that hands agents a working directory should assume they
  write into it.
* **The `CHANGELOG.md` conflict resolution committed its own conflict markers, and the guard that
  should have prevented it is what caused it.** The resolver asserted `"\n=======\n" not in out`;
  a bare `=======` is legal Markdown and occurs elsewhere in a 100k-line changelog, so the
  assertion tripped, the script died **before writing anything**, and the `git add` on the next
  shell line was not chained behind its exit code. `git rebase --continue` then committed the
  markers. Repaired by rebasing both commits, and verified per commit rather than at the tip.
  The shape is general: **a resolver's guard must match the marker PREFIXES (`^<<<<<<< `,
  `^>>>>>>> `) rather than a substring that legal content can contain, and the command that
  stages its output must be chained behind it.** A check that fails open is worse than no check,
  because it also stops the work that would have succeeded.

### What the shipped code actually did, measured before anything was changed

The brief listed eight rules. Six were already enforced by `recall/context.py`. Two were not, and
both were found by executing the shipped code against each rule rather than by reading it:

| Rule | Shipped behaviour |
|---|---|
| Title precedence | **defective.** The frontmatter scan compared `key.strip()` and returns on its first hit, so an indented `title:` nested under any other mapping outranked the document's own when it appeared above it |
| Root-relative paths only | **unenforced.** `/etc/passwd`, `C:\Users\…`, `//server/share` and `../../../etc/shadow` all reached the rendered `source:` field verbatim |
| Caps 256 / 256 / 512 | held |
| Neighbour ≤ 200 per side | held |
| Complete chunk preserved | held |
| Degradation order | held (verified by a monotonic budget sweep, not by reading the list) |
| Mode and version recorded | held, in both metadata and the profile identity |
| Raw text identical across modes | held |

The second one matters beyond tidiness: an absolute path in the embedded text puts the host's
filesystem layout into stored vectors, and makes one corpus embed differently on two machines.

### What landed

| Area | Change |
|---|---|
| Title | the frontmatter key must be top level; the basename fallback reads the whole path |
| Paths | `root_relative_source` refuses absolute, drive-lettered, UNC and traversing paths, **in every mode including `none`** — the mode is chosen by the profile, so a guard reachable only on the contextual path is one the cheapest caller skips |
| Caps | `TITLE_MAX_CHARS`, `SOURCE_MAX_CHARS`, `SECTION_MAX_CHARS`, `SECTION_DEGRADED_MAX_CHARS`, `NEIGHBOR_MAX_CHARS` replace literals at call sites |
| Ladder | `DEGRADATION_ORDER` is now the order the ladder is BUILT from, and `DOCUMENT_DEGRADATION_ORDER` is a derived suffix of it |
| Neighbours | folded to one line before the 200 is counted |
| Docs | a "Deterministic context modes" section in `ENTERPRISE_RETRIEVAL.md`, and a `CHANGELOG.md` entry |

### The audit found more than the session did, and four of its findings were mine

The commit went through the tiered CCA pipeline at DEEP (forced: `RUN_STAKES` and `RUN_NUM` both
fired on the diff). Ten auditors, 33 raw findings. **Deterministic coverage was NONE**: `cca_checks`
is not installed in this environment and is not on the package registry, so no static backend was
available and every verdict below rests on LLM adjudication plus my own re-execution. I verified
each P1 myself by running it rather than by reading the report.

| # | Finding | Verdict |
|---|---|---|
| 1 | The basename fallback split the guard's **truncated** return value. At a 264-character path the cut landed on a `/`, the basename was empty and the `title:` field vanished from the passage; at 261 the title was `not`, and two files in one deep directory got identical titles | CONFIRMED by execution, **fixed** |
| 2 | The guard validated `normalised` and returned `normalised[:256]`, so truncation ran after the checks and could MANUFACTURE a `..`: `"a" * 253 + "/..x"` passed the traversal check and came back ending `/..` | CONFIRMED by execution, **fixed** |
| 3 | `^[A-Za-z]:` refused `a:b/notes.md`, legal on Linux and macOS and exactly what `relative_to(root).as_posix()` produces; the refusal aborts a run whose earlier batches are already committed | CONFIRMED by execution, **fixed** |
| 4 | `DEGRADATION_ORDER` read as the ladder's specification and was an independent second literal. Renaming the ladder's labels left all 52 tests green | CONFIRMED by mutation, **fixed** |
| 5 | `metadata->>'content_hash'` is SQL NULL when the key is absent, so the content-hash arm of the load-bearing test compared NULL with NULL. Stripping `content_hash` from every stored row left the test **named for content hashes** green | CONFIRMED by execution against the database, **fixed** |
| 6 | An adjacent chunk containing `\nsource: /etc/shadow` put a second `source:` line into this chunk's rendered passage, and the neighbour cap was the only one counting un-normalised characters | CONFIRMED by execution, **fixed** (folded) |

The first three are one root cause: a cap applied after a check, in a function whose docstring
argued carefully about ordering control-character stripping against the traversal check and then
mutated its value after validating it. **The guard did not hold on its own output**, and it was
written in the same session that added the guard.

Findings 4 and 5 are the same class as each other and it is the class this repository keeps
recording: a check that compares an object with itself. The order constant agreed with the ladder
because both were transcribed from the same intention, not because anything bound them; the two
content-hash arms agreed because both were absent.

### The mutation sweep, twice, and what it caught that the first pass did not

Every mutation is applied and restored **by bytes** (`write_text` rewrites LF as CRLF on Windows,
which left ten files dirty for a previous session), and the harness refuses to report a result for
a mutation whose search string is not found exactly once.

| Pass | Result |
|---|---|
| First, 27 mutations | 17 killed, **8 survivors, 2 never applied** |
| After fixing the survivors, 38 mutations | **38 killed, 0 unapplied** |

The eight survivors were all mine, and five were the same defect:

* three cap tests asserted `len(field) == TITLE_MAX_CHARS`, so raising the constant raised both
  sides and the published cap could move while the test stayed green;
* the neighbour test asserted `previous == chunks[0][-200:]` against a fixture of `"A" * 1000`,
  whose first 200 characters equal its last 200 — the fixture could not tell a head from a tail;
* the `index_fingerprint` test passed for the wrong reason: each mode was indexed under its own
  profile, so the fingerprints differed whether or not the mode was part of them. It now holds one
  legacy embedder fixed across four policies, which is the only arrangement that attributes the
  difference to the mode.

And after the audit's fix to finding 4, the ladder test became tautological in a new way: the
ladder is now emitted FROM `DEGRADATION_ORDER`, so comparing what it emits against that constant
compares the code with itself. The sweep caught that too. The expected rungs are now written out.

### Decisions a reader should be able to reverse

1. **`root_relative_source` refuses rather than sanitises, and raising aborts the run.** Every
   caller in the package passes a root-relative path, so the refusal is unreachable in normal use.
   The alternative — skip and count the file like `vanished_before_read` — was not taken, because
   a source name the indexer cannot represent is a defect to see rather than a file to lose.
2. **The rendered `field: value` form is not a parsed format.** The chunk is interpolated verbatim
   because rule 5 requires it, so a document containing a line that looks like a field renders
   one. Structural fields and neighbour excerpts cannot forge a line; the chunk can. Folding the
   chunk would have been the alternative and it contradicts rule 5.
3. **The source cap truncates the HEAD of a long path**, so two paths sharing 256 characters render
   an identical `source:` field. Keeping the tail would disambiguate better (the leaf is what
   identifies a document). Not changed here: it is a behaviour change with no finding behind it.
4. **`index_fingerprint` deliberately depends on the mode** while nothing else stored does.

### Carried forward, unfixed and recorded rather than lost

**The one worth acting on: a dual-write re-index writes NOTHING to the shadow generation.**
`Indexer.index_path` reads its skip set from the ACTIVE store alone (`recall/index.py:426`) and
skips a file whose active fingerprint matches (`:459`), so `pending_shadow_chunks` stays empty and
`_flush` returns before appending an outbox event. Measured against real PostgreSQL: a second
dual-write run over the same corpus with the shadow emptied reported
`IndexStats(files=0, chunks=0, skipped=3)` with `active 15 shadow 0` and `pending []`. Nothing in
the return value, the logs or `pending_events()` distinguishes "the shadow is up to date" from "the
shadow received nothing", and `cutover`'s `_require_non_empty_shadow` only catches a TOTALLY empty
one — a shadow attached midway through a corpus is silently missing every file already indexed.
**This is pre-existing on master and outside this diff**, which is why it was not fixed here.

Also carried: the test role is a superuser in `docker-compose.yml` and in all three CI database
jobs, so `FORCE ROW LEVEL SECURITY` is inert and every `set_config('recall.tenant_id', …)` in the
suite is unfalsifiable there (this session added one negative control on `unprivileged_dsn` that
does establish it); `recall/index.py` still stores the absolute host path in the `source` column,
so the CHANGELOG's "filesystem layout in stored vectors" is fixed for the vectors and remains true
of that column; `structure_chunks`'s `text_start`/`text_end` are code-point offsets into the
frontmatter-stripped body but are stored beside the file path under names that do not say so; and
a further 12 P3 findings (naming, positional tuple reads, an unused import idiom) were reported
and deferred.

### Gates run

| Gate | Result |
|---|---|
| `ruff check .` | clean (0.15.22) |
| `mypy` | clean, 139 source files |
| `pytest -q`, before the audit fixes | 2412 passed, 36 skipped, 0 failed (12 m 10 s) |
| Mutation sweep | 38 of 38 killed, 0 unapplied |

The three `tests/test_bench_systems.py` failures the previous session recorded **did not occur
here**, which supports its diagnosis: they are triggered by `fastembed` being installed locally,
and this session's venv is a fresh `.[dev]` install without it. That is not a fix, only evidence
about the cause.

### Standing blockers

| Blocker | Kind | Effect | Change |
|---|---|---|---|
| **No latency reference host.** VPS2 has 12 cores under a permanent load average near 8 from unrelated live production. | External dependency. Do not work around it. | Latency is **PENDING**; promotion blocked on latency grounds. Quality and safety gates still run. | unchanged; this session measured nothing on VPS2 and cites no timing |
| **No production corpus.** | Open | Nothing may be claimed about enterprise-corpus behaviour. | unchanged |
| **No approved local generator confirmed.** | Open | The generator-neutral evidence path stays unexercised end to end. | unchanged |

### What the next session should start with

1. **The dual-write skip defect above.** It is the only item here that can silently produce a
   half-populated shadow generation and let a cutover promote it. It has a measured reproduction
   and no shipped workaround.
2. Backlog session 3, the outbox drain, is **done** (the 08-05 control-plane entry below). Session
   4, making the documentation true, is still open and still names the `RECALL_DSN` contradiction.
3. **Session 10's campaign is now unblocked**: the harness exists, every rule has a test, and every
   test has been shown able to fail. Which of the three modes retrieves best is still unmeasured,
   and this session deliberately measured nothing about it.

---

## 2026-08-05, the missing link: the promotion gate has a producer, and it refuses

Backlog session 10, taken ahead of session 4. Branch cut fresh from `origin/master` at `fa673e5`
into a separate worktree; `origin/master` moved to `c02645b` (PR #202, OIDC subject binding) while
this ran and the branch was merged onto it before the PR.

### Session ledger

| # | Item | Outcome |
|---|---|---|
| 1 | Frozen input manifests, on the ladder digest pattern | done, `recall/eval/promotion/manifest.py` |
| 2 | The per-question record schema, exactly the sixteen named fields | done and CLOSED — an extra field is refused, not ignored |
| 3 | Corpus adapters for all five named corpora, plus the in-repo labelled set | done; only `labelled` could be EXECUTED here, see below |
| 4 | Aggregator emitting the gate input and a machine-readable decision | done, `recall/eval/promotion/aggregate.py` |
| 5 | Profile-scoped calibration, failing closed on cross-profile reuse | done, `load_for_profile` extends `load_for` and does not weaken it |
| 6 | Pick ONE resume mechanism, factor it out, use it | done, 3 → 2; the third is deliberately left, with the reason recorded |
| 7 | Latency PENDING must block promotion rather than pass silently | done, `latency_p95_ms: float \| None`; PENDING is a failure |
| 8 | Prove it end to end on a small corpus with baseline == candidate | done, `results/promotion/decision.null-difference.json` |
| 9 | Prove every new test can fail | done, 29 of 29 mutations turned their named test red |
| 10 | Audit this session's own work | done, DEEP tier, 13 fixes, 10 of 10 red→green proofs verified |

97 new tests across six files. `ruff check .` clean on the CI-pinned ruff (local 0.16 reports 420
errors on untouched code; the pin is `>=0.5,<0.16`, and 0.15.22 is clean). `mypy` clean on the new
modules.

### What landed

| Area | Change |
|---|---|
| Frozen manifests | `manifest.py`, the same shape as `benchmarks/ladder/manifest.py` rather than a second one: sorted canonical rendering, digest over body AND provenance, digest excluded from what it covers, reader refuses a mismatch. `freeze` refuses to overwrite an existing manifest |
| Record schema | `records.py`, sixteen fields, closed. `output_hash` covers the retrieval output and the configuration identity but **deliberately not `stage_timings`** — timings are not reproducible, so a hash that moved with the clock could not answer "did these two arms return the same thing?", which is the question the null-difference proof needs |
| Adapters | `labelled`, `peps`, `locomo`, `ladder`, `longmemeval`, `mtrag`. `recall/eval/peps_questions.json` was in the tree and referenced by no code; it is wired now. MT-RAG becomes one corpus per domain, because this repo's macro average is the UNWEIGHTED mean over corpora and its domains differ in judged-query count by nearly a factor of two (the AUD-1 finding) |
| Aggregator | `QuestionOutcome`/`SafetyMetrics`/`RetrievalGateInput`, `PromotionDecision` JSON. Primary hit@5 and MRR; hit@1, hit@20, nDCG@5, false confidence, false abstention and latency recorded beside the decision and labelled `secondary_diagnostic_only` |
| Calibration | `save_for_profile`/`load_for_profile` in `recall/calibration.py`, keyed on `EmbeddingProfile.fingerprint()`. `ArmConfig.artifact_stem` puts the fingerprint in every ledger filename |
| Resume | `recall/eval/resume.py`; `benchmarks/ladder/run.py` and `benchmarks/beam/run.py` both delegate to it |
| Latency | `RetrievalGateInput.latency_p95_ms` accepts `None` = PENDING, and PENDING FAILS |

### The four refusals, and why each one exists

The aggregator mostly refuses to build a gate input. Each refusal closes a way the gate could pass
without having been asked anything.

| Refusal | What it prevents |
|---|---|
| Arms that do not cover the frozen manifest exactly, or whose rows carry a different `input_hash` | A paired test over a set that quietly shrank between the arms is not the test that was declared |
| Safety metrics that are NaN because their class is empty | `recall/eval/metrics.py` returns NaN on an empty class, which is right for publishing and wrong for a gate: the gate compares with `>`, every comparison against NaN is False, so an empty safety check reads EXACTLY like a passed one |
| Unanswerable questions in the paired quality set | hit@5 on a question with no relevant document is 0.0 for every system that has ever existed, so including them dilutes the delta toward zero by an amount that depends only on how many were included. They score on the safety axis, which is the axis they were written for |
| An arm that retrieves no labelled document for ANY question | See below. This one was written because it happened |

### What was measured

**The gate's `promoted=True` branch executed for the first time.** The gap matrix recorded that
`tests/test_promotion.py` held one test asserting `not decision.promoted`, and that "a gate that has
only ever been shown to fail is not evidence that it can pass; it is compatible with a gate that
refuses everything." It is not: a fixture improving 4/20 to 18/20 on two corpora, with green
security and a measured p95 under budget, returns `promoted=True` with no failures. Every refusal
below is then flipped off THAT configuration, one condition at a time, so each test shows its own
branch firing rather than sharing a fixture that would have failed anyway.

**End to end on the labelled corpus, baseline == candidate.** 25 questions (14 answerable, 11
unanswerable), 22 documents, `bge-small-symmetric-v1`, both arms the same configuration. All 25 rows
share an `output_hash` across the two arms, so the difference is null by construction rather than by
measurement. The gate refused on five counts:

| Failure | |
|---|---|
| macro paired hit@5 bootstrap interval does not clear zero | delta exactly 0.0, interval [0.0, 0.0] |
| no improving corpus has Holm corrected paired significance below 0.05 | Holm p = 1.0 |
| **superseded trust rate was NOT MEASURED** | both arms degraded; `superseded_trust_rate: null` |
| security verification is not green | `--security-green` defaults to False; unverified is not green |
| **retrieval p95 latency is PENDING** | `gate_input_p95_ms: null` |

Removing only the PENDING condition from an otherwise-promotable fixture leaves exactly one failure,
which is how the test shows that PENDING is what fired and not something else about the fixture.

The third row is the one this session's audit added, and it is the more consequential of the two
"not measured" failures. `recall/trust.py` overwrites **every** verdict with `unverified` in
degraded mode, *after* `evaluate` has computed the real one, so a superseded hit and a clean one
leave identical rows. The first version of this harness therefore computed
`superseded_trust_rate = 0.0` for a degraded arm and **satisfied** the gate's zero-tolerance check
by never having measured it. It is now `None`, which blocks, on the same pattern as a PENDING
latency: an unmeasured input produces a decision that names what was missing rather than a
traceback or a pass.

**Two defects that only a real run could surface.** The harness was written, unit-tested green, and
then run for the first time against a live store:

1. `score_arm` never passed `input_hash` into `build_record`. Caught by the adapter tests, before
   the live run.
2. **The labelled corpus scored 0 of 14, and nothing raised.** Its labels are `cache_decision.md:0`;
   `PgVectorStore` chunk ids are content hashes. Every comparison was a miss, both arms were equally
   wrong, and the gate computed a delta of exactly zero and produced a clean-looking refusal. The
   numbers were meaningless and the artifact would not have said so.

   The fix is `search.LABEL_KEYS`: five declared id spaces, `label_kind` required on every adapter
   with **no inherited default**, since a default is how a new adapter inherits a comparison that is
   silently wrong for its corpus. The detector is `aggregate.VacuousArm`, because a fix with no
   detector only covers the mistake that was already made. Both are mutation-proven.

### The audit of this session's own work, and what it found

The commit went through the tiered CCA pipeline at DEEP (forced: `RUN_NUM` and `RUN_STAKES` both
fire on a diff full of rates and thresholds). Four auditors, 36 raw findings, 13 fixed. Every
deterministic backend was available for Python (`definedness`, `nullability`, `taint`, `type`,
`clock_leak`).

**Six of the findings are the same failure class this harness was built to refuse, found in the
harness.** Each was proven by executing the shipped code, not by reading it:

| Finding | The gate could... |
|---|---|
| `cmd_decide` narrowed the frozen manifest to the corpora present in the BASELINE ledger | ...return `promoted=true` on a question set that shrank after the freeze, stamped with the full manifest's digest. Measured: the same two-corpus manifest returned `promoted=false` citing a regression on the second corpus, and `promoted=true` with that corpus absent from both ledgers |
| `latency_p95_ms` was checked only against `None` | ...pass the budget on a NaN, because `nan > budget` is False, and report `latency_status: MEASURED`. NaN is reachable from this package's own `_percentile` over an empty sample |
| `record_from_dict` copied `output_hash` instead of recomputing it | ...be flipped by editing ten rows of a ledger. The manifest was tamper-evident; the EVIDENCE was not. Measured: `promoted=false` → `promoted=true` |
| a wholly degraded arm computed `superseded_trust_rate = 0.0` | ...satisfy a zero-tolerance safety check by never having measured it |
| `artifact_stem` omitted `retrieval_profile`, `generation`, `candidate_pool` | ...compare a ledger with itself: two arms differing only in `--retrieval-profile` produced the identical filename, the second resumed the first's rows and scored nothing |
| `--no-resume` read nothing but still appended | ...gate on the rows the run was told to discard, since `read_ledger` keeps the FIRST occurrence of an id |

Two of those sat directly under docstrings claiming the opposite. `artifact_stem` said it carried
"the COMPLETE immutable identity" while carrying three of six fields, and the `reranking_status`
comment said a configured-but-failed reranker "is visible here" when `HybridRetriever` sets
`reranking_ran` *before* calling the reranker, making `"failed"` unreachable. The first is fixed;
the second is a comment correction, because making it observable needs a change in the retriever
and that is a different diff. **A comment that claims more than the code is the same defect as a
guard that cannot fire, and this session produced both while writing about both.**

Also fixed: `--corpus mtrag` matched nothing (the adapter stamps `mtrag:<domain>`, the CLI filtered
on equality, so a wired corpus was unrunnable); `ArmConfig.generation` was recorded on every row
and never passed to `trusted_search`, so the evidence named a generation the search never used;
`secondary_metrics` pooled across corpora while the headline is macro (a factor-2.2 disagreement
inside one document, the AUD-1 finding again); `write_decision` allowed a bare `NaN` token into a
committed artifact; the decision named its arms only by the free-text label a human typed; and
`--dsn` was required, putting the database password into argv.

Ten red→green proofs, each verified against the pre-fix code by
`cca_tautology_check verify`: **10 of 10 RED**. Each carries a control test asserting the
behaviour that did NOT change, because a refusal that swallows the passing case is not a fix.

**The anti-regression pass then found two more, in the fixes themselves, and both are the same
lesson twice.**

The `--dsn` fix moved the credential out of argv and deleted `required=True` without replacing the
guarantee it carried. `PgVectorStore` does not validate a DSN, so `None` reached psycopg as an
`AttributeError` *after* a model load, and an EMPTY string — the ordinary shape of an unset systemd
or CI variable — is a valid libpq conninfo meaning "use the local defaults". On a peer-auth host
that connects to the operator's own database, where the command then creates, indexes and drops a
table. **Removing a check because its side effect was undesirable removed the check.**

The MT-RAG fix shipped with a test that could not fail. It re-implemented the `startswith`
expression inline against the adapter instead of calling the CLI's filter, so it passed with the
fix reverted — a guard that cannot fire, inside the fix for a filter that matched nothing. The
comprehension is now the named `scope_questions`, the test calls it, and a mutation reverting the
filter turns it red. It is the fifth instance of a non-discriminating assertion recorded here, and
the first one this program caught in its own audit fixes rather than in the code under audit.

⚠️ **The harness that first checked those three mutations reported two of them RED when the tests
did not exist at all** — the edit that was supposed to add them had silently not applied, and a
non-zero pytest exit for "no tests ran" is indistinguishable from one for "the test failed" if you
only read the exit code. That is the repository's own standing rule about reading an exit code
through a pipe, in a new costume. The re-check parses `N failed` out of pytest's summary and
reports `INCONCLUSIVE` otherwise; all three are RED under it.

**Mutation sweep: 29 of 29.** Each new guard was disabled in the source and its named test asserted
to go red. One survived the first pass, and it is the instructive one: deleting the
absent-fingerprint branch in `load_for_profile` left the identity-equality check below it, which
also returns None, so a test asserting only `is None` passed with the branch removed. The branch's
real contribution is the DIAGNOSIS — "this file records no `profile_fingerprint`" is an actionable
instruction, while "it was fitted under identity None" describes nothing that exists — so the test
now asserts the warning. That is the fourth instance of a non-discriminating assertion recorded in
this repository.

**Resume: three mechanisms became two, not four.** `benchmarks/ladder/run.py::_recorded` and
`benchmarks/beam/run.py::_already_done` now both delegate to `recall/eval/resume.py`. The ladder's
behaviour is unchanged. Beam gains one behaviour deliberately: a truncated final line used to raise
`JSONDecodeError` there and now counts as "not yet scored". These sidecars are read after an
interruption, so a torn tail was the one malformed line that function was guaranteed to meet, and
refusing to resume from it meant re-paying for every question in the file. `recall/eval/gap_run.py`
is **not** migrated, and the reason is recorded in `resume.py`'s own docstring rather than left to
be rediscovered as "a fourth mechanism appeared": its unit of resume is a whole corpus result FILE
carrying a `status` marker, its artifacts under `results/gap/` are committed in that shape, and
unifying it would rewrite a published artifact format for no resume benefit.

### What is NOT proven, stated rather than implied

- **Only `labelled` was executed.** It is the one corpus whose documents ship in the repository.
  `peps`, `locomo`, `longmemeval` and `mtrag` are wired and unit-tested against synthetic fixtures
  in their own formats, but their `label_kind` mappings have **not** been confirmed against a real
  index. `ladder`'s reader is exercised against a manifest written by the ladder's own writer,
  including the digest refusal, but not against a live retrieval. Each mapping was derived by
  reading the code that writes those ids, and each is named in `search.py`'s table with the function
  it mirrors. The vacuity refusal is what turns a wrong one into a loud failure instead of a zero.
- **The proof run is DEGRADED.** A plain store has no generation-bound certified calibration, so
  strict trust mode refuses it — correctly. The run used `--trust-policy development`, every verdict
  is `unverified`, and the artifact records `trust_verdicts: {"unverified": 25}` for both arms so
  that its `false_confidence: 1.00` cannot be read as a property of this library's abstention. The
  two arms stay comparable to each other; neither is comparable to a trusted run. **Wiring a
  strict-mode gate run needs a generation-bound certified calibration end to end, and no session has
  done that.**
- **`--trust-policy` defaults to strict** rather than to the mode that would have made this session's
  run easier. A harness defaulting to development mode would silently score a degraded system and
  publish the numbers as if the trust gate had run.
- **No campaign verdict was produced, and none was attempted.** This session built the producer.

### Standing blockers

| Blocker | Kind | Effect | Change |
|---|---|---|---|
| **No latency reference host.** VPS2 has 12 cores under a permanent load average near 8 from unrelated live production. | External dependency. Do not work around it. | Latency is **PENDING**; promotion blocked on latency grounds. Quality and safety gates still run. | **now enforced in code**: `latency_p95_ms=None` is a gate FAILURE, and the harness will not supply a number without `--certified-latency-p95-ms`. Nothing was measured on VPS2 this session |
| **No production corpus.** | Open | Nothing may be claimed about enterprise-corpus behaviour. | unchanged |
| **No approved local generator confirmed.** | Open | The generator-neutral evidence path stays unexercised end to end. | unchanged |
| **No generation-bound certified calibration has ever been produced end to end.** | Open, NEW | Every gate run is degraded, so `superseded_trust_rate` is **NOT MEASURED** and blocks promotion on its own. | newly identified this session, and **enforced in code** after the audit: a degraded arm reports `None`, not `0.0` |

### What the next session should start with

Backlog session 4, making the documentation true, which the previous entry named and which is still
first. It was not taken here because the goal was the producer; nothing in this session touched the
`RECALL_DSN` contradiction, and the sharpest consequence recorded there still stands: a green
`recall-enterprise readiness` run certifies the credential the CLI connected as, which the enterprise
doc tells the operator to make the MIGRATION role.

Then, in the promotion lane specifically, and in this order:

1. **A generation-bound certified calibration, end to end.** This is now a hard blocker, not a
   quality concern: `superseded_trust_rate` comes back NOT MEASURED for a degraded arm and fails
   the gate by itself, so **no promotion decision can be green until a trusted run is possible**.
   `recall/trust.py` overwrites every verdict with `unverified` in degraded mode, so the
   information is destroyed at the source and no amount of care in the harness recovers it. The
   path is `CalibrationRepository.calibrate` → `publish` against a built generation; nothing wires
   it end to end today.
2. **Confirm one external corpus's `label_kind` against a live index** — `locomo` is the cheapest,
   since `recall/eval/locomo.py` already writes the documents. One confirmed mapping turns four
   derived-by-reading claims into three.
3. Only then a campaign with a real candidate arm.

Carried forward unchanged: the MT-RAG baseline still has no `results/ARTIFACTS.md` row
(deliberately), and `bench/mtrag-symmetric-baseline` is still local and unpushed.

---

## 2026-08-05, control plane: the untested and unreachable surfaces closed

Backlog session 3, plus the parts of sessions 6, 7 and 11 that could not be separated from it.
Branch cut fresh from `origin/master` at `061c810`.

### Session ledger

| # | Item | Outcome |
|---|---|---|
| 1 | Five missing operator subcommands: `replay`, `parity`, `readiness`, `status`, `retire` | done, all five driven through `main()` in tests |
| 2 | The eleven behaviours in the brief, enforced and tested | done, see the table below |
| 3 | Readiness fails for every declared condition; uncertified calibration reports degraded abstention | done, 31 collected tests over branches that had **zero** |
| 4 | Two-ledger question resolved per session 2's decision | done, all three follow-ups executed |
| 5 | Integration suite against real PostgreSQL as a non-superuser | done, 36 tests |
| 6 | Prove the RLS tests can fail | done, and it caught a defect in one of my own tests |

81 new tests across three files: 36 integration, 31 readiness, 14 CLI. `ruff check .` clean on
the CI-pinned ruff (`>=0.5,<0.16`), `mypy` clean across 138 source files.

### The working tree was not mine, and that changed where the work happened

`C:/…/eva/work/RE-call` held **another session's live uncommitted work**: `recall/cache.py`,
`recall/context.py` and `recall/embeddings.py` modified, plus untracked
`recall/embedding_registry.py`, `tests/test_embedding_cache_identity.py` and
`tests/test_embedding_profile_registry.py`, all timestamped minutes before this session started.
That is backlog session 8 (profile registries, `cache_key` fields) in progress.

Branching in that clone moved their uncommitted changes onto my branch. It was reverted
immediately: the clone is back on `codex/embedding-profile-registry` with all six files exactly as
they were, and this session ran in a separate `git worktree`
(`…/RE-call-enterprise-cp`). Nothing of theirs was committed, stashed or deleted. The next session
should check `git worktree list` and `git status` in the primary clone before branching there.

### What landed

| Area | Change |
|---|---|
| Operator CLI | `replay`, `parity`, `readiness`, `status`, `retire`. `replay` opens only the generations the pending events name; `status` prints operation ids and counts and never a payload |
| Identifier allowlist | `validate_table_name` was `str.isidentifier()`; now `^[a-z_][a-z0-9_]{0,45}$`, a 46-byte ceiling set by the longest derived suffix (`_tenant_isolation`, 17 bytes) rather than by the 63-byte identifier limit. `delete_sources_across` had a **second, weaker** copy of the check and now calls the same one |
| Retired generations | `StoreRegistry` refuses a generation outside the servable set, per request, and the operator CLI refuses one too (`replay` writes through that path). The ACTIVE slot uses `SERVABLE_ACTIVE_STATES` = `{ready, active}`, matching `set_route`'s own gate; the SHADOW slot also allows `building` |
| Erasure | `forget_memory` scrubs erased sources out of pending outbox payloads, keyed on the sources the CALLER named, and reports `outbox_events_scrubbed`. An event left with no sources is completed rather than left to block `cutover` |
| Readiness | Verifies **both** ledgers; an unreachable control plane is a failure rather than a traceback; a retired active generation fails; the `calibration` argument is passed again from `recall_mcp/server.py` |
| Control-plane migrator | Advisory lock `recall-control-plane-migrations-v1`, refusing rather than waiting |
| Docs | `docs/MIGRATIONS.md` documents the second ledger; `docs/ENTERPRISE_RETRIEVAL.md` documents the new commands, the retirement rule, and corrects "polling fallback" to what it is, a cache TTL |

### The behaviours from the brief, and where each is enforced

| Behaviour | Test |
|---|---|
| Physical tables come only from validated registry rows, matching the allowlist | `test_the_physical_table_allowlist_refuses_what_isidentifier_allowed`, `test_delete_sources_across_uses_the_same_allowlist` |
| No client parameter can select a generation or table | `test_no_client_parameter_can_name_a_generation_or_a_table`, `test_no_subcommand_accepts_a_physical_table_except_create_generation` |
| Completed event payloads are cleared | `test_replay_converges_both_generations_after_a_crash` |
| Audit records keep id, timestamp, status, counts, never corpus text or source paths | `test_the_retained_audit_record_holds_counts_and_no_corpus`, asserted column by column from `information_schema` |
| Both embeddings prepared before either generation is written | `test_dual_write_reaches_both_generations_and_leaves_no_pending_event` |
| Forget deletes from active and shadow in ONE transaction | `test_erasure_across_generations_is_atomic_when_the_second_table_fails` |
| A failed shadow write allows the active write only with a durable pending event | `test_a_failed_shadow_write_leaves_a_durable_pending_event` |
| Cutover refused while any event is pending | `test_cutover_is_refused_while_an_event_is_pending` |
| Route cache keyed on tenant and generation | `test_the_route_cache_is_keyed_on_tenant_and_generation` |
| LISTEN, with a 5 s fallback | `test_listen_delivers_a_route_change_and_the_ttl_bounds_a_missed_one` |
| In-flight searches finish on their generation, new ones use the new route | `test_a_route_change_during_a_search_leaves_the_acquired_store_alone` |

### What was measured

**The suite had been asserting tenant isolation as a superuser, and five tests depended on it.**
`docker-compose.yml` ships `POSTGRES_USER=recall`, which is the cluster superuser, and row level
security is inert for one. Pointed at an unprivileged role for the first time, the suite went from
0 to 7 failures, and the same defect class accounted for four of them: **a verification read
issued without the `recall.tenant_id` GUC, which under FORCE RLS returns nothing and is then
interpreted as a fact about the data.**

| Test | What the missing GUC did | Fix |
|---|---|---|
| `test_enterprise_control_plane` teardown | deleted 0 routes, reported success, then failed the generation delete on a foreign key | set the GUC |
| `test_enterprise_control_plane` body | `SELECT payload …` returned NO ROW, and `payload is None` was read as "the payload was cleared" | set the GUC, and assert the row is visible before asserting its contents |
| `test_v08_table_is_adopted_without_rewriting_existing_data` | `fetchone()` returned None; the assertion failed with a `TypeError` that says nothing about tenancy | set the GUC |
| `test_temporal_adapter_puts_the_intervals_where_recall_reads_them` | read 0 rows and reported "the adapter wrote nothing" when the adapter wrote fine | set the GUC, to tenant `temporal` rather than `default` |

The second row is the one to remember: the assertion passed **for the wrong reason** on a
superuser, because "the payload was cleared" and "the policy hid the record" are the same
observation when you only look for `None`.

The fifth, `test_serving_role_has_dml_but_cannot_run_ddl`, genuinely needs `CREATEROLE`, since its
subject is a boundary between two roles. It now skips when the configured role cannot provision
one, and is unchanged on any DSN that can.

`tests/conftest.py` grew an `unprivileged_dsn` fixture that either uses an already-unprivileged
`RECALL_TEST_DSN` or provisions `recall_rls_probe` (`NOSUPERUSER NOBYPASSRLS`) from a privileged
one, and **skips** rather than falling back if it can do neither. Both branches were executed.

**The RLS mutation proof, and what it caught.** `ALTER TABLE … NO FORCE ROW LEVEL SECURITY` on the
control-plane tables, then on the per-test chunk tables through a pytest plugin that fails the run
if the mutation does not apply.

| Arm | Before | Mutated | Restored |
|---|---|---|---|
| Control plane (`recall_tenant_routes`, `recall_migration_events`) | pass | **fail** | pass |
| Chunk tables (`c_active_*`, `c_shadow_*`) | pass | **fail** | pass |

The chunk-table arm is the one worth recording, because the first version of that test **passed
under the mutation**. `PgVectorStore.count()` carries `WHERE tenant_id = %s`
(`recall/store.py:1776`), so asserting through the store measured the query PREDICATE while
claiming to measure the POLICY. The test now issues an unpredicated `SELECT count(*)` under
another tenant's GUC, which only the policy can answer, and it also asserts the same connection
DOES see the rows once it names the owning tenant, so "returned zero" cannot be explained by an
empty table. Under the mutation it now fails on exactly that assertion.

**Symlink confinement was verified on Linux, because the assertion skips on Windows.** A skipped
test is not coverage, so this branch's `recall/index.py` (sha256 `60653f85586f…2909`) was run on
VPS2 under Python 3.12: the unconfined glob reached one path resolving outside the root
(`corpus/direct.md` to `/tmp/…/outside/secret.md`) and `candidate_files` dropped it. Root SSH was
used rather than qwen-mcp, whose file roots do not cover this program. The first attempt at the
negative control asserted the wrong thing (it looked for the TARGET's name, while the escape
appears under the LINK's name) and failed loudly, which is why the control is now in the test
itself: without it, an implementation with no confinement at all would produce the same answer on
a platform whose glob never reaches the escape. Note for the next reader: on Python 3.12.3
`Path.glob("**/*.md")` did **not** recurse into the directory symlink, contradicting the version
range in `_confined_to`'s docstring. The file-symlink escape is what reproduces.

### The audit of this session's own work, and what it found

The commit went through the tiered CCA pipeline at DEEP (forced: the diff touches erasure, tenant
isolation and migration). Ten auditors, 89 raw findings, 14 fixed in `bb24ab8`, the rest reported.
`cca_checks` IS installed in this checkout, so `python`'s deterministic backend was available
(definedness, nullability, taint, type, clock_leak) and one finding carries a `hypothesis`
artifact rather than an LLM verdict.

**Four auditors independently found the same Critical, in code written and reviewed earlier the
same session.** `forget_memory` gated the outbox scrub on `to_delete`, which resolves from the
CHUNK TABLES, so the scrub could not fire in the one state it was written for: a crash between
`append_event` and the two writes leaves the payload as the only copy with both tables empty.
Erasure answered "no matching source(s) found" and the next replay wrote the text back into both
generations.

The mutation proof is the part to carry forward:

| | the new test | the pre-existing erasure test |
|---|---|---|
| pre-fix gate restored | **FAILS** | **passes** |
| fixed | passes | passes |

The old test seeded both chunk tables before creating the crash-shaped event, so `to_delete` was
never empty and it passed with or without the gate. It is the balanced fixture that cannot
discriminate, which is why the defect survived review, and it is the third instance of that class
recorded in this repository.

**The second Critical was a comment.** `erase_sources_from_pending` claimed its `FOR UPDATE`
excluded a concurrent `replay_pending`. It cannot: `pending_events` is a plain SELECT, and under
READ COMMITTED a plain SELECT is never blocked by a row lock. A guard that read as protection and
could not fire, written the same day as the entry above warning about exactly that. Both paths now
take a shared per-tenant advisory lock.

Also fixed: `retire_generation` skipped its only check when the tenant had no route row (a typo'd
`--tenant` retired unconditionally and printed success); `enterprise_cli._open` had no state check,
so `replay` would WRITE into a retired generation, which is precisely what `retire_generation`
delegated its safety to; the identifier bound was 63 when derived names add up to 17 bytes, so a
legal table produced a truncated index and readiness then reported a missing index for one that
exists (confirmed by a falsifying example at n=47); a control plane ahead of the package was fatal,
which would refuse every older replica after a forward migration; and readiness reads
`recall_schema_versions` with the serving credential while no GRANT for it shipped anywhere, so a
least-privilege install would not have booted.

**Three comments in the first commit claimed more than the code did, and were corrected rather
than kept.** The calibration identity branch is still unreachable from the server, because
`load_for` filters on the embedder and stamps its argument onto what it returns;
`validate_table_name` is not yet the single chokepoint, since `PgVectorStore.__init__` and
`recall.schema._validate_target` still use `str.isidentifier()`; and the allowlist does not refuse
unquoted SQL keywords. Tightening the two remaining gates is deliberately deferred: it changes
behaviour for tables that already exist and needs a compatibility decision this change does not
make.

### Corrections to the inherited snapshot

- **`check_enterprise_readiness`'s calibration branch is now reachable, not deleted.** The gap
  matrix left this as a decision. The argument is supplied again from `recall_mcp/server.py:336`;
  `load_for` returns None for a calibration belonging to another embedder, so the warning now
  means "there is none", which is actionable, rather than "this call site does not pass one",
  which was not. The identity-mismatch failure is exercised by
  `test_a_calibration_for_a_different_embedder_is_a_failure_not_a_warning`.
- **Erasure reaching the outbox was implemented, not only decided.** The gap matrix framed it as a
  policy question. The policy chosen is scrub, not discard: one event covers a batch of sources
  and only some are erased, so the erased records are removed from the payload and an event left
  with nothing to do is completed. Discarding the whole event would have lost the surviving
  sources' shadow writes; leaving it pending would have deadlocked `cutover` on an operation that
  must never be replayed.
- **`docs/ENTERPRISE_RETRIEVAL.md:49` said "five second polling fallback".** It is a route cache
  TTL, not a poll. Corrected here rather than deferred to session 7, because the new
  `test_listen_delivers_a_route_change_and_the_ttl_bounds_a_missed_one` would otherwise have been
  documented as testing something that does not exist.

### Three pre-existing failures left alone, and why

`tests/test_bench_systems.py`'s three DSN-gated tests fail with `TrustRefusal: INDEX_NOT_READY`.
They fail identically on `origin/master` at `061c810` with no changes applied, under **both** an
unprivileged and a superuser DSN, so this is not a privilege effect and not a regression from this
work. The trigger appears to be `fastembed` being installed locally, which CI's `test` job
deliberately does not install, so the tests take a different embedder path than CI exercises.
Diagnosing that is its own change, and it is outside this session's scope.

Full-suite baseline, both runs on the same unprivileged DSN: `origin/master` at `061c810` failed
exactly the same 7 tests this branch initially failed. Four of the seven are fixed here, one now
skips with a stated reason, and three are the `test_bench_systems` cases above.

### Standing blockers

| Blocker | Kind | Effect | Change |
|---|---|---|---|
| **No latency reference host.** VPS2 has 12 cores under a permanent load average near 8 from unrelated live production. | External dependency. Do not work around it. | Latency is **PENDING**; promotion blocked on latency grounds. Quality and safety gates still run. | unchanged; this session measured nothing on VPS2 |
| **No production corpus.** | Open | Nothing may be claimed about enterprise-corpus behaviour. | unchanged |
| **No approved local generator confirmed.** | Open | The generator-neutral evidence path stays unexercised end to end. | unchanged |

### What the next session should start with

Backlog session 4, making the documentation true, and it is now **more** urgent than it was, not
less. Two of its five entries were done here (the second ledger, and the `ENTERPRISE_RETRIEVAL`
route-fallback wording), but the audit showed this session **widened** the `RECALL_DSN`
contradiction rather than leaving it alone: `docs/MIGRATIONS.md:46` now documents `RECALL_DSN` as
the credential that applies control-plane migrations, 34 lines below line 12 calling it the
deprecated fallback for the SERVING DSN, so the contradiction moved inside one file. Five new
subcommands hang off `enterprise_cli._dsn()`, which reads only `RECALL_DSN`, and four of them
(`readiness`, `status`, `parity`, `replay`) are serving-side reads that need no DDL privilege.

The sharpest consequence, worth fixing first: `recall-enterprise readiness <tenant>` reports "row
level security is ineffective for the runtime database role" based on the role the CLI connected
as, which the enterprise doc tells the operator to make the MIGRATION role. A green readiness run
currently certifies a credential that may never serve a request.

Then the rest of session 4: `README.md:628-632` contradicting `README.md:205`, the `CHANGELOG.md`
entry (every comparable feature commit added one; this one did not), and the Qwen3 record.

Carried forward from the audit, unfixed and recorded rather than lost: the two remaining weak
identifier gates (`PgVectorStore.__init__`, `recall.schema._validate_target`); no scan or preflight
for registry rows that the tightened allowlist would now reject at read time, which would fail a
boot and also crash the `status` command meant to diagnose it; `erase_sources_from_pending` does
not branch on `operation_kind`, so a `forget`-kind event would be voided without executing its
delete (latent, no producer exists); the test probe role is left behind on the target cluster with
no teardown; and CI runs the new enterprise tests on PostgreSQL 16 only, while the matrix job that
covers 17 has a fixed file list that was not extended.

After that, backlog session 10: the promotion gate has still only ever been observed to fail.

Two items carried forward and still open: the MT-RAG baseline has no `results/ARTIFACTS.md` row
(deliberately), and `bench/mtrag-symmetric-baseline` is still local and unpushed.

---

---
## 2026-08-05, embedding profile identity: one registry, a cache that cannot alias, the Qwen rejection published

### Session ledger

| # | Item | Outcome |
|---|---|---|
| 1 | Collapse the two profile vocabularies into one registry | done, `recall/embedding_registry.py`; both dict literals are gone |
| 2 | Asymmetric semantics, with the legacy `embed()`-only fallback preserved | done, the declared encoder mode is now the dispatch key |
| 3 | Cache keyed on the complete immutable identity | done, `EmbeddingProfile.fingerprint`; `cache_key` refuses a bare profile ID |
| 4 | Offline enforcement, proven rather than asserted | done, unit tests plus a real-loader run on VPS2 |
| 5 | Preserve and publish the Qwen rejection | done, registry record plus `ENTERPRISE_RETRIEVAL.md`, tied together by a test |
| 6 | Prove every new test can fail | done, 67 of 67; 66 by mutation here, 1 on VPS2 because it skips on Windows |

This is backlog session 8 (items 22, 23, 24), plus item 9 from session 4 and the
symlink-escape half of item 29, taken ahead of session 3.
The previous entry names session 3, the outbox drain, as what to start with. It is untouched and
still first in the backlog.

### Two other sessions landed work while this one ran

`origin/master` moved four times during this session: #192 and #191/#190 (dependabot), then
**#197** (OIDC wiring, 22 CCA audit fixes) and **#196** (erasure and control-plane fixes). #196 and
#197 together touched 26 files, four of which this session also touched: `.env.example`,
`docs/ENTERPRISE_RETRIEVAL.md`, `recall/generations.py` and `tests/test_generations.py`.

The branch was rebased onto `cd0cbe0`. Three of the four auto-merged; `tests/test_generations.py`
was an add/add conflict at the tail, where both sides appended tests. It was resolved by keeping
**both**, upstream's first, and the resolution script asserts every test function from each side is
present by name in the result and that no conflict marker survives, rather than trusting a reading
of the diff. Nine upstream tests and one of this session's are all present.

Everything below was re-verified on the rebased tree, not on the tree the work was written against.

### What landed

**One registry.** `recall/embedding_registry.py` owns `profile_id`, `model_name`, `dimension`,
`query_mode`, `passage_mode`, `context_mode`, `normalization`, `instruction_version`,
`chunker_version`, the artifact digest and the backend, for all six registered IDs. The two
literals it replaces were `recall_mcp/service.py:113-120` (profile to context version, six entries)
and `recall/context.py:37-41` (profile to context mode, three entries plus a silent default).

Two properties do the work, and both are tested:

* `context_version` is **derived** from `context_mode` rather than declared next to it, matching
  what `Indexer.__init__` already enforces. The two values cannot disagree because there is only
  one of them.
* A profile added to the registry needs no second edit. The test adds a seventh profile to the
  registry alone and asserts that both `context_policy_for_profile` and `make_embedder` already
  know it. That test is red against the old arrangement, which is what makes it a guard rather
  than a description.

`make_embedder` is now environment parsing only. `RegisteredProfile.build` is the single
construction site, and `RegisteredProfile.identity` the single constructor of a runtime
`EmbeddingProfile` for a registered ID.

**Asymmetric semantics.** `query_mode` and `passage_mode` are dispatch keys handed to the backend,
not documentation of one: `FastEmbedEmbedder` resolves `getattr(model, mode)` and refuses a mode
the backend does not have, rather than falling back to the symmetric encoder. Dimension discovery
goes through the passage encoder, and a probed width that contradicts the registry refuses
startup. `GenerationManager.build` was still indexing with `embedder.embed`; it now uses
`embed_passages`. The `Embedder` protocol is unchanged and an `embed()`-only embedder still works
through both helpers, which six tests pin.

**Cache identity.** `EmbeddingProfile.fingerprint()` is a SHA256 over the complete identity, and
`cache_key` now takes the profile rather than its ID. The signature deliberately refuses a bare
string: the old call passed the profile ID, so the unsafe call is the one that used to be correct,
and a `str | EmbeddingProfile` union would have let every existing caller keep the weaker key.

This answers backlog item 24 as a side effect. `normalization`, `instruction_version`,
`chunker_version` and `dependencies` were read by nothing; they are now key material, so a change
in any of them re-partitions the cache instead of silently serving vectors produced under the old
value.

⚠️ **The change invalidates every existing embedding cache entry.** Old entries were keyed by
profile ID and are simply not found. The cost is one re-embed of whatever was cached; nothing is
served under a key that no longer describes it.

**Offline enforcement.** Artifact verification moved ahead of the backend import, so a missing or
tampered tree fails the same way whether or not the optional extra is installed. Startup is proven
to complete with connections refused, and to refuse on a missing artifact, a checksum mismatch, a
malformed digest, an empty tree and a symlink escaping the artifact root.

**The Qwen rejection.** Registered, marked rejected, and carrying the measurement that decided it.
`RejectionRecord` refuses to be constructed without one, because a verdict with no number is an
opinion the next session re-litigates. A test asserts `ENTERPRISE_RETRIEVAL.md` carries every
number the registry carries, so neither copy can be deleted quietly.

### What was measured

**The real loader, offline, on VPS2.** Root SSH, not qwen-mcp: this program lives in
`/opt/recall-enterprise`, which is outside qwen-mcp's four file roots. The check ran against
fastembed 0.8.0 and the artifact provisioned at
`/opt/recall-enterprise/models/bge-fastembed-cache`, from `/var/tmp/recall-profile-check`
(scratch, left in place). `/opt/recall-enterprise` was not modified.

| Check | Result |
|---|---|
| `artifact_tree_sha256` of the provisioned tree | `9a443d711e06…c919c`, **equal to the digest recorded in `manifest.json` on 2026-08-03** |
| Socket block fires (positive control) | yes, on `create_connection`, `getaddrinfo` and `socket.connect` |
| `bge-small-symmetric-v1` loads with connections refused | yes, dim 384, both encoders return 384 |
| `bge-small-asymmetric-v1`, `bge-small-context-section-v1` | both load, dim 384 |
| Checksum mismatch | refused |
| Missing artifact | refused |
| Deployment identity fingerprint | `c992f9f018d68570acb82d27233599cc20a2d65ed485b0e2fcfb18201cceacc7` |

The digest agreement is worth naming: the manifest value was produced by a different tool months
earlier, so this is an independent confirmation of the tree-hash implementation, not a
self-comparison.

**Finding: `bge-small-asymmetric-v1` is asymmetric in name only.** Under fastembed 0.8.0,
`embed`, `query_embed` and `passage_embed` return **byte-identical** vectors for
`BAAI/bge-small-en-v1.5` (`embed == query == passage`, cosine 1.0). `TextEmbedding.query_embed`
delegates to the model implementation, and this one applies no query prefix.

Nothing in this session depends on that being false, and nothing here changes it. What it means:

* the three context profiles and `bge-small-asymmetric-v1` currently differ from
  `bge-small-symmetric-v1` in the passage TEXT they embed, not in the encoder they use;
* the dispatch still has to be correct, because `qwen3-embedding-0.6b-384-v1` does use a distinct
  instruction-prefixed query encoder, and because a future fastembed or model could add one
  without changing a profile ID;
* a claim that this deployment uses a distinct query encoder for BGE would be **false**. If a BGE
  query instruction is wanted, that is a new experiment and needs registering. It was not done
  here: new model behaviour is out of scope for this session.

**Finding: the socket block has to be a `socket.socket` subclass.** Replacing `socket.socket` with
a function looks stricter and is unusable: `ssl` builds `class SSLSocket(socket)` at import time,
so the swap turns any later `import ssl` into a `TypeError`. The first VPS2 run failed exactly
that way, because fastembed's import chain reaches `requests`. The local test now uses the same
instrument as the VPS2 check, so the unit test and the real run are measuring the same thing.

### Proving the tests can fail

67 tests across the six touched files plus one in `test_generations.py`, every one shown red.
Not a coverage number: a mutation harness applied one narrow change at a time, recorded which tests turned red,
and restored. It refuses to report anything if a mutation's search string is absent, so a silently
unapplied mutation cannot read as "the tests survived it".

| Route | Tests | How |
|---|---|---|
| Wide sweep, 57 mutations | 64 | one mutation per guard, across `embedding_registry`, `embeddings`, `cache`, `context`, `index`, `service`, `timing`, `generations`, `calibration`, `readiness`, and the published document |
| Two targeted mutations | 2 | see below |
| Run on VPS2 | 1 | `test_a_symlink_escaping_the_artifact_root_is_refused` **skips on Windows**, and a skipped test is never red. Run on Linux against the current module: the guard refuses, and with the guard removed it does not. A skip is not a pass, and this is the difference |

**The sweep found a test of mine that could not discriminate.** `make_embedder hardcodes the
artifact path variable` survived: the Qwen test set no digest, so `make_embedder` refused for the
missing digest rather than the missing path, and it passed identically whether or not the profile's
own variable was read. The test now supplies a real provisioned tree and a real digest, so the only
thing missing is `RECALL_QWEN_MODEL_PATH`, and the mutation kills it.

**Two survivors were bad mutations, not weak tests**, and both needed aiming at what the test
actually asserts. Relaxing `strict=True` on the FastEmbed side changed nothing the test reaches,
because that branch only runs when no digest is supplied, which never happens on the registry path;
all four strict resolves have to go before a missing artifact stops raising `FileNotFoundError`.
And dropping the relative path from the tree digest left the NUL separator behind, so an added file
still moved the hash by one byte; the separator has to go with it. Both kill their test once
aimed correctly.

⚠️ The harness restores with `Path.write_text`, which rewrites LF as CRLF on Windows, so it left
ten files dirty in `git status` with no content diff. Line endings were normalised afterwards and
`git status` verified back to exactly this session's files. A next session running it should
restore by BYTES.

### Gates run

| Gate | Result |
|---|---|
| `ruff check .` | clean |
| `mypy` | clean, 139 source files |
| `pytest -q` | 2263 passed, 35 skipped, 0 failed on the rebased tree (14 m 03 s, dedicated database) |

**The local suite needed a dedicated database, and then a FRESH one.** Two things bit, in order.

First, contamination: a `ConcurrentMigrator: another RE-call schema migrator is already running`
cascade across 74 tests, from another Python process on this machine holding the migration
advisory lock on the shared `localhost:5432` container. Nothing was killed, because a process that
might belong to a concurrent session is not mine to kill; the suite moved to a throwaway container
via `RECALL_TEST_DSN`.

Second, after the rebase, **2273 errors, all one cause**:
`MigrationChecksumMismatch: applied migration 0008_generation_foundation.sql checksum drift`. #196
edited an already-applied migration, and the throwaway database still had the previous version in
its ledger. That is the checksum guard working exactly as designed, not a defect in either change,
but it means **a test database provisioned before #196 must be recreated, not reused**. A next
session on this machine should expect both.

### Decisions a reader should be able to reverse

1. **A rejected profile warns, it does not refuse.** `make_embedder` logs a warning naming the
   verdict and the measurement, then builds it. Refusing outright (with an explicit override
   variable) would be the fail-closed choice and is one line; it was not taken because it changes
   operator-facing behaviour beyond what was asked. VPS2's active profile is
   `bge-small-symmetric-v1`, so nothing live is affected either way.
2. **Calibration is still keyed by profile ID alone.** `load_for` matches `Calibration.embedder`
   against the profile ID, so two different artifacts under one ID share a threshold. The
   fingerprint now exists and would be the stricter key. Not changed here: calibration artifacts
   on disk carry the ID, and calibration semantics were out of scope.
3. **The default `FastEmbedEmbedder()` still claims `bge-small-symmetric-v1` with a
   `legacy-unverified` digest.** That is a legacy identity sharing a registered ID. Left alone
   deliberately: every evaluation harness and published results table labels that embedder, and
   renaming it would move published labels and orphan existing calibration files. Readiness
   already fails it for the unpinned digest.
4. **`EmbedderIdentity` in `recall/lineage.py` overlaps with `EmbeddingProfile` and was not
   merged.** They serve different layers (manifest provenance against runtime encoder identity).
   Worth a deliberate decision at some point rather than drift.

### Standing blockers

| Blocker | Kind | Effect | Change |
|---|---|---|---|
| **No latency reference host.** VPS2 has 12 cores under a permanent load average near 8 from unrelated live production. | External dependency. Do not work around it. | Latency is **PENDING**; promotion blocked on latency grounds. Quality and safety gates still run. | unchanged. The VPS2 work this session was correctness only and cites no timing |
| **No production corpus.** | Open | Nothing may be claimed about enterprise-corpus behaviour. | unchanged |
| **No approved local generator confirmed.** | Open | The generator-neutral evidence path stays unexercised end to end. | unchanged |

### What the next session should start with

1. **Session 3 of the backlog, unchanged: give the migration outbox a drain.** `replay_pending`
   still has no producer and no operator command, and `cutover` still refuses forever after a
   crash between the event append and its completion. It remains the only item that can deadlock a
   production migration with no shipped workaround.
2. Decide item 1 above: should a rejected profile refuse to start?
3. If the asymmetric BGE profiles are meant to use a distinct query encoder, register that as an
   experiment. Today they do not, and the registry now says exactly which encoder each profile
   claims, so the gap is visible rather than assumed.

---

## 2026-08-05, gap matrix: 21 requirement areas audited against master

### Session ledger

| # | Item | Outcome |
|---|---|---|
| 1 | Audit 21 requirement areas against the merged program | done, every verdict carries a `file:line` or an explicit "no such symbol anywhere" |
| 2 | Confirm or refute eight inherited leads | done, 6 confirmed, 1 sharpened, 1 refuted |
| 3 | Decide the two-ledger question and record reasons | done, **keep separate**, with three follow-ups |
| 4 | Produce the ordered backlog mapped to sessions 3 to 11 | done, 37 items |
| 5 | Rebase onto #194, which landed the prior session's status file first | done, both entries kept |

Read-only against the code. Nothing was fixed. Deliverable:
[docs/superpowers/plans/2026-08-05-enterprise-gap-matrix.md](superpowers/plans/2026-08-05-enterprise-gap-matrix.md).

### A concurrent session wrote the entry below, and it landed first

The 2026-08-04 entry was **still being written while this audit ran**. It grew from 170 to 266 lines
mid-session, gaining an AUD-1 fix, a corrected "macro-averaged" label and a rebuilt discriminating
test. That session shipped it as #194, which merged before this one opened, so the 08-04 entry
reached `master` on its own and this section was rebased on top of it.

The add/add conflict that produced was resolved by keeping **both** entries, newest first, rather
than by taking a side. `master`'s header and its whole 08-04 entry are carried through **byte for
byte**, asserted rather than eyeballed: the resolution script compares the pre-entry and post-entry
line ranges against the merged blob and refuses to write if either differs.

The lesson for the next session that touches this file: it is a shared handoff, so two sessions will
race it again. Read `origin/master`'s copy immediately before editing, never a snapshot taken earlier
in the session, and append rather than rewrite.

### Corrections to the inherited snapshot

- **This file was untracked when this audit started.** The 2026-08-04 entry records creating it as
  done, but at that point it existed only as an untracked working-tree file on the local branch
  `bench/mtrag-symmetric-baseline`, in no ref. #194 has since committed it; see the note above.
- **`bench/mtrag-symmetric-baseline` was local and unpushed when this audit ran.** The MT-RAG runner
  commit existed on that branch only, with no remote and no PR. It has since shipped as #194 and is
  on `origin/master`; this session never touched it.
- **The audited tree is not PR 181.** Five PRs merged after it, four of which changed audited code:
  #182 calibration binding, #184 strict trust, #185 shared pool and tenant-scoped readiness, #189
  store latency instrumentation. Verdicts describe `origin/master` at `8147d96`, and the matrix says
  per row where the current state is mostly later work.
- **`recall/migrations/sql` holds 0001 to 0011, not 0001 to 0010.** `0011_calibration_binding.sql`
  arrived with #182.

### Verdict summary

| Verdict | Count | Areas |
|---|---|---|
| Implemented and tested | 1 | 17 |
| Implemented, insufficiently tested | 6 | 1, 3, 6, 7, 9, 16 |
| Partially implemented | 10 | 2, 4, 5, 10, 11, 12, 13, 14, 15, 20 (12, 13 and 14 are "implemented, zero tests") |
| Missing | 1 | 19 |
| Intentionally rejected / out of scope | 0 | none |

Areas 8 and 21 are counted above by their dominant state and neither fits one column cleanly: area 8
is implemented in code with its **required record** missing, and area 21 is partial with two live
contradictions. The matrix says so per row.

### The findings that change what a later session should do first

1. **`replay_pending` has no producer and no operator command.** `recall/control_plane.py:298` is its
   only reference in the whole repository, and `recall-enterprise` has no `replay` subcommand
   (`recall/enterprise_cli.py:22-41`). A crash between the event append and its completion
   (`recall/index.py:625` and `:631`) leaves a pending row, and `cutover` then refuses forever
   (`recall/control_plane.py:278-279`). The recovery mechanism exists, looks correct, and cannot be
   invoked. This is session 3.
2. **Erasure does not reach the outbox.** A pending event's payload holds full chunk text and vectors
   (`recall/index.py:614-623`) and is cleared only on completion (`recall/control_plane.py:261`).
   Nothing removes it on `recall_forget`. For a right-to-erasure path that is a policy question, not
   only a bug.
3. **The promotion gate has never been shown to pass.** `tests/test_promotion.py` is one test and it
   asserts `not promoted`. A gate only ever observed to fail is compatible with a gate that refuses
   everything.
4. **Three whole areas have zero tests**: route notification and polling (12), dual writes (13),
   atomic cross-generation forget (14). `ShadowIndexTarget`, `get_shadow`, `make_profile_embedder`,
   `delete_sources_across`, `invalidate_route`, `replay_pending`, `RetrievalOverloaded`, `stage_ms`,
   `check_enterprise_readiness` and `enterprise_cli` each have **no test reference anywhere** in
   `tests/`.
5. **The Qwen3 rejection record does not exist.** `git grep -in "qwen"` over tracked `*.md` returns
   nothing. PR 181 describes the profile as "promotion gated"; no gate and no record are in the
   repository.
6. **`check_enterprise_readiness`'s calibration branch cannot fire.** PR 181 passed
   `calibration=calibration`; commit `0341c15` (#182) removed the argument, and master calls the
   function with three (`recall_mcp/server.py:336-339`). The parameter defaults to `None`
   (`recall/readiness.py:160`), so every enterprise boot takes the `calibration is None` path at
   `:195-196`: a permanent degraded-readiness warning, and the identity-mismatch **failure** at
   `:198` is unreachable from the server. It reads as a calibration gate and is not one. Nothing
   caught it because the function has no test.

### Leads: what survived verification

| Lead | Verdict |
|---|---|
| Profile registry is a dict literal in `service.py::make_embedder`, partly duplicated in `context.py` | **confirmed**: `recall_mcp/service.py:113-120` and `recall/context.py:37-41`, two independent literals, already differing in extent |
| `cache_key` omits `context_version` and `artifact_digest` | **confirmed**: `recall/cache.py:26`; both are independently settable at `recall/embeddings.py:334-339` |
| `latency_budget_ms` is enforced nowhere; only the promotion gate reads it | **sharpened**: `git grep` returns three hits, all inside `recall/profiles.py`. The gate's budget is a separate caller-supplied field (`recall/promotion.py:33`). **Nothing** reads the profile's value |
| `recall/evidence.py` is complete but unexported and unreachable | **confirmed**: absent from `recall/__init__.py`; referenced only by `tests/test_evidence.py` and one prose line |
| `recall/promotion.py` has no producer outside its test | **confirmed**: `tests/test_promotion.py:8-13` is the only construction site |
| `FAST_PROFILE` and `QUALITY_PROFILE` share `candidate_k=20` | **refuted as a defect**: `docs/ENTERPRISE_RETRIEVAL.md:57-59` specifies "the same candidate pool" deliberately, so cost differences are attributable to the reranker alone |
| `readiness.py`, `validate_generation_parity`, `replay_pending`, `enterprise_cli.py` have no tests | **confirmed, one narrowed**: `tenant_readiness` / `process_readiness` (later session) *are* covered by `tests/test_tenant_readiness.py`; `check_enterprise_readiness` is not. The other three have no test reference at all |
| Two migration ledgers exist | **confirmed**: see the decision below |

### The ledger decision

**Keep `recall_schema_migrations` and `recall_schema_versions` separate. Do not merge.** Merging is a
one-way door guarded by committed checksums (`recall/schema.py:53-54`); the two ledgers have
different scoping (per target table with a `__global__` bucket, versus database-global) and different
lifecycles (the control plane must exist before any generation does); and the enterprise deployment
is opt-in, so merging would impose control-plane tables on every deployment.

The split must be made deliberate rather than accidental, which needs three things it does not have:
an advisory lock on `ControlPlane.apply_migrations` matching `recall/schema.py:24`; readiness that
verifies **both** ledgers rather than only the one `check_schema` covers; and a paragraph in
`docs/MIGRATIONS.md`, which currently never mentions the second ledger. Full reasoning in the matrix.

### Documentation contradictions found on master

Both are shipped, and each makes an operator following one document wrong according to the other.

- `docs/ENTERPRISE_RETRIEVAL.md:13` sets `RECALL_DSN` to the **migration** role, and
  `recall/enterprise_cli.py:14` reads it. `docs/MIGRATIONS.md:12` calls `RECALL_DSN` the deprecated
  fallback for the **serving** DSN, with `RECALL_MIGRATION_DSN` as the schema owner.
- `README.md:628-632` says strict calibration enforcement "has not landed yet"; #184 landed it and
  `README.md:205` already marks it ✅.

### Standing blockers

| Blocker | Kind | Effect | Change |
|---|---|---|---|
| **No latency reference host.** VPS2 has 12 cores under a permanent load average near 8 from unrelated live production. | External dependency. Do not work around it. | Latency is **PENDING**; promotion blocked on latency grounds. Quality and safety gates still run. | unchanged |
| **No production corpus.** Everything measured is the public MT-RAG release. | Open | Nothing may be claimed about enterprise-corpus behaviour. | unchanged |
| **No approved local generator confirmed.** | Open | The generator-neutral evidence path stays unexercised end to end. | **now blocks backlog item 11** (session 5): the evidence boundary can be made reachable and tested against a fake generator, but the real path stays unexercised until this resolves |

### What the next session should start with

Session 3 of the backlog: give the migration outbox a drain. Add `recall-enterprise replay`, test
`replay_pending` against the actual crash shape, test the `cutover` refusal branch, and decide what
erases a pending event's payload when a tenant invokes erasure. Everything else in the program can
wait behind a deadlock that has no shipped workaround.

Two items carried forward from 2026-08-04 and still open: the MT-RAG baseline has no
`results/ARTIFACTS.md` row (deliberately), and `bench/mtrag-symmetric-baseline` is still local and
unpushed.

---

## 2026-08-04 — MT-RAG symmetric baseline salvaged, validated, archived, runner committed

### Session ledger

| # | Item | Outcome |
|---|---|---|
| 1 | Salvage the untracked runner and the finished run off `/var/tmp` | done |
| 2 | Validate the frozen run against its preregistration | done, 8 checks, all PASS |
| 3 | Archive with a SHA256 manifest and a provenance note | done, 29 files |
| 4 | Commit the runner under version control | done, this branch |
| 5 | Create this status file | done |
| 6 | Fix AUD-1 (the "macro-averaged" label) | done, label corrected and pinned by a discriminating test |

No new benchmark run was performed. Implementation work was out of scope for the salvage itself;
the AUD-1 fix (item 6) was authorised separately after the audit surfaced it.

### Corrections to the inherited snapshot

Two facts in the session brief did not survive verification, and are corrected here so the next
session does not inherit them again.

- **`docs/ENTERPRISE_PROGRAM_STATUS.md` did not exist.** It was described as the authoritative
  prior-session handoff. It was absent from `origin/master` and from every ref. This file is its
  first version, not an update.
- **`summary.json` reports gap warnings on two arms, not one.** The brief named the single warning
  on `recall_default_last`. `sparse_last` reports 507 of 507. See the explanation below; it is
  expected behaviour, but a reader told to expect one warning would read the arm as broken.

Also worth carrying forward: the local checkout was on `codex/enterprise-retrieval-program` at
`1aa93ec` while `origin/master` had advanced to `8147d96`. This branch was cut fresh from
`origin/master`.

### Salvage record

The runner existed only as untracked files inside a `/var/tmp` checkout, and the results existed
only in `/var/tmp`. Both are now in two durable places.

| What | From | To |
|---|---|---|
| `benchmarks/mtrag/{README.md,__init__.py,run.py}`, `tests/test_mtrag_benchmark.py` | `/var/tmp/re_call_mtrag_20260803/RE-call` (rev `3d3c905`, untracked) | this repository, and `…/runner/` in the archive |
| `results/official_run_1/` (6 predictions, 6 metrics, manifest, summary) | `/var/tmp/re_call_mtrag_20260803/results/` | `/var/lib/recall-benchmarks/2026-08-04-mtrag-symmetric-baseline/results/` |
| `benchmark.log`, 5 index logs, `provision.log`, both shell drivers, `preregistered_manifest.json` | `/var/tmp/re_call_mtrag_20260803/` | same archive directory |

Archive root: **`/var/lib/recall-benchmarks/2026-08-04-mtrag-symmetric-baseline/`** on VPS2,
29 files, 61 MB, `MANIFEST.sha256` covering every file, `NOTE.md` carrying the provenance.
`sha256sum -c MANIFEST.sha256` passes. Every archived artifact was compared byte for byte against
its `/var/tmp` source at archive time. **The source directory was not deleted.**

VPS2 access for this work used root SSH, not qwen-mcp: qwen-mcp's file roots are
`/opt/sentiment_agent`, `/var/lib/qwen_agent`, `/var/log/qwen_agent` and `/etc/systemd/system`,
and this program lives outside all four.

### Validation verdicts

Every check ran against the archived copy, with the expected values taken from the preregistration
and the MT-RAG release rather than from the predictions themselves.

| # | Check | Verdict | Evidence |
|---|---|---|---|
| 1 | Archive is byte-identical to the `/var/tmp` source | **PASS** | 14 files compared by SHA256, zero mismatches |
| 2 | Release inputs match the manifest | **PASS** | `reference.jsonl` hashes to the manifest's `input_sha256.tasks` |
| 3 | Each of the six prediction files holds exactly the 507 frozen task IDs, once each | **PASS** | per file: 507 rows, 507 unique, 0 duplicates, 0 missing, 0 extra, against the ID set derived from `reference.jsonl` |
| 4 | Prediction SHA256 for the two known arms | **PASS** | `recall_default_last` = `d0f4ce2d…51676`, `recall_default_recent3` = `12cc5e3f…3fef1`, both as expected |
| 5 | Scored metrics use exactly the 332 judged tasks | **PASS** | all six arms: `overall.count` 332, domains clapnq 83 / cloud 86 / fiqa 58 / govt 105, and `per_query` keys set-equal to the qrel query IDs |
| 6 | The six arms match the preregistration exactly | **PASS** | names in order, and each `metrics.arm` equal field for field to the matching `frozen_arms` entry. The preregistration was not edited or relabelled |
| 7 | DB chunk counts per domain | **PASS** | clapnq 183 408, cloud 72 442, fiqa 61 022, govt 49 607, total 366 479 |
| 8 | Revisions and adapter identity | **PASS** | RE-call `3d3c905…`, MT-RAG `cc5b1d4…`, adapter SHA256 `a675d900…e2347` equal to the salvaged `run.py` |

The four previously unrecorded prediction hashes, now facts:

| arm | SHA256 |
|---|---|
| `recall_rerank_last` | `cfd5f9a48d59b36511f4d803770f3da1e47239c85c0f0c1984510b185f5bea2c` |
| `recall_rerank_recent3` | `6defd4f299de6d0880a1d3dd889f72f34f4794240749e2b77bc0321db5ad273d` |
| `dense_last` | `d7e0b61dbc78987231ff0740115115db4d783a3b56c8fd75dc3c7005c24430a1` |
| `sparse_last` | `3d1395294d5b676d9fb8748e9f512f0f7cd59a0adb8fc5d0deb3e078a3523333` |

#### Two findings worth carrying forward

**The DB count check first returned zero, and that was the tenant isolation model working.** A
plain `SELECT count(*)` on `recall_mtrag_bge_v1_<domain>` as the runtime role returns 0 rows, not
the counts above. The tables carry `FORCE ROW LEVEL SECURITY` with a `tenant_isolation` policy on
`current_setting('recall.tenant_id', true)`; with the GUC unset, the policy correctly matches
nothing. The true counts were confirmed three independent ways: `pg_stat_user_tables.n_live_tup`,
a count with `recall.tenant_id` set, and a count as the table owner. All rows are tenant
`default`. A future session that reads zero here should set the GUC before concluding the index is
empty.

**`gap_warning` on `sparse_last` fires on every query by construction.** `gap_warning` is computed
over the dense candidate pool (`recall/guards.py`, floor 0.50 cosine), and an empty candidate set
counts as a gap, fail-closed. `sparse_last` sets `use_dense=False`, so the pool is always empty and
the flag always fires. It means "no dense evidence was gathered", not "the corpus lacks an answer".
This is documented in `HybridRetriever`'s docstring. The single warning on `recall_default_last`
is the real kind: task `34d3cde930baaf8a80a37bede060c827<::>2` (govt, judged), query "Who was the
last Kolb to live in the house?", where every dense candidate scored below the floor. The other
four arms report zero.

### What was measured

Full results are in the archive; the headline is reproduced here for handoff only, and the
artifact is authoritative.

**These are POOLED means over all 332 judged queries, not macro averages.** Each domain counts in
proportion to its judged-query count. The distinction is worth 1.5 % to 6.6 % and the two
definitions do not agree on one arm ordering; see AUD-1 below, now fixed in the README and pinned
by a test. The `elapsed` column is per-arm total including prediction writing and scoring, not
retrieval time alone (finding AUD-4, open).

| arm | nDCG@5 (pooled) | Recall@5 (pooled) | nDCG@10 (pooled) | elapsed (s) |
|---|---|---|---|---|
| `recall_default_last` (primary) | 0.3701 | 0.4081 | 0.4048 | 416.0 |
| `recall_default_recent3` | 0.3205 | 0.3604 | 0.3667 | 587.3 |
| `recall_rerank_last` | 0.4227 | 0.4555 | 0.4661 | 18 468.1 |
| `recall_rerank_recent3` | 0.3173 | 0.3671 | 0.3676 | 20 480.8 |
| `dense_last` (ablation) | 0.3304 | 0.3556 | 0.3677 | 120.7 |
| `sparse_last` (ablation) | 0.2542 | 0.2936 | 0.2905 | 418.3 |

Run window 2026-08-04T08:21:20Z to 2026-08-04T19:36:21Z, elapsed 11 h 15 m. The two reranked arms
consumed 96 % of it.

**The p50 and p95 latency figures in the artifacts are diagnostic only.** They were measured on a
12-core host under unrelated live production at a load average of roughly 8. No promotion decision
on latency grounds may cite them.

### Runner now under version control

`benchmarks/mtrag/` and `tests/test_mtrag_benchmark.py` are committed on this branch. They existed
in no ref of the repository before it.

Two changes were made to the salvaged `run.py`. First, it failed `mypy` with two `var-annotated`
errors and `disallow_untyped_defs` applies to `benchmarks/`, so two local variable annotations were
added. Second, `score_predictions` gained a docstring as part of the AUD-1 fix below. **The
committed `run.py` is therefore not byte-identical to the adapter that produced the run**, so its
SHA256 no longer matches `adapter_sha256`. The byte-exact adapter is preserved at
`…/runner/run.py` in the archive, and that is the copy the manifest's `adapter_sha256` refers to.

Behaviour-neutrality was proven, not assumed. The annotation-only change was verified by compiling
both files and comparing every code object on opcodes, operands, names and argument layout;
fingerprints were identical. Adding the docstring then broke that equivalence in
`score_predictions` (a docstring lands in `co_consts`), so bytecode identity was **replaced by a
stronger check rather than quietly dropped**: both adapters were run over the real frozen
predictions for all six arms, and every figure was required to equal the archived metrics file.

> All six arms reproduce the archived scores exactly, to full float precision, under both the
> archived and the committed adapter (`committed == archived == frozen` for every arm). Example:
> `recall_default_last` overall nDCG@5 is `0.3700930547143303` from both.

Both proofs carry a negative control (mutate `0.0` to `1.0` in `ndcg_at` for the bytecode check;
reverse one task's context order for the reproduction check), each confirmed to report a
difference, so a green result is detection rather than a check that cannot fail.

Gates run on the change: `ruff check .` clean, `mypy` clean across 137 source files,
`pytest tests/test_mtrag_benchmark.py` 3 passed, and the claim-gate suite
(`test_published_numbers_have_artifacts`, `test_results_artifact_model_stack`,
`test_results_artifact_provenance`, `test_findings_crossrefs`) 235 passed with
`benchmarks/claim_gate.py` exiting 0. No `results/ARTIFACTS.md` row was added and no number was
published into a gated document.

### Audit of the committed runner (CCA, DEEP tier, no-fix)

The commit hook required a bug review, so the runner went through the tiered CCA pipeline at DEEP
(forced by the numeric-path flag: this code computes the published metrics). Eight auditors, 38 raw
findings, deduplicated and verified. **Nothing was fixed:** implementation work was out of scope for
this session, and changing the adapter would diverge it further from the frozen run. Everything
below is reported for a decision.

**Deterministic coverage for this run was NONE.** `cca_checks` is not installed in this repo or its
venv, so no static backend was available. Where a verdict below says "measured" or "read", that is a
command I ran; where it says "reasoned", it is LLM adjudication only.

| ID | Finding | Verdict | Basis |
|---|---|---|---|
| **AUD-1** | `benchmarks/mtrag/README.md` said the adapter reports "macro-averaged nDCG and Recall". `score_predictions` pools all 332 judged queries into one list and divides by 332, weighting each domain by its judged-query count (clapnq 83, cloud 86, fiqa 58, govt 105). This repo already defines macro as the *unweighted* mean of per-corpus values (`recall/promotion.py:63-71`). | **CONFIRMED, P1 → FIXED** | measured |
| **AUD-2** | `preregistered_manifest.json` records `embedder_model` but no reranker model or revision, although two of the six arms rerank. The archived reranker identity had to be recovered by reading `recall/rerank.py`, not from the artifact. | **CONFIRMED, P2** | read |
| **AUD-3** | `index_domain`'s completion guard (`final_count != seen`) compares row *counts*, never row *identity*. Re-running against a changed corpus under the same `--table-prefix` yields `final_count == seen` and the guard passes on a silently mixed index. | **CONFIRMED, P2** | reasoned |
| **AUD-4** | Per-arm `elapsed_s` is evaluated after prediction writing and after `score_predictions` (which re-parses all four qrels files), so it is not retrieval wall time. The `latency_ms` p50/p95 are unaffected: those come from the in-loop list. | **CONFIRMED, P2** | read |
| **AUD-5** | Neither `FastEmbedEmbedder` nor `CrossEncoderReranker` is constructed with the offline-enforcement arguments the library exposes, so the harness can fetch model weights over the network at runtime. Runtime model downloads are on this program's standing out-of-scope list. | **CONFIRMED, P2** | reasoned |
| **AUD-6** | `run.py` has no module docstring, and `argparse.ArgumentParser(description=__doc__)` therefore renders `--help` with an empty description. | **CONFIRMED, P3** | read |
| **AUD-7** | `p50`/`p95` use `ordered[int(p * (n-1))]`, which floors. At n=507 the reported p95 is the 481st of 507 values, the 94.87th percentile, and the bias is always toward the fast end. | **CONFIRMED, P3** | measured |
| **AUD-8** | `run_arm` constructs its reranker, four `PgVectorStore`s and the task list *before* its `try`, and `PgVectorStore.__init__` opens a connection eagerly (`recall/store.py:606`) with no `__del__`. A failure while opening store 3 leaks stores 1 and 2. Bounded: the exception aborts the run and the process exit closes the sockets, so this cannot accumulate. Both auditors rated it P1; downgraded on that mitigation. | **CONFIRMED, P3** | read |

Also confirmed and minor: the `DATABASE_URL` dotenv fallback is the only non-`RECALL_`-prefixed
config key in the codebase; the overall aggregation lacks the empty-list guard its per-domain twin
has; the duplicate-`_id` failure reports itself as a "partial/mixed index".

Dropped as false positives, with evidence: the "no TypedDict" and "god file" structural findings
(style preferences on a deliberately frozen benchmark adapter, and `ruff` plus `mypy` are green),
and the zip-bomb finding (the only path to a hostile archive is the operator pointing
`--mtrag-root` at their own tampered copy).

**AUD-1 in detail, because it is the one that touches a published label.** Recomputing every arm
both ways from the archived per-domain rows:

| arm | nDCG@5 pooled | nDCG@5 macro | delta |
|---|---|---|---|
| `recall_default_last` | 0.3701 | 0.3602 | −0.0099 (−2.67 %) |
| `recall_default_recent3` | 0.3205 | 0.3073 | −0.0132 (−4.11 %) |
| `recall_rerank_last` | 0.4227 | 0.4133 | −0.0094 (−2.21 %) |
| `recall_rerank_recent3` | 0.3173 | 0.3067 | −0.0106 (−3.35 %) |
| `dense_last` | 0.3304 | 0.3253 | −0.0051 (−1.53 %) |
| `sparse_last` | 0.2542 | 0.2374 | −0.0168 (−6.61 %) |

The pooled figure is higher than the macro figure for **every** arm and **every** metric, by 1.5 %
to 6.6 %. Arm ordering is preserved under nDCG@5 and nDCG@10, but **not under Recall@5**, where
`dense_last` and `recall_default_recent3` swap. So this is not purely cosmetic: on one of the three
reported metrics the two definitions disagree about which arm is better. The headline conclusion
(`recall_rerank_last` first, `recall_default_last` second) survives both definitions.

This finding did **not** go to the adversarial skeptic panel. Its verdict rests on a definition read
out of `recall/promotion.py` and on a measurement executed over the artifacts, so it is
artifact-backed, and the pipeline forbids routing an artifact-backed verdict to three LLM skeptics
to re-litigate.

**AUD-1 is fixed.** The label was corrected rather than the metric. Changing the aggregation would
have invalidated the archived run's comparability without a re-run, for a figure that is already
derivable: `domains[*]` is in every metrics file, so anyone wanting the macro average can compute
it from what is archived. Three things changed:

1. `benchmarks/mtrag/README.md` now states that `overall` is a pooled mean over judged queries,
   says explicitly that it is not a macro average in `recall/promotion.py`'s sense, and records
   the size of the gap.
2. `score_predictions` gained a docstring stating the averaging semantics at the place the
   arithmetic happens, so the next reader does not have to re-derive it from the loop.
3. `test_score_predictions_is_macro_average` was **renamed and its fixture rebuilt**. This was the
   load-bearing part. The old fixture used one judged query in each of two domains, where pooled
   and macro both give 0.5, so a test whose name asserted "macro" passed identically under either
   definition. It could not have failed. The replacement,
   `test_score_predictions_overall_is_pooled_not_domain_macro`, uses three judged queries in
   clapnq against one in cloud: pooled gives 0.75, domain macro gives 0.5, and the test asserts
   both the pooled value and that it differs from the macro of the domain rows.

The new test was proven able to fail: `score_predictions` was mutated on an isolated copy to
compute the domain macro, and the test's assertions were replayed against it. Green on the real
implementation, red on the mutant. The proof script refuses to report a result at all if the
mutation fails to apply, so a silently unapplied mutation cannot read as success.

### Standing blockers

| Blocker | Kind | Effect |
|---|---|---|
| **No latency reference host.** VPS2 has 12 cores under a permanent load average near 8 from unrelated live production. It cannot serve as the 16-vCPU idle reference environment. | External dependency. Do not work around it. | Latency is **PENDING**. Promotion is blocked on latency grounds. Quality and safety gates still run. |
| **No production corpus.** Everything measured so far is the public MT-RAG release. | Open | Nothing may be claimed about enterprise-corpus behaviour. |
| **No approved local generator confirmed.** The evidence boundary requires a local generator, and none has been confirmed approved. | Open | The generator-neutral evidence path stays unexercised end to end. |

### What the next session should start with

1. Decide whether the MT-RAG baseline earns a `results/ARTIFACTS.md` row. It was deliberately not
   added here. Adding one means satisfying `claim_gate.py` and the two artifact tests in the same
   commit, and a new gated section costs a marker per number.
2. Resolve the latency reference host as an external dependency, or record explicitly that
   promotion stays blocked. It is not solvable on VPS2.
3. Confirm whether an approved local generator exists. Until it does, the evidence boundary cannot
   be exercised end to end.
