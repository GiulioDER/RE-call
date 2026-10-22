# Audit memory: anchored records for the CCA auditor brief

Date: 2026-10-21
Status: draft, owner decisions 1 to 8 recorded; implementation plan in
`docs/AUDIT_MEMORY_PLAN.md`

## Intent

CCA (the `cca-audit` package) already saves every fix it lands to an append-only ledger, and
RE-call serves those notes from the `cca` tenant. The audit pipeline reads them back into the
Layer 1 auditor brief. Measured in CCA's own pre-registered replay (cca-audit
`docs/preregistrations/2026-10-20-prior-bugs-in-brief.md`), the brief did not raise confirmed
findings (71 without it, 54 and 55 with it), and the RE-call "similar fixes" arm's false-positive
rate was 5.4 points above the no-brief arm, 0.4 over that replay's 5-point bound. The follow-up
replay (`2026-10-20-prior-bugs-rule-and-noise.md`) found the confirmed-findings deficit is not
established beyond noise, and that the brief's rule sentence alone, with an empty block,
reproduces it. So the honest summary is: no measured benefit, and a small measured cost.

This design makes the memory serve **audit precision**: more true findings kept, fewer false
positives, and no repeat of a false positive that was already judged on the same code. Interactive
bug-fixing sessions are a later step and are out of scope here.

## Why the current setup cannot do this

Measured on the live tenant on 2026-10-21:

1. **RE-call keeps no structure.** `recall/frontmatter.py` keeps `VALIDITY_KEYS`, `recall_graph`
   and one facet, `type` (`FACET_KEYS`), which is `project` on every CCA memo. Repo, files,
   severity, category and status exist only in prose, so nothing can be filtered on the server and
   `cca_checks bugs prior --files` joins file paths client side against the local ledger.
2. **Anchors are coarse.** The 637 backfilled rows carry every file their fix commit touched
   (`cca_checks/bugs/backfill.py`), so a bug in one function is attributed to unrelated files in
   the same commit. The 54 forward rows carry their L4 fix entry's files (`forward.py`), which is
   better but still has no symbol. The memo text says "Files changed by that commit" for both.
3. **The semantic search abstains on auditor-style queries.** The calibration set is hand
   paraphrases of memos; a query worded like a symptom returned five genuinely relevant lock and
   race fixes, every one `low_confidence`, and abstained.
4. **False positives are never recorded.** The scorecard holds `FALSE_POSITIVE` dispositions, but
   with no file, symbol, claim or evidence, and the L6 architect never sees them.

## Goals

1. The brief carries **prior bugs anchored to the code under change**: same repo, same file, same
   symbol, served by an exact lookup in RE-call.
2. A finding that matches an **accepted false positive on unchanged code** is suppressed
   automatically and visibly.
3. Confirmed findings do not fall relative to a run with no brief, and the false-positive rate
   falls.
4. **Nothing is lost in the round trip** CCA to RE-call to CCA: one front matter schema, and a
   ledger row rebuilt from RE-call's answer equals the row CCA wrote (see Round-trip contract).
5. **Regression edges**: a bug whose defect was introduced by an earlier CCA fix carries
   `caused_by` that fix, derived deterministically (see Regression edges).
6. **Lessons**: recurring patterns across several bugs are distilled into lesson records, accepted
   before use, and shown as context (see Lessons).
7. **Code join**: records join to the current code in a code tenant by the same anchor key, so a
   record says whether the code it describes still exists unchanged (see Code join).

## Non-goals

- The "similar fixes" semantic block. It is removed from the brief and returns only after a version
  recalibrated on real audit queries beats the no-brief arm in a pre-registered replay (owner
  decision, 2026-10-21).
- Semantic recurrence search ("code elsewhere that looks like a past bug") in the brief. It is
  built as a measured experiment under Code join and enters no brief until a pre-registered
  replay supports it.
- Turning `recall_report_use` into ranking or calibration input.
- Interactive bug-fixing sessions, HUNT mode, cross-repo priors.

## Anti-goals (a failure even if it works)

- **Invisible suppression.** A suppressed finding must remain in the run's record with its verdict
  and the citation of the record that suppressed it. Two past failures in this owner's projects
  motivate this: suppression rules in a citation checker hid real defects by moving them to another
  category, and a "do not flag" list containing an unreviewed choice silenced seven of nine
  auditors on the finding that mattered.
- **Suppression on changed code.** A false-positive record never applies once the code it was
  judged on has changed.
- **Fewer confirmed findings** than the no-brief arm.
- **RE-call needing access to the audited repository.** RE-call stores and matches anchors; the
  caller computes them.

## Decisions made by the owner (2026-10-21)

1. Audit runs first; the L1 brief is the consumer.
2. A false positive may suppress future findings automatically when fp-check marked it
   `FALSE_POSITIVE` **and** the L6 architect accepted it.
3. L6 is extended to review the `FALSE_POSITIVE` list. Only false positives it accepts, in a run
   whose final L6 verdict is `APPROVED`, become suppressing records.
4. Version one ships anchored prior bugs and accepted false-positive records only.
5. The anchored prior-bugs block ships unmeasured, as context only, and is measured by a
   pre-registered replay once the trigger under Measurement is reached.
6. The data must flow in one front matter format, with nothing lost between CCA, RE-call and back.
7. Lessons with distillation, `caused_by` edges, and the join from memos to code chunks are in the
   plan.
8. The plan is divided into sessions that each work independently and report to one orchestrating
   session.

## The record

Each CCA memo carries one new front matter key, a one-line JSON object, following the
`recall_graph` idiom:

```yaml
recall_record: {"v": 1, "kind": "false_positive", "repo": "owner/name", "finding_id": "BUG-003", "run_id": "...", "category": "concurrency", "severity": "P2", "anchors": [{"file": "pkg/pool.py", "symbol": "Pool._lease", "code_sha": "<sha256>"}], "claim": "one line"}
```

- `kind` is `bug`, `false_positive`, `false_positive_retraction`, `lesson` or
  `lesson_retraction`. A retraction carries `supersedes:` naming the record it retracts.
- **The record carries every ledger field**, not a subset: for a bug, every `BugRow` field
  (`bug_id`, `tier`, `date`, `project`, `repo`, `files`, `finding_id`, `category`, `severity`,
  `auditor`, `fix_commit`, `run_id`, `evidence`, `sources`, `summary`, `supersedes`, `ordinal`) plus
  `anchors` and `caused_by`. The prose body is rendered from the record for people and for
  embedding, and is never parsed back. This is what makes a row rebuildable from RE-call's answer.
- `anchors` is a list of `{file, symbol, code_sha}`. `file` is repo-relative with forward slashes.
  `symbol` is the dotted name of the innermost enclosing function or class, or `null` for module
  level. `code_sha` is the SHA-256 of that symbol's source text with line endings normalised to LF
  and trailing whitespace stripped from each line; for a `null` symbol it is the hash of the whole
  file under the same normalisation. Lesson anchors set `code_sha` to `null`: a lesson is about a
  place, not one version of its code. The resolver and hash are specified once, as **anchor test
  vectors** (source text in, expected symbols and hashes out) committed identically to both
  repositories, and each repository's tests assert its implementation reproduces them.
- `recall_record` is metadata, not a truth claim. It does not touch validity, so the boundary in
  `docs/TRUTH_EXTRACTION_DESIGN.md` holds. Validity stays in `valid_from` and `supersedes`.

## Round-trip contract

The owner's requirement: one front matter format end to end, and nothing lost between CCA, RE-call
and back to CCA. Three mechanisms make it checkable rather than hoped for.

1. **One schema, published by RE-call.** `recall_record` version 1 is a JSON Schema file shipped in
   the RE-call package. CCA keeps a copy and a test asserts the copy equals the published file for
   the pinned RE-call version. A version RE-call does not know is refused at index time with a
   warning, never partially read.
2. **RE-call returns the object exactly as authored.** No field is renamed, added, defaulted,
   normalised or dropped: the parsed JSON CCA wrote and the `recall_record` RE-call returns compare
   equal as JSON values. `name`, `description`, `valid_from` and `supersedes` also survive and are
   returned alongside.
3. **Three tests, at three levels**, each with red proof:
   - CCA: ledger row to memo to parsed front matter to ledger row is the identity, for every kind,
     over golden fixtures that include the hard cases: Unicode, a summary containing `---`, `:`
     and quotes, a record over 10 KB, several anchors, a `null` symbol, `supersedes` and `caused_by`.
   - RE-call: the same golden memo files, committed identically to both repositories, are indexed
     into a lite store, and `recall_records` returns each authored object unchanged.
   - Live, after deployment: a CCA command compares every ledger row against RE-call's full dump
     of the `cca` tenant (`recall records --all`) and reports missing records, changed fields and
     server-side records with no ledger row. The refresh worker runs it after every rebuild and
     records the result; any difference fails the tick.

## Regression edges (`caused_by`)

A bug fixed by CCA that was itself introduced by an earlier CCA fix carries `caused_by` naming that
fix's `bug_id`. Today such a chain exists only in prose (one RE-call record says the PERF-001 fix
"promoted a dormant defect into a live one").

- **Derived deterministically in the run, never by a model.** When L4 fixes a finding, CCA runs
  `git blame` on the base-commit lines the fix modified or removed, collects the commits that
  introduced them, and maps each to ledger rows of the same repo whose `fix_commit` equals it. The
  result rides in the run record, because nothing downstream has a checkout.
- Historical rows can be back-filled by a separate command run where the repositories are checked
  out; rows it cannot resolve stay without `caused_by`.
- **RE-call** serves the edge both ways: `recall_records` returns `caused_by` and the reverse
  `caused`, and `recall_related` gains the relation `caused_by`.
- **Use in the brief:** a prior bug whose fix later caused a regression is marked as such. Context
  only.

## Lessons

A lesson is a short rule distilled from several bugs, such as "pool connections returned on a
worker thread have broken liveness checks three times in this module", with the member bugs cited.

- **Record:** `kind: lesson`, with `rule` (one sentence), `pattern` (what an auditor should look
  for), `anchors` (repo plus file, optional symbol, `code_sha: null`), `members` (at least three
  `bug_id`s), `accepted_by` and `accepted_at`.
- **Distillation** is a CCA command run on demand or on a schedule. It groups bug records by repo,
  category and file or symbol, follows `caused_by` chains, and keeps groups of three or more. For
  each group a model drafts `rule` and `pattern` from the members' records. Drafts are proposals,
  stored outside the ledger, and never reach RE-call.
- **Acceptance is the owner's**, through a CCA command that turns a proposal into a lesson row and
  memo. A model drafting relations has measured poorly in this project (an earlier pilot of
  model-proposed supersession edges reached 0.09 precision against a 0.75 bar), and a lesson is
  shown to every future audit of its scope, so one wrong lesson costs many runs.
- A lesson never lapses on its own when code changes, since it describes a place and a pattern. It
  ends by a `lesson_retraction` row.
- **Use in the brief:** a LESSONS block for the changed files, at most three, each with its member
  citations, worded as past patterns to check, never as "do not flag". Context only, covered by the
  same deferred measurement (an ablation arm without lessons).
- Model calls in distillation follow the paid-test rule: tests replace the transport; a live call
  needs the explicit opt-in.

## Code join

Records join to current code by the anchor key, so the brief can say whether the code a record
describes is still there unchanged, changed, or gone.

- **RE-call code chunks carry the same anchor.** For Python files, the code chunker splits on the
  symbols `ast` reports and writes `code_anchor: {repo, file, symbol, code_sha}` into chunk
  metadata, computed to the shared anchor test vectors. Other languages keep today's heuristic
  chunking and get no anchor, so they never join.
- **The join is two lookups on one key, never a cross-tenant read.** `recall_records` also matches
  `code_anchor` in code tenants, so a client asks the `cca` server for records and the code
  tenant's server for chunks with the same anchors. Each server reads only its own tenant.
- The brief labels each prior bug and lesson with the code status at the current default branch:
  `unchanged`, `changed`, or `gone` (symbol no longer present).
- **Code tenants:** the existing RE-call code tenant switches to the anchored chunker, and a new
  tenant indexes cca-audit. The private repositories CCA also audits are not indexed in this
  plan.
- **Recurrence experiment:** using a bug record's fixed code as a query against the code tenant,
  to find sibling code with the same pattern. Built behind a pre-registration and measured before
  anything reaches a brief.

## RE-call changes

1. **Parse and store `recall_record`** in `recall/frontmatter.py`, validated against version 1.
   An invalid object is an index-time warning naming the file and the reason, and the chunk is
   indexed without it; it is never silently dropped. The object lands in chunk `metadata`, which is
   already JSONB on both chunk tables, so no migration is needed. Lite stores the same object.
2. **An exact lookup with no embedding**, as an MCP tool `recall_records` and a CLI command
   `recall records`:
   - Input: `repo` and `anchors` (list of `{file, symbol, code_sha}`, at most 200), or `ids` (a
     list of record ids such as `bug_id`), or, on the CLI only, `--all` for a full dump of the
     tenant's records; optional `kinds`.
   - Output, per matching record: `kind`, `recall_record` (exactly as authored), `name`,
     `description`, `source`, `chunk_id`, `valid_from`, `supersedes`, `superseded_by`,
     `valid_until`, `generation_id`, the reverse edge `caused`, and `anchor_match`, one of `exact`
     (file, symbol and hash equal), `symbol_changed` (file and symbol equal, hash differs) or
     `file_only`. A record with several anchors reports each anchor's match.
   - It matches `recall_record.anchors` in record tenants and `code_anchor` in code tenants (see
     Code join), with the same rules.
   - No calibration and no abstention: an exact key match is not a relevance claim. Superseded and
     expired records are returned and marked, never hidden.
   - Read-only, tenant-scoped like every other read tool, bounded result count.
   - The CLI command is classified for lite (served on `sqlite:///`), per the rule that every new
     CLI command is either served there or refused by name.
3. **Generation and calibration.** Generation build reuses an earlier chunk set without re-parsing
   front matter whenever URI, sha256, pipeline fingerprint and `metadata_rule_version` match
   (`recall/generations.py`, `_METADATA_RULE_VERSION`). The parser change therefore bumps
   `_METADATA_RULE_VERSION`, so memos written with `recall_record` before the parser is deployed
   pick it up on the first build after deployment. The existing calibration is carried forward as
   today. The exact lookup reads the active generation only.

## CCA changes (the other half, in `cca-audit`)

1. **Richer `FALSE_POSITIVE` disposition rows** (Step 2.5b): add `file`, `line`, `symbol`,
   `code_sha`, `claim`, fp-check's evidence references, and a `fp_reason` code: `not_real`
   (criteria a and c), `out_of_scope` (criterion b, or "out of scope" in the deferred pass) or
   `settled` (criterion d). **Only `not_real` may become a suppressing record.** An out-of-scope
   finding sits on unchanged code by construction, so its hash would stay `exact` and suppress a
   real defect the first time that code is in scope; a settled-decision finding is already handled
   by the settled-decisions block.
   - For a `not_real` false positive whose evidence is a mitigation elsewhere (criterion c: a
     caller's guard, a config value), each evidence location is added as an extra anchor, so the
     record lapses when the mitigation changes.
   - Every anchor is computed during the audit run and carried in the run record. Nothing
     downstream has a checkout of the audited repository.
2. **L6 false-positive acceptance.** The architect is handed every `FALSE_POSITIVE` with its claim
   and evidence and returns `ACCEPT` or `REJECT` with a reason for each, recorded per finding. A
   rejected false positive is reported as disputed and routed `UNCERTAIN → human`; it does not by
   itself change the fix verdict.
3. **Writing the records.** In a run whose final L6 verdict is `APPROVED`, each accepted false
   positive (`fp_reason: not_real`) with resolvable anchors becomes a `false_positive` ledger row
   and memo. These do not wait for the run's pull request to merge, since the anchor hashes already
   pin the code they were judged on, so they need a write path at `pipeline close` beside the
   existing pending capture. Today `pending.capture` returns early when L4 lists no fixes, which
   would lose an APPROVED run whose findings were all false positives. A finding with no resolvable
   anchor produces no suppressing record.
4. **Bug rows carry finding-level anchors** from the L2 finding location, hashed over the code at
   the end of the run (the fixed code), not the commit file list. The memo template writes
   `recall_record` and, when set, `supersedes` (today `render_memo` never writes it).
5. **The brief.** Step 1 drops the three `recall_search` similar-fix queries. The agent calls
   `recall_records` with the changed symbols' anchors and saves the response; `bugs prior` renders
   prior bugs from it instead of joining the local ledger, labelling each with its `anchor_match`
   and listing `exact` and `symbol_changed` before `file_only`. The block is context only: the
   instruction that tells auditors to leave out a listed bug is removed, because the follow-up
   replay found that sentence alone reproduces the confirmed-findings deficit. The
   settled-decisions query to the project's own memory server is unchanged.
6. **Suppression is mechanical and happens after L2, not in the brief.** A deterministic step
   between L2 and L2.5 calls `recall_records` for each finding's anchors. A finding is suppressed
   only when all of these hold:
   - it matches a current (not superseded, not retracted, not expired) `false_positive` record;
   - every one of the record's anchors is `exact`;
   - the `category` is the same;
   - the finding is not high stakes, and its priority is not higher than the record's.

   A suppressed finding gets the new verdict `PRIOR_FP`, skips fp-check, and appears in the report
   and the disposition ledger with the record's citation. A finding that matches but fails the last
   condition goes to fp-check as usual, with the record attached as evidence, so the stronger
   verification DEEP requires for a high-stakes P1 is never bypassed. Auditors are not shown
   false-positive records.
7. **Revocation.** A wrongly accepted record is retracted by a `false_positive_retraction` row
   that supersedes it; a retracted record never suppresses again. An environment switch disables
   all `PRIOR_FP` suppression at once, for the case where the match rule itself is wrong.

## Measurement

Pre-register before running anything. Two things are measured, separately:

1. **Anchored prior bugs: shipped unmeasured, as context only, measured later** (owner decision,
   2026-10-21). Measuring now would be uninformative: only the 54 forward rows can carry
   finding-level anchors, so the block would be empty in most runs of CCA's existing eight-run
   replay; identical replay arms have moved by up to 6 per run and 13 in total; and one replay
   costs 8 to 16M tokens.
   - **Trigger, counted in runs rather than rows** (a row count does not guarantee a non-empty
     block in the runs being replayed): **12 completed DIFF-mode runs whose anchored block held at
     least one `exact` or `symbol_changed` record, spanning at least two repositories.** Twelve
     leaves room to pick eight to ten replay runs after excluding any whose pull request never
     merged; the first replay lost 2 of 8 runs to an empty block.
   - To make the trigger countable, each run record stores the block's record count by
     `anchor_match`, taken from the saved `recall_records` response.
   - **When it fires:** pre-register the replay, then run it on runs chosen from those twelve, arms
     "no brief" and "version one brief", three repeats per arm, because one sample per arm cannot
     separate an effect from the noise above. The pre-registration fixes the numeric noise band
     from the repeats of the no-brief arm, before the version one arm is scored.
   - Until then, the block's presence is not claimed as a benefit anywhere.
2. **False-positive suppression**, prospectively, because no accepted records exist yet. Every
   `PRIOR_FP` verdict is read by a person while they number fewer than about fifty, asking one
   question of each: would fp-check have confirmed it? Any confirmed true positive among them stops
   automatic suppression until the match key is narrowed.

## Constraints

- The RE-call repository is public: no host names, addresses, account names or memo content in
  code, tests, fixtures or this document.
- Serving on the deployment host is frozen until the AML official runs finish (about 2026-10-30).
  RE-call work is built and tested on a branch; deployment waits for the serving sync after that.
  While an official run is live, test suites may run on the deployment host only inside the
  bounds the plan's Global Constraints fix, so they never compete with that run.
- Every new behaviour test carries red proof (fails against a plausible mutation, then passes).
- No test spends API credit unless `RECALL_RUN_PAID_TESTS=1`.
- The local ledger stays the system of record and the write side. RE-call is the read path for
  every client and host, so Claude and Codex runs see the same records. Freshness lag is one
  refresh cycle (hourly at worst), which is accepted.

## Left to the builder

- Tool and CLI argument names beyond those fixed above, and the exact output envelope.
- Whether to add a JSONB expression index; the tenant holds under a thousand records today.
- The symbol resolver's fallback for languages other than Python (file-level anchors, which then
  never suppress unless the whole file is unchanged). The Python resolver is fixed by the anchor
  test vectors.
- The distillation grouping heuristics and the model used to draft lessons, within the rules
  under Lessons.
- Whether historical ledger rows can be re-anchored from archived run files; rows that cannot keep
  commit-level `files` and are excluded from the anchored prior-bugs block.
- Order of landing: the RE-call half first is the natural order, since CCA consumes it.
