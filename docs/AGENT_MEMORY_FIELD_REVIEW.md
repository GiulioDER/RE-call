# Six agent-memory articles, one paper, read against RE-call

Date of reading: 2026-08-19. Every claim about RE-call below was checked against this worktree
(`claude/agent-memory-recall-research-9ab019`, head `fa6fe04a`, 54 commits behind `origin/master`)
on that date. Where I cite a line number, I read that line today; where the checkout is behind
master, I say so rather than assume the line survived.

## What I read, and whose interest it serves

Weighting a claim by who benefits from it is not cynicism here, it is necessary: four of the seven
sources are marketing for a commercial memory product, and the two most technically load-bearing
pieces are the two with the least to sell.

| # | Source | Author | Interest | Date | Engagement | What it is evidence of |
|---|---|---|---|---|---|---|
| A1 | Everything it remembers has the same authority | Edward Izgorodin | Mnemoverse, commercial memory engine | 2026-08-19 | 2 reactions, 4 comments | The closest independent statement of RE-call's own thesis |
| A2 | Short-term, long-term and what's just a database | Multigrid | multigrid.ai, commercial | 2026-08-07 | 0, 0 | The best schema and compaction checklist of the set |
| A3 | Agent memory in TypeScript | Gabriel Anhaia | book series and Hermes IDE | 2026-08-06 | 0, 0 | Implementation hygiene, little architecture |
| A4 | Not merely storage and retrieval | Gaurav Dadhich | Maximem.ai founder, closed source | 2026-07-25 | 1, 2 | Abstract of the paper below |
| P1 | Agentic Context Management (arXiv 2607.21503) | Gaurav Dadhich | same | 2026-07 | n/a | A vocabulary, and two headline numbers whose configuration is fully stated and whose system is not runnable by anyone else. See the correction at the head of Part 5 |
| A5 | Agent memory architecture | Amar Dhillon | none apparent | 2026-04-23 | 0, 0 | A taxonomy tutorial. Lowest information density of the set |
| A6 | Agent memory v2, seven rules after the poisoning | israelhen153 | none apparent | 2026-06-23 | 2, 2 | The most useful piece here, and the only one honest that nothing is built yet |

Two engagement facts worth carrying: A1 and A6 both got substantive technical comment threads that
changed the authors' positions in public, and in both cases the comments are better than the
articles. A5, A3 and A2 got zero comments, so nothing in them has been stress-tested by anyone.

I read the full comment threads on A1, A4 and A6 (eight comments total, fetched through the dev.to
API rather than the rendered page). Re-fetch any of them with:

```bash
curl -sS "https://dev.to/api/comments?a_id=4405699"
```

## Headline

The field has independently converged on most of RE-call's design, which is external validation
worth having and worth saying out loud. The genuinely new material is narrow and it is concentrated
in one place: **A6's Rule 4, that confidence orders a pile but must not promote anything, and that
promotion has to come from independent corroboration instead.** That is the same conclusion
RE-call's own abstention measurements reached from the opposite direction, and neither party has
acted on it. That is the research idea in this whole reading list.

Everything else is either already shipped here, cheap to add, or a different product.

One thing I did not expect to find. Checking a claim in Part 3 turned into a measurement, and the
measurement outranks the reading: **zero of the 152 memos in RE-call's own memory store, and zero of
the 59 documents in this repository's `docs/`, declare a validity window or a supersession edge.**
The feature the README leads with is switched off in the corpus its author uses daily. Part 2b has
the numbers and the re-measure command.

## Part 1: where the field arrived at what RE-call already does

Independent convergence is the cheapest external validation available, and four of these came from
authors who have never seen this repository.

| Claim, and who makes it | RE-call today | Verified |
|---|---|---|
| Separate "when a fact was true" from "when we learned it" (A1, citing SQL:2011 bitemporal) | Implemented, and taken further: `first_indexed_at` is the transaction-time axis and `not_yet_known` is its own verdict, distinct from `not_yet_valid` | `recall/types.py` (`Verdict`), `recall/trust.py` |
| Close a fact instead of replacing it; replacing is the destructive operation (A1) | This is what `supersedes` plus a retained predecessor row is. The superseded memory stays retrievable with its successor named | `recall/frontmatter.py`, `recall/trust.py` (`abstain_reason`) |
| The provenance tag must survive retrieval into the prompt, because the read path silently strips out-of-band metadata (A6 Rule 5, the thread's deepest cut) | Implemented, with the same reasoning arrived at independently: `marked_text` puts an in-band warning in front of the text precisely because LangChain and LlamaIndex render content and drop metadata | `recall/trust.py` (`marked_text`) |
| A third state for "the verifier was down", because binary admit and reject collapses exactly when the checker fails (A6 Rule 2) | `unverified` is exactly this verdict, and its docstring makes the same argument for why it must not be folded into `low_confidence` | `recall/types.py` (`Verdict`) |
| The unverified path fails closed (A6 Rule 3) | Strict mode raises `TrustRefusal` and produces no result object at all | `recall/types.py`, `recall/trust.py` |
| Cap what retrieval injects; five facts, not fifty (A2, A3) | `k` defaults to 5 and is clamped down by the process profile, never raised per request | `recall_mcp/server.py` (`recall_search`) |
| Connecting a memory tool does not make an agent use it; the fix is a standing instruction in the project rules (A1, presented as the single most valuable takeaway in that article) | `recall setup` scaffolds a `CLAUDE.md` section and a starter `memory/MEMORY.md`, then indexes it | `docs/archive/CLAUDE_MD_MEMORY_SCAFFOLD_DESIGN.md`, status implemented 2026-08-12 |
| Keep the store human-inspectable as a plain list, because reading the memories is the only reliable way anyone has found to catch poisoning (A2) | Memories are files. Reading them is `cat` | by construction |

The last row deserves more weight than it looks like it deserves, and Part 6 returns to it.

## Part 2: the one place the articles and RE-call's own numbers agree that RE-call is wrong

A6's Rule 4 is the strongest argument in the set. Stated plainly: the model's confidence was high
when it hallucinated, so a confidence threshold re-admits the exact bug it was installed to stop.
Confidence is allowed to decide which unverified claim to check next. It is not allowed to promote
anything. Promotion must come from a *different* source, and sources are ordered by kind rather
than by score, so a tool result outranks a confident model even when the model is more confident.

RE-call reached a compatible conclusion by measurement, from the other end, and the two have never
been put side by side:

- Far-gap abstention works: accuracy 1.00 on PEPs, 0.89 on the real corpus.
- Near-miss abstention fails: false-abstain 0.481 on LongMemEval, and six candidate signals all
  score AUC at or below 0.753, with the best one's 95% interval topping out at 0.826, below the
  roughly 0.90 a usable gate needs.
- The conclusion recorded there is that the bar is *excluded* rather than merely unproven, because
  every alternative measured worse.

Source: `docs/EVIDENCE.md` and `results/FINDINGS.md` sections 9 and 10, as of this checkout.

Read together, these say the same thing twice. A scalar score, whether a model's self-reported
confidence or a calibrated cosine, cannot separate a near-miss from a hit, and tuning it harder is
not the missing move. A6 names the move that is structurally different: corroboration. RE-call has
measured that it needs one and has not tried this one.

That is the research lane I would open.

## Part 2b: an unplanned measurement, and it changes the ranking

While checking item 2 below I counted how many memos actually declare the metadata the trust layer
runs on. Measured 2026-08-19, on this machine:

| Corpus | Markdown files | With frontmatter | `valid_from` | `valid_until` | `supersedes` |
|---|---|---|---|---|---|
| RE-call's own memory store (`~/.claude/projects/C--Users-gde00-Documents-recall/memory`) | 152 | 142 | 0 | 0 | 0 |
| This repository's `docs/` | 59 | n/a | 0 | 0 | 0 |

The 142 files that do carry frontmatter carry exactly three keys between them: `name`,
`description`, `metadata`. Re-measure:

```bash
python -c "import pathlib,re; fs=[p for p in pathlib.Path.home().joinpath('.claude/projects/C--Users-gde00-Documents-recall/memory').rglob('*.md')]; t=[p.read_text(encoding='utf-8',errors='replace') for p in fs]; print(len(fs), sum(bool(re.search(r'^valid_until:',x,re.M)) for x in t), sum(bool(re.search(r'^supersedes:',x,re.M)) for x in t))"
```

**What this means.** Validity-aware retrieval is the first row of the README's capability table, and
on RE-call's own dogfood corpus it is inert: with no validity window and no supersession edge, every
hit resolves to `Validity(None, None, None)` and the verdict layer can only ever return `ok`, or
`unverified` in degraded mode. The distinguishing feature is switched off, in the one corpus its
author uses daily.

This is not a bug and I am not claiming the corpus is wrong. It is the authoring problem, and it is
the exact twin of the observation A1 calls its single most valuable takeaway. A1 says connecting a
memory tool does not make an agent use it. The measurement above says **adding a validity field does
not make an author fill it**, and RE-call has already paid for the first lesson and not yet noticed
the second.

`benchmarks/check_p1_supersession_density.py` encodes precisely this reasoning as a pre-registered
kill gate for LOCOMO: below a floor of explicitly revised state, the supersession mechanism cannot
move an aggregate score no matter how well it works. The same gate, pointed at the dogfood corpus,
reads zero. That is worth knowing before spending anything on new validity semantics, and it demotes
item 2 and item 5 below an item that did not exist when I started ranking.

### Item 0: make the scaffold write validity metadata, and measure whether it survives

The scaffold already exists and already writes memory files (`docs/archive/CLAUDE_MD_MEMORY_SCAFFOLD_DESIGN.md`).
It does not teach the format that makes those files trustworthy. Extending it to emit `valid_from`
and a `supersedes` convention, and to instruct the agent to close a memo rather than replace it, is
a small change to a shipped feature.

**Pro.** It is the cheapest possible route to a corpus where the trust layer does anything at all,
it needs no new engine work, and it produces the fixture that items 1, 2 and 5 all need and none of
them currently have. It also converts a documentation convention into a measurable outcome: the
count above becomes a metric that should climb.

**Con.** An agent told to stamp validity on everything will stamp it wrongly, and a wrong
`valid_until` is worse than an absent one, because absent reads as "unknown" and wrong reads as
"judged". The honest version writes `valid_from` (which is always knowable) and `supersedes` (which
is a deliberate act), and leaves `valid_until` unset by default rather than inviting a guess.

**Cost:** 1 to 2 days, plus a re-measurement of the table above 30 days later, which is the only
part that proves anything.

### Item 0: done 2026-08-19, and what the measurement showed

Implemented in `recall/setup.py` (`_claude_md_block`, `_memory_md_starter`, `scaffold_memory_index`),
with tests in `tests/test_setup.py`. The scaffolded `CLAUDE.md` section now tells the agent to stamp
`valid_from`, to leave `valid_until` unset unless a real end date is known, and never to edit or
delete a memo whose fact has changed but to write a successor carrying `supersedes:`. The starter
`memory/MEMORY.md` teaches the same shape, with the example date injected rather than left as a
placeholder, because `recall/index.py` calls `validity_bounds` and fails fast: an unfilled
`<YYYY-MM-DD>` would break `recall_index` on the directory the wizard indexes seconds later.

**Verified against a live pgvector store, not asserted.** Two memos written from the template, the
second closing the first, indexed with the offline hashing embedder and evaluated through
`recall.trust.evaluate`:

| Template | Edges read back from the store | Verdicts |
|---|---|---|
| Before (`name`, `description`, `metadata`) | none | both memos `ok`, and the **stale one ranked first** |
| After | `{package-manager-npm.md: package-manager-pnpm.md}` | current memo `ok` and first, old memo **`superseded`** with its successor named |

That is the inert case and the working case, side by side, on the same query. `recall lint` returns
0 errors and 0 warnings over the whole scaffolded directory including the supersession edge.

**One defect found and fixed on the way.** Teaching the `supersedes` key put the word into
`MEMORY.md`, and `closure-marker-unlinked` fired on the file the tool had just written: `recall
setup` produced a corpus that `recall lint` complained about. Two changes came out of that, one
accepted and one rejected after measuring:

- **Accepted.** `recall/lint.py` now excludes fenced code blocks before searching for closure
  markers, via `prose_only`. The diagnostic says "body prose", and a fenced block is a sample, not
  an assertion. Blast radius measured over both corpora to hand: **zero** of 152 memos in the memory
  store lose a warning, and exactly one file in `docs/` does (`ENVIRONMENT.md`, whose marker is the
  comment `# deprecated serving fallback` inside a shell fence), which is the false positive.
- **Rejected.** Extending the same exclusion to inline code spans would have silenced **4 of 6**
  warnings in the memory store and **10 of 26** in `docs/`. A memo writing "this `supersedes` the
  old approach" is asserting a relation, so that suppression is a real loss rather than a false
  positive, and the measurement is what separated the two cases. The starter was reworded instead:
  the mechanism is explained in `CLAUDE.md`, which is where instructions belong, and `MEMORY.md`
  keeps a fenced example and stays an index rather than becoming a manual.

**Still unmeasured, and this is the part that proves the item.** Whether authors and agents actually
fill the fields. Re-run the Part 2b count in 30 days (2026-09-18); the number to beat is zero.

### Result, measured 2026-08-20, the day the change merged

**The prediction above is left exactly as written. This is appended under it, not substituted for
it, because the interesting part is that the success criterion was wrong rather than the number.**

```
recall memory store    172 files | valid_from 0 | valid_until 0 | supersedes 0
recall docs/            70 files | valid_from 0 | valid_until 0 | supersedes 0
```

Zero, everywhere, as at the baseline. The denominator moved though: **152 files at baseline, 172
now, so twenty memos were written after the change and not one carries validity metadata.** Five of
those twenty were written by the agent that implemented the change, hours afterwards.

**That is not evidence about adoption, and reading it as such would be the error this document keeps
warning about.** Checking whether the mechanism can reach the measured corpus at all:

- The store at `~/.claude/projects/<slug>/memory/` is **Claude Code's** memory directory. It carries
  no `recall setup begin` marker, so `recall setup` has never scaffolded it and never will.
- Its `MEMORY.md` mentions `valid_from`, `valid_until` and `supersedes` **zero** times.
- This repository has **no** `memory/` directory and **no** scaffold block in its `CLAUDE.md`.

The memo format in that store is dictated by the operator's global `CLAUDE.md` memory contract,
which the change does not touch. So **the 2026-09-18 re-measure would have returned zero by
construction**, and the null would have been read as "authors do not fill the fields" when the true
cause is "the instruction never reaches them". A measurement whose result is fixed in advance by its
own apparatus is not a test.

**What the change does do is unaffected, and was verified separately.** A memo written from the new
template carries `valid_from` into the store and earns a `superseded` verdict against a live
pgvector index. That holds for its actual scope: a project that runs `recall setup` and gets a fresh
`memory/MEMORY.md`. Nothing above this line is retracted; only the criterion for proving it is.

**Two ways to make the claim testable, neither yet done:**

1. **Point the instrument at a corpus the scaffold governs.** Run `recall setup` into a project, use
   it for real work, and count there. Smaller and slower-moving denominator, but it measures the
   mechanism instead of a corpus the mechanism cannot see.
2. **Extend the mechanism to this corpus.** Add `valid_from` and the close-rather-than-replace rule
   to the memory contract in the operator's global `CLAUDE.md`, which is what actually governs this
   store. That is an operator decision about their own standing instructions, not a code change.

Until one of those happens, item 0 is **a shipped and verified improvement with no adoption
evidence**, and it should be described that way rather than as pending a September measurement.

## Part 3: candidate work items, ranked

Each item states the source, what exists today, the honest case against, and a test that could
falsify it. Costs are my estimate in session-days for a first measurable version, not for a
shipped feature.

### 1. Corroboration as a second admission axis, independent of confidence

**Source:** A6 Rule 4, sharpened by A1's promotion path (human correction, merged pull request,
repeated success).

**Today:** admission is a calibrated threshold on a single score. `recall/promotion.py` exists but
it promotes *benchmark arms*, not memories; it has nothing to do with a memory earning authority.

**The idea:** ask a second question at retrieval time. Does an independent memory in the corpus,
from a different source file and a different write event, support the same claim? Count of
independent supports becomes a feature alongside confidence, and a new verdict (`uncorroborated`)
becomes available for a hit that scores well and stands alone.

**Pro.** It attacks the one failure RE-call has measured and then excluded a fix for. It is
structurally independent of cosine, so it can carry information the six tested signals could not.
It fits the existing verdict vocabulary rather than adding a parallel system. `recall/truth_extraction/`
already has claim extraction machinery to build on. And it is the axis a competitor's public design
converged on without knowing this repository exists.

**Con.** Corroboration and similarity are correlated in an embedding store, so the new feature may
buy far less independent signal than the argument promises; that is the outcome to expect and the
reason to preregister. It introduces a new false-abstain mode, because a true fact recorded exactly
once is uncorroborated by construction, and a memory store is full of those. It costs a second
retrieval pass per query. And "independent" is doing heavy lifting: two memos written by the same
agent in the same session are not independent evidence, so the corroboration counter has to know
about write cohorts, which RE-call does not currently record (see item 6).

**Cost:** 3 to 5 days to a measurable feature on existing fixtures.

**Test that would kill it:** on the LongMemEval near-miss set, does a corroboration-count feature
push AUC above 0.826, the upper bound of the best signal already tested? If it lands inside the
existing interval, it is not a new axis, it is a restatement of similarity, and the honest move is
to record that and stop.

### 2. Event-conditioned invalidation, next to date-based `valid_until`

**Source:** Tae Kim's comment on A4, which is better than A4. A fact's boundary should be set at
write time based on what event would make it false, not on how old it is. A6's decay-on-contradiction
is the same instinct.

**Today:** `valid_until` is a timestamp. Frontmatter carries `title`, `supersedes`, `valid_from`,
`valid_until` and nothing else.

**The idea:** an optional `invalidated_by` field naming the condition, not the date. A checker
flags memos whose condition looks met and queues them for a human to supersede. No automatic
demotion.

**Pro.** It fixes a real authoring problem: nobody knows the date a decision stops holding, so
`valid_until` goes unset and the memory never expires, which is the failure A2 describes as a store
converging on contradicting itself. Authors usually *do* know what would falsify a memo. It stays
inside the file-plus-frontmatter model, so it costs no new storage. Flagging rather than demoting
keeps it fail-safe.

**Pro, second order.** It is measurable today, before building anything: count how many memos in
the dogfood memory corpus carry an unset `valid_until`. If that number is small the whole item is
unnecessary.

**Con.** Evaluating a condition needs either a query against the corpus or a human, and the first
adds a class of wrong answers to a layer whose whole selling point is that it does not guess. It
adds a second way for metadata to be malformed, and `invalid_metadata` is already a verdict.
Nothing here is free of the risk that authors fill the field with prose nobody can evaluate.

**Cost:** 1 day for the measurement, 2 to 3 days for the field plus a flagging checker.

### 3. Expose `known_as_of` through MCP, the CLI and the adapters

**Source:** A1's two-time-axes section, which presents this as the capability nobody has.

**Today:** implemented end to end in the engine and tested. `known_as_of` appears in `recall/store.py`,
`recall/trust.py`, `recall/reasoning.py`, `recall/eval/longmemeval_perq.py`,
`benchmarks/membench/recall_temporal.py` and three test files. It appears in **zero** files under
`recall_mcp/`, `recall_interop/` or in `recall/cli.py`. Re-measure:

```bash
grep -rln "known_as_of" --include=*.py recall_mcp/ recall_interop/ recall/cli.py
```

**Pro.** The cheapest item on this list by a wide margin. The hard part is built, tested and
documented. "What did we believe on the day we made this decision" is a question no competitor in
this reading list can answer, and right now no RE-call user can reach it either.

**Con.** It widens the tool signature, and a parameter nobody passes is a maintenance cost with no
return. Worth confirming the demand is real before shipping it in three places rather than one.

**Cost:** under a day for the MCP tool and the CLI flag.

### 4. Return contradictions as a pair, not as an ordered list

**Source:** A1, which argues a contradiction is information and resolving it silently by embedding
distance is backwards.

**Today:** partially done, and split between surfaces. The MCP `recall_search` result carries every
hit with its verdict, `superseded_by` and validity window, so the caller can see both sides. The
LangChain and LlamaIndex adapters do the opposite: `servable_hits` filters to trusted hits unless
the caller opts in (`recall/trust.py` (`servable_hits`)). And `abstain_reason` already names the successor when
the best candidate is superseded, which is the single most useful half of this.

**Pro.** An explicit `contradictions` block (current claim, superseded claim, both windows) turns
an implicit ordering into the product's actual story. It is a presentation change over data that
is already computed.

**Con.** A1 concedes the cost honestly: preserving contradictions spends context budget. Worse for
RE-call specifically, handing a reader model two conflicting memories and hoping it reasons well is
the exact behaviour this library exists to avoid; the measured position here is that refusing is
safer than delegating ambiguity. So this should be off by default and framed as a debugging and
audit surface rather than a retrieval default.

**Cost:** 1 to 2 days, mostly deciding the default.

### 5. An authority tier on memories, and the revocation field

**Source:** A1's evidence-versus-policy split, plus two comments that are sharper than the article.
Max Quimby proposes a promotion boundary, so an episode cannot silently graduate into policy. Reid
Marlow proposes a third field on every promoted item naming *who may revoke it*, because old policy
survives by sounding official.

**Today:** nothing. Every memory is the same kind of thing, distinguished only by validity and
score. RE-call's own thesis is that not everything deserves the same authority, and the schema does
not encode any authority distinction.

**The idea:** an optional `authority` field in frontmatter with a small closed vocabulary, used as
a ranking and refusal prior, so a single episode cannot outrank a merged decision record.

**Pro.** Cheap, additive, and it fills the gap between what RE-call claims and what its schema
encodes. It is the one structural idea in A1 that RE-call has not already built.

**Con, and it is a real one.** A tier that changes nothing measurable is decoration, and this one
is easy to add and hard to validate: it needs a fixture where an episode and a policy genuinely
compete, and the memory corpus may not contain enough such pairs to measure on. A mislabelled memo
becomes a ranking exploit. And a closed vocabulary chosen now will be wrong later.

**Cost:** 1 day for the field, and the fixture is the real work.

### 6. Run-scoped provenance, so a bad cohort can be revoked together

**Source:** A2, which recommends keeping the run and step that wrote each fact precisely so a bad
fact can be traced to its run and its whole cohort deleted together.

**Today:** `recall_forget(sources)` deletes by source. `Provenance` carries source, file, ord and
`indexed_at`. There is no write-event identity, so the blast radius of one bad authoring session
cannot be expressed.

**Pro.** This is incident response, which is a thing RE-call's audience actually has to do, and it
is the natural companion to item 1, whose corroboration counter needs to know that two memos from
one session are not independent evidence. Two features, one field.

**Con.** RE-call indexes files rather than accepting writes, so "run" only exists if the writer
stamps it, which means the value depends on the scaffold cooperating and is zero for corpora
ingested from elsewhere. Partial coverage of a provenance field is worse than none if callers treat
its absence as clean.

**Cost:** 2 days including the forget filter.

### 7. A consumer-side gate at the tool-call boundary

**Source:** A6 Rule 6, and its best detail: the boundary that matters is not the row read, it is the
tool call, because a conclusion derived from unverified data can trigger an irreversible action two
hops downstream.

**Today:** RE-call returns verdicts and advice. Nothing enforces them. A caller who opts into
untrusted hits gets a warning in the text and no other obstacle.

**Pro.** It converts advice into a guarantee, which is the difference between a retrieval library
and a trust layer. The transitive-use argument is the part nobody else in this reading list saw.

**Con.** RE-call is a library, not a harness. Enforcing at someone else's tool-call boundary means
either shipping a base class people must inherit, which is a framework, or documenting a pattern,
which is what the advice field already is. I would ship a small helper in `recall_interop` and a
documented pattern, and explicitly not a framework.

**Cost:** 2 days for the helper and the doc.

### 8. Per-category half-lives, with re-verification only where it is about to be used

**Source:** the A6 comment thread, where the author's reply is better than his article. Decay is
per-category rather than uniform, categories declare their decay function at schema time, and
re-verification fires at retrieval only for rows that are both below their category threshold and
in the current retrieval set. You pay the verifier on the rows you are about to use.

**Pro.** The cost-bounding rule is genuinely good and generalises: verify what you are about to
serve, not the database.

**Con, and it is decisive for RE-call today.** A verifier in the retrieval path is an LLM call in
the memory layer, and "no RE-call memory-layer LLM calls" is a measured, published advantage in the
head-to-head. Adopting this as designed would spend the thing RE-call currently wins on. Keep the
cost-bounding rule as a principle; do not put a model in the loop to satisfy it.

**Cost:** not recommended in this form.

### 9. Scope hierarchy below the tenant

**Source:** A5's actor-centric memory, and P1's organizational scope hierarchy.

**Today:** `tenant_id` is the isolation boundary and `source` is the only filter below it.

**Pro.** Multi-user deployments will want an actor scope inside a tenant without paying for a
tenant each.

**Con.** Row-level security and tenant routing already exist and work; a second scope axis is
schema and policy surface for a demand nobody has expressed yet. This is a "when a user asks" item.

**Cost:** not now.

## Part 4: what I would decline, and why

**Compaction and context-budget management (A2, A3, and P1's compacting primitive).** A2's
compaction checklist is the best writing in the set: drop old tool outputs first but leave a stub
so the agent does not repeat the action, drop superseded results outright, summarise the middle
into structured findings rather than prose, never drop the system prompt or the task statement, and
compact in large infrequent steps because rewriting the prefix invalidates the prompt cache and
makes compaction a paid event rather than a saving. A3 adds the two implementation bugs worth
knowing: never split a tool-use and tool-result pair, and feed the previous summary forward so the
tenth compaction is not a summary of a summary.

All of it is correct, and none of it is RE-call's product. Compaction lives inside the agent
harness, over the transcript, and building it means competing with every framework's built-in
compaction on their ground with no trust angle. The one piece that *is* RE-call-shaped is P1's
notion of *validated* compaction, where fidelity after compaction is checked rather than assumed:
checking whether a compacted context still supports a claim is an entailment question, and
entailment machinery is already here. That is a narrow, honest read of a broad claim, and it is the
only part of the compaction story I would spend anything on.

**Anticipation and prefetch (P1's fifth primitive).** Speculative retrieval spends tokens on
guesses about the next turn. RE-call's entire thesis is that not guessing is the feature. I would
state that tension in public rather than build the primitive.

**Chasing 92% and 93.2%.** Still declined, but on narrower grounds than I first gave: see the
corrected Part 5. Their LoCoMo figure excludes the adversarial class, so it is a different quantity
rather than an unchecked one. Beyond comparability, the memory index records the current
lane as EnterpriseRAG-Bench, where the top-five threshold is 61.03 against a shaped RE-call
baseline of 46.16. Opening a LongMemEval headline chase would be a second front, and the recent
screens on that lane were measured and rejected rather than parked, which is the wrong moment to
split attention.

**A5 entirely.** It is a competent taxonomy tutorial (memory vault, actor versus agent, semantic
and episodic and procedural) with no failure modes, no measurements and no comments. The only
transferable idea is procedural memory as a first-class type, meaning tool definitions and policies
stored as memory, which overlaps item 5 and is better taken from A1.

## Part 5: what P1's headline numbers are, and are not

> 🔁 **Corrected 2026-08-20, and the correction goes against me.** Everything below the marker
> was written from the arXiv abstract page, because I never read the paper body, and it was unfair
> in a specific and checkable way. The original text said "the paper states the reference
> implementation is closed source". **The paper says no such thing.** It says the *mechanism
> internals* are proprietary, which is a narrower claim about components, and its evaluation
> harness is **public and open** (`maximem-ai/memory_and_context_eval_harness`), with the
> methodology and per-category counts published too. The original also said the configuration was
> "described in prose": it is Table 2, which specifies dataset, split, answer model, judge,
> retrieval configuration, harness, artifacts and run tag per benchmark. I sourced the closed
> source claim from the author's dev.to comment and attributed it to the paper, which is exactly
> the sourcing error the rest of this document is fussy about. What follows replaces it.

P1 reports **92.0% on LongMemEval** (460/500, full official 500-question set, all six categories)
and **93.2% on LoCoMo** (categories 1 to 4), for a reference implementation called Maximem Synap.
Both benchmarks already have harnesses here: `recall/eval/longmemeval.py` and `recall/eval/locomo.py`.

**The paper answers, unprompted, four of the five questions `results/FINDINGS.md` sections 9e and 9f
had to reverse-engineer from Mem0's harness.** Its Table 2 states the answer model (`gpt-5-mini`),
the judge (`gpt-5-mini`, binary CORRECT/WRONG against gold), the official distributions with no
custom subsets and no relabeling, the harness repository, and the run tag. Section 6.3 reports the
weak categories as plainly as the strong ones, and the residual errors concentrate in LongMemEval's
multi-session category at 75.2%.

**On LoCoMo category 5 it is more explicit than Mem0 ever was, and it corroborates our own finding.**
The paper excludes category 5, says so in the setup rather than in a footnote, states that inclusion
or exclusion "moves the headline score by ten points or more", calls that the most common source of
incomparable LoCoMo numbers, and names the convention it is following: the original paper, Mem0 and
Zep. That is independent third-party confirmation of what §9f established by reading Mem0's code.

**It also adopts the comparability discipline this repository argues for.** It refuses head-to-head
presentation outright, and its Table 3 lists each vendor's self-reported figure beside the answer
model that produced it, labelled as the published landscape and explicitly "not a controlled
comparison". Appendix C is a reproducibility statement committing to state dataset, split, question
counts, answer and judge models, retrieval configuration, harness commit SHA and run date for every
result, and to say directly when a result cannot be re-run by a third party.

### What still holds, stated narrowly

1. **The harness is reproducible; the result is not.** The evaluation code is open and the datasets
   are public, but the system under test is not runnable by anyone else, so a third party can
   re-run the harness and cannot reproduce 92.0. The paper concedes the adjacent half itself: per-run
   artifacts (answers, retrieved context, judge verdicts) are "available on request rather than
   published".
2. **The categories differ from ours, so the numbers still are not ours to sit beside.** Their
   LoCoMo figure excludes the adversarial class by convention. Category 5 is the axis RE-call exists
   to measure, so 93.2% and our 0.444 remain different quantities, for the reason §9f already gives.
3. **The answer model is part of the result.** The paper says this plainly and uses it as an
   argument in its own favour: 92.0 came from a *smaller* answer model than the strongest competitor
   configurations, which it reads as evidence the gain sits in the context layer.

### What this changes for RE-call

The useful conclusion is the opposite of the one I first wrote. This is not a marketing figure to
wave away; it is a competitor publishing its configuration to roughly the standard this repository
holds itself to, and independently confirming the category-5 problem that §9f documents. Two
consequences worth acting on. Their reported weakness, multi-session reasoning at 75.2%, is the
regime our own LongMemEval work already lives in. And their exclusion of the abstention class is
the clearest available argument for RE-call's positioning: the headline numbers the field compares
on are computed with the axis RE-call optimises removed from the denominator.

## Part 6: three positioning notes that cost nothing

**1. A competitor stated RE-call's thesis, in public, better than the README does.** A1's core
sentence is that the problem is not how much an agent remembers, it is that everything it remembers
carries the same authority. That is this library's premise, written by someone selling a different
product, which makes it the most useful third-party framing available.

**2. Revocation authority is free here and expensive for store-based competitors, and the
competitor said so first.** In the A1 thread, a commenter proposed that every promoted memory needs
a named revoker, and Izgorodin's reply concedes something sharp: when policy lives in a rules file
in a repository, revocation authority already exists, because it is whoever can merge, and git
supplies the revoker, the history and the date it stopped being true at no cost; the moment policy
migrates into a memory store, that authority quietly disappears and no schema asks for it back.
RE-call indexes files in a repository. It therefore inherits the revoker, the audit trail and the
human review gate for free, and this is nowhere in the positioning. That argument is worth a
paragraph in the README and possibly a post of its own.

**3. The scaffold is a shipped feature that the capability table does not mention.** A1 closes by
saying that if you take one thing from the article, take the standing instruction, because
connecting a memory tool does not make an agent use it, and it is both the cheapest fix and the one
most often missing. `recall setup` already scaffolds exactly that. The README capability table lists
six rows and this is not one of them. Verify before writing it up, since this checkout is 54 commits
behind master:

```bash
git show origin/master:README.md | grep -n "scaffold"
```

## Part 7: what I would do next

Two things, in this order, and not the seven above.

**First, item 0, because the measurement in Part 2b outranks everything I had planned to recommend.**
Zero of 152 memos in RE-call's own memory store declare a validity window or a supersession edge, so
the capability the README leads with is inert on the corpus its author uses every day. No new
validity semantics are worth designing on top of a corpus that declares none, and item 1's
corroboration counter needs exactly the write-event and supersession structure that is missing. This
is authoring work, not engine work, and it is cheap.

**Second, item 1**, which is the only candidate here that touches a failure RE-call has measured,
published and then closed the door on, and A6's Rule 4 is an argument for reopening it from a
direction that was never tested.

Per the standing rule, that means a preregistration before any measurement, committed before the
run. It is written and committed alongside this document:
`docs/preregistrations/2026-08-19-corroboration-abstention-axis.md`.

The prediction written down there is the uncomfortable one, and it is a null. Corroboration
count is correlated with embedding similarity in a vector store, so my expectation is that its AUC
on the LongMemEval near-miss set lands inside the interval of the best signal already tested, at or
below 0.826, and that the honest result is a recorded negative that closes the question properly
rather than a seventh signal that fails the same way as the first six. If it lands above, the
abstention gate has a second axis and that is a genuinely new result.

Item 3 is a separate, unrelated, sub-day change that needs no preregistration because nothing about
it is a measurement: `known_as_of` is built, tested, and unreachable from every surface a user
actually touches.
