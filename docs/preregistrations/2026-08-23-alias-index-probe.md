# Pre-registration: can index-time memo-to-goals aliases rescue the missed queries?

**Date:** 2026-08-23   **Status:** predicted, not yet measured
**Probe:** `scripts/agent_ab_prepare_alias_corpus.py` + `scripts/agent_ab_probe_alias_index.py`,
committed with this record

## The question

If every memo in the benchmark corpus is augmented at index time with task-intent alias queries
generated from the memo's own text, do the recorded queries of `agent-ab-skill-001`'s 15
miss-sessions retrieve their governing memo at top-5, without breaking the 31 sessions that hit?

## Why this probe, and why the direction reversed

The decompose-expansion probe (2026-08-23, same day) measured query-side rewriting at 3 of 15
against a build bar of 5, and showed why verbatim: a rewriter at query time holds exactly the
information the agent held, so it reproduces the agent's blind spot. The informational asymmetry
points the other way: at index time the memo STATES the hazard, so deriving "what task leads
here" is generation from full information. This probe measures that direction with the same
recorded queries, the same top-5, the same stdio transport.

## Design, stated before any number

- **Frozen sources, not the live store.** The live memory directory now contains memos about the
  benchmark itself, naming tasks and governing memos, so it is contaminated as a source. The
  preparation script recovers the 194 sources of the frozen generation from the evidence corpus,
  read-only: a live file is used only when its sha256 still equals the recorded `source_sha256`;
  anything drifted is reconstructed from the generation's own chunks with a joiner LEARNED from
  the sha-verified multi-chunk files, and flagged in `aliases.json`.
- **Two rebuilds in this session's own container** (never the evidence volume): `probe_control`
  from the recovered sources verbatim, `probe_aliased` identical except each memo gains a "Tasks
  that can lead here" section of 5 generated aliases. Separate databases, because the builder's
  generation-reuse check is not tenant-scoped. Same committed builder, chunker, embedder and
  calibration query set for both. Index files stay unaliased. Builds run with
  `RECALL_INDEX_BATCH_CHUNKS=64` and `RECALL_FASTEMBED_BATCH=16`, sequentially, per the standing
  embedding bounds; this corpus is the named local exception (its database is on this machine).
- **The alias generator sees only memo text.** The fixed prompt is committed inside the
  preparation script; the script structurally cannot read the benchmark archive, and every alias
  is recorded verbatim for audit. `anthropic/claude-haiku-4.5`, temperature 0, 5 aliases per memo.
- **Replay**: all 46 admitted on-arm `memory_only` sessions that searched, their recorded queries
  against both rebuilds. Session-level outcome: governing memo in top-5 for any of the session's
  queries.

**Apparatus gate, which can VOID the probe:** the control rebuild must reproduce at least 13 of 15
misses AND at least 28 of 31 hits. Below either bound the rebuild is not the instrument the run
used, no verdict is read from the aliased arm, and the mismatch is investigated under a new
record. This gate exists because the rebuild differs from the original corpus in calibration
identity and in any reconstructed sources, and those differences must be shown small before the
treatment difference means anything.

## What I predict

Mechanism rates, the class I estimate best; the expansion probe's 3 of 15 is the floor any
worthwhile result must clear.

| # | Prediction | Falsified if |
|---|---|---|
| 1 | **Rescue: 7 to 11 of 15** | 3 or fewer, no better than query-side expansion |
| 2 | **`ts-lf-rewrite`: at least 2 of 6 rescued.** Its memo says "scripting an edit in Python"; an alias like "update a file with a python script" is derivable, though the memo text may never say "version" | 0 of 6 |
| 3 | **Retention: at least 28 of 31.** Aliases add competing text corpus-wide; some dilution is the price and more than 3 lost hits means the price is the story | 27 or fewer |

**Decision rule, full partition this time** (the expansion record left a gap between its bands and
paid for it): rescue **>= 7** with retention >= 28, build the index-side feature and preregister
its A/B. Rescue **4 to 6**, mechanism real but weak: no build, one registered iteration on alias
design is licensed. Rescue **<= 3** or retention **<= 27**, index-side aliasing in this form is
dead, and the honest conclusion is that the formulation gap is not closable by generation on
either side at current model quality.

## How it will be measured

```bash
eval "$(scripts/session-db.sh up)"
python -u scripts/agent_ab_prepare_alias_corpus.py --evidence-dsn <evidence> --out benchmarks/artifacts/agent_ab/alias-probe
python scripts/agent_ab_build_corpus.py --source .../sources-control --dsn <session>/probe_control
python scripts/agent_ab_build_corpus.py --source .../sources-aliased --dsn <session>/probe_aliased
python -u scripts/agent_ab_probe_alias_index.py --archive ~/.claude/archive/agent-ab-skill-001 \
  --control-dsn <session>/probe_control --aliased-dsn <session>/probe_aliased
```

## What I already know

- The 15 misses, 31 hits and their queries: [[hazard-query-instruction-result-2026-08-23]],
  archive `~/.claude/archive/agent-ab-skill-001/`.
- Query-side expansion rescued 3 of 15: [[query-side-expansion-reproduces-the-blind-spot]].
- Session extraction verified against the archive before this record: 46 sessions, 31 hit, 15
  missed. The evidence DB holds 194 sources, 1006 chunks, with per-source sha256.
- ⛔ One prompt, one run, one verdict, as before: no rewording queries, no iterating the alias
  prompt against the result. A second prompt is a second registration.

## Confounds I can name now

1. **Calibration differs between rebuilds and from the original.** Each build fits its own
   threshold. Ranks, not verdicts, decide the endpoints, and the apparatus gate bounds what the
   rebuild differences can be hiding.
2. **The alias generator is the agent's own model.** A failure licenses one registered follow-up
   with a stronger model, not a silent retry.
3. **Author overfitting.** I know which four tasks missed; the prompt is generic and every alias
   is auditable, but the indirect tuning risk is the same as in the two records before this one.
4. **Recorded queries, future feature.** Feasibility gate only; a live A/B follows only on a
   build verdict.
5. **Alias sections change chunking.** A memo whose alias section lands in its retrieving chunk
   may also SHIFT what that chunk says; retention is the endpoint that watches the damage.

---

## Result (2026-08-23)

**Status: measured, and VOID by the registered apparatus gate.** Control rebuild: misses
reproduced **15/15** (gate >= 13, passed), hits reproduced **26/31** (gate >= 28, **failed**). Per
the rule above, no verdict is read from the aliased arm, and none of the decision-rule branches
applies.

**The gate failure localizes completely, which is the gate working.** All five lost hits are the
same session family, `ts-raise-on-missing`, whose governing memo `missing-input-becomes-a-clean-null`
is the ONE governing memo among the 20 sources that had to be reconstructed from chunks (the
recovery report showed the chunker transforms text: no joiner reproduced any sha-verified file, so
every reconstruction was known-approximate). The other 174 sources were bit-exact, and every one of
their sessions reproduced, misses and hits alike. Chunk counts: original 1,006, control rebuild
1,016, the drift again localized to the 20 reconstructions.

**Observation, labelled as unregistered because the probe is void:** within the rebuilt
instrument, which is internally valid (same builder, same sources, differing only in the alias
sections), **control and aliased agree on 46 of 46 sessions**. Zero rescues, zero losses. The
alias sections, which read exactly as intended ("Update repository files programmatically using
Python" appended to the CRLF memo), had no measurable effect on any recorded query's top-5 in
either direction. Two candidate mechanisms, undecided here: the appended section merges into an
existing chunk and dilutes rather than creating a retrievable target; and uniform augmentation
raises every memo's goal-vocabulary similarity together, cancelling in the ranking. Deciding
between them belongs to a new record, if one is ever worth writing.

**What a follow-up would need, recorded so the void is not quietly re-run:** exact recovery of
`missing-input-becomes-a-clean-null` (or registered exclusion of its five sessions with the gate
rewritten accordingly), and a design answer to the 46/46 null: separate alias DOCUMENTS pointing
at the memo, rather than appended sections, is the variant the dilution mechanism predicts would
behave differently. Given that both generation-side directions have now measured at or near zero
(query-side 3/15, index-side 0 effect within-instrument), the prior on prompt-generated vocabulary
bridging this gap at current model quality is low, and the record says so rather than proposing a
third variant by reflex.
