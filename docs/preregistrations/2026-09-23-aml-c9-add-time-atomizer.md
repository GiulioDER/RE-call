# C9: the atomic stage built inside Add, verified live

Status: frozen when committed. Nothing below the "Result" heading may be written before the
measurement, and nothing above it may be edited afterwards; corrections are appended.

## Question

C8 (`C8_routed_specialists_grounded_graph`) carries an atomic rescue stage that reads a file
artifact built by hand after ingest (`scripts/build_aml_atomic_rescue_artifact.py`) and bound to
the served corpus fingerprint. The official AML flow calls Add, then Search, with no pause, and
requires each chunk to be persisted and searchable before its HTTP 200 response (live contract,
read 2026-09-23 at https://agentmemoryleaderboard.ai). Every Add changes the fingerprint, and no
one can run the builder in between, so under the official flow C8's stage falls back on every
query. The live C8 reference measured exactly that with no artifact: 187 of 187 attempted,
0 active, 187 fallback.

C9 (`C9_routed_specialists_grounded_graph_atomic`) is C8 with one change: `recall_aml.atomic_views`
builds the `micro` views of each Add request from its own raw windows, embeds them with each scope's
embedder (Code4, and the Context4 specialist) and stores them in an isolated tenant per scope,
before Add returns. Search reads the nearest views by an exact dense query and applies the same
selection and placement. Its defaults are mode `active` and placement `fused` (after fusion, so the
fused top five cannot change; user decision 2026-09-23).

Does C9 activate on every query under the official Add then Search flow with no manual step, does
its selection reproduce the artifact selection, and does the fused placement keep the top five?

## Frozen apparatus

- Host: **VPS3 only.** VPS2 serves the official Multimodal Full run and is not touched in any way.
- Service: an isolated C9 instance built with the C8 reference recipe
  (`~/atomizer-c8/service-setup.sh` on VPS3), changed only in: its own database `atomizer_c9`, its
  own port, variant `C9_routed_specialists_grounded_graph_atomic`, **no** artifact root, **no**
  atomic calibration or pipeline variables, and a **fresh, empty** embedding cache, so every vector
  is computed the way an official run computes it. Served commit: the head of branch
  `claude/aml-atomizer-add` that carries this record, printed by `/version` and recorded in the
  result.
- Corpus: CAMBench coding manifest SHA-256
  `58055df1828b2c1e51bc3c7f9f82e916145c67aa58332f22ce1b86b2d849b814`, 196 sessions, 1,220
  windows, loaded by `load_frozen_corpus` in `scripts/aml_c7_qualification.py`.
- Harness: `scripts/c8_atomizer_live_receipt.py` at `origin/claude/atomizer-c8-study`, the same
  script that produced the C8 live receipt; `ingest` with one worker, then `search` on the dev split
  (153 probes) plus the 34 task prompts, 187 queries in total, the same set the C8 receipt used.
- One ingest, three search passes, the service restarted between passes with only the environment
  changed:

  | pass | environment | purpose |
  |---|---|---|
  | `c9-off` | `RECALL_ATOMIC_RESCUE_MODE=off` | control |
  | `c9-dense` | `RECALL_ATOMIC_RESCUE_PLACEMENT=dense` | parity with the C8 live `on` pass, which used dense placement |
  | `c9-fused` | nothing overridden | the official configuration |

- Parity baseline: the C8 live `on` rows on VPS3 (`~/atomizer-c8/private/`), served commit
  `3953e69d`, artifact `micro`, dense placement.
- Budget: USD 1.00 for this record, inside the program's USD 10 cap.

## Predictions

Written before any C9 service exists. My own record is that I over-predict effect sizes by two to
four times (eleven of twelve earlier predictions too high), so the quality prediction is
deliberately small.

1. **Activation.** In `c9-dense` and `c9-fused`: 187 of 187 queries attempted, active and candidate
   available, 0 fallback, with no artifact file anywhere on the host and no builder run. In
   `c9-off`: 0 attempted. Every Add returns 200 (196 of 196).
2. **View store.** The Code4 view tenant holds **11,749** views, the count the C8 `micro` artifact
   builder produced from the same windows, and the Context4 view tenant holds the same number. The
   raw tenant holds 1,220 windows and no view rows.
3. **Selection parity.** `c9-dense` top-8 ids equal the C8 live `on` top-8 ids on **at least 180 of
   187** queries. Allowed sources of difference: pgvector versus NumPy float32 cosine at ties, the
   Context4 view grouping (per request here, 64-view batches in the builder; 2 of 187 queries
   routed there in the C8 receipt), and graph records recompiled by gpt-4o-mini.
4. **Fused invariant.** `c9-fused` top-5 ids equal `c9-off` top-5 ids on **187 of 187** queries.
5. **Quality direction** (probes, exact gold at rank 8, paired against `c9-off`): gains 2 to 5,
   losses 1 to 4, net 0 to +2. Ranks 1 to 5: no change at all, by item 4.
6. **Latency.** Search p95 in `c9-fused` exceeds `c9-off` by 20 to 150 ms (one extra exact scan
   over about 11,749 rows of 1,024 dimensions).
7. **Spend** under USD 1.00: the gpt-4o-mini anchor compiler about USD 0.15, Voyage about USD 0.4.

## What falsifies it, and what follows

- Any fallback, or fewer than 187 active, in `c9-fused`: C9 does not do what it exists to do. No
  official run with it.
- Any top-5 difference in item 4: the fused placement is wrong. No official run until fixed.
- Parity below 180 of 187: the view-store selection diverges from the measured artifact selection;
  explain every divergent query before any official run.
- Items 5 to 7 inform the decision but cannot block it on their own: item 5 is a small sample, and
  the quality evidence for `micro` is the round-3 confirm and the memory-tenant check.

## Result

Measured 2026-09-23 on VPS3, served commit `dfa833df` (every response carried it), variant
`C9_routed_specialists_grounded_graph_atomic`, database `atomizer_c9`, port 18016, fresh
embedding cache, no artifact root, no builder run. VPS2 was not touched. Ingest 10:43:30 to
11:34:06 UTC (first and last Add in the service log), one worker; passes 12:54 to 12:57 UTC. Private rows stay on VPS3 in
`~/atomizer-c9/private/`; only the aggregates below leave it.

| # | prediction | measured | verdict |
|---|---|---|---|
| 1 | 187 of 187 attempted, active, candidate available, 0 fallback in `c9-dense` and `c9-fused`; 0 attempted in `c9-off`; 196 of 196 Adds | exactly that in all three passes; 196 of 196 Adds, 0 failed, 0 HTTP errors on 561 searches | **held** |
| 2 | 11,749 views per scope; 1,220 raw windows, no view rows in the corpus | 11,749 Code4, 11,749 Context4; 1,220 raw; 0 view rows in the raw or Context corpus tenants | **held** |
| 3 | `c9-dense` top 8 equals C8 live `on` on at least 180 of 187 | **182** of 187; exact@8 membership equal on **187** of 187 | **held** |
| 4 | `c9-fused` top 5 equals `c9-off` on 187 of 187 | **187** of 187 | **held** |
| 5 | probes at rank 8, fused against off: gains 2 to 5, losses 1 to 4, net 0 to +2; no change at ranks 1 to 5 | gains **4**, losses **4**, net **0**; ranks 1 and 5: 0/0 | **held, at the low edge** |
| 6 | search p95 in `c9-fused` exceeds `c9-off` by 20 to 150 ms | `c9-fused` p95 **751.8 ms**, `c9-off` p95 **862.0 ms**: fused was 110 ms **faster** | **falsified** |
| 7 | spend under USD 1.00 | not read from a bill; see below | not measured |

Detail by pass (probes then tasks, paired gains/losses (net) against `c9-off`):

| | @1 | @5 | @6 | @8 | @10 |
|---|---:|---:|---:|---:|---:|
| `c9-fused`, probes | 0/0 (+0) | 0/0 (+0) | 5/3 (+2) | 4/4 (+0) | 3/2 (+1) |
| `c9-fused`, tasks | 0/0 (+0) | 0/0 (+0) | 1/0 (+1) | 1/0 (+1) | 1/0 (+1) |
| `c9-dense`, probes | 4/0 (+4) | 5/1 (+4) | 5/1 (+4) | 4/3 (+1) | 3/1 (+2) |
| `c9-dense`, tasks | 0/2 (−2) | 1/0 (+1) | 1/0 (+1) | 1/0 (+1) | 1/0 (+1) |

The `c9-dense` rows reproduce the C8 live `on` receipt cell for cell (probes 4/0, 5/1, 4/3, 4/1 at
@1, @5, @8, @10 there; the @10 cell has one gain fewer here, 3/1 against 4/1), including the two task losses at rank one
that motivated fused placement. `c9-fused` removes those by construction and gives up the rank-one
to rank-five gains, which is the trade the user chose.

**Where the parity misses come from.** Of the 5 top-8 differences from C8, 2 are the two
Context-routed queries (the view grouping differs, as predicted) and 2 already differ between the
`c9-off` and C8 `off` controls. Exact first-gold rank agrees with C8 on 157 of 187 in the dense
pair and on exactly 157 of 187 in the off pair, so the deeper rank differences are a property of
the base retrieval between the two ingests (the graph records are recompiled by gpt-4o-mini each
time), not of the atomizer.

**Item 6 was wrong in direction.** One sample per pass. The off pass ran first, right after a
restart, and the fused pass last, so a warm cache is a plausible reason the extra exact scan did
not show; that explanation is untested. What the sample does show is that the view scan is not
visible against pass-to-pass noise of about 100 ms on this host.

**Spend (estimated, not billed).** gpt-4o-mini anchor compiler: the same 196-session ingest cost
USD 0.143 over 207 calls in the C8 receipt, so about USD 0.14 here. Voyage: about 1,220 windows
and 11,749 views, each in two scopes, plus the compiled records and 561 queries, on the order of
1.6M tokens, roughly USD 0.3. Total about USD 0.45, inside the USD 1.00 budget.

**Decision, by the rules above.** None of the blocking conditions fired: 0 fallback, 187 of 187
active, the fused top five unchanged on every query, and parity above the bar. C9 does under the
official Add then Search flow what C8 could only do with a hand-built artifact.
