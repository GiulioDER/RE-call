# Pre-registration: does C9 keep its Coding retrieval with timestamped windows?

**Date:** 2026-09-24   **Status:** predicted, not yet measured. Measurement waits for the user's
explicit go.

## The question

`2026-09-24-aml-c9-reader-dates.md` found that C9 with `content_only_windows=False` (renderer
`timestamp-role-content-v1`) wins back the 17.5 LoCoMo temporal points C9 loses when the reader
sees `content` alone, at -0.26 overall. Its decision rule recommended that build for the next
Textual run, subject to this check, because **one endpoint serves both tracks** and
`message-content-only-v1` came from the Coding parity release
(`2026-09-20-aml-code4-exact-parity-official.md`), not from a Textual decision.

On the frozen Coding screen, is C9 with timestamped windows **non-inferior** to C9 as served?

## What changes, and how the input is made realistic

The existing Coding harnesses (`scripts/aml_c7_qualification.py` and the C8/C9 receipts built on
it) flatten each session into ONE `role: user` message with no timestamp. On that input the flag
would add three words per session and could not show what it does to a real AML Coding Add, which
carries a trajectory of messages with roles and timestamps.

So this check sends **one message per transcript event**, with the event's `role` and its `ts` as
Unix milliseconds (`scripts/aml_c9_coding_window_check.py`, `event_messages`). Each event's text is
exactly what `_render_transcript` takes from it, normalised the same way, so:

- with the served renderer the stored text is **byte for byte the historical screen's**: joining
  the event texts with single spaces gives the flattened session back, which the script checks for
  all 196 sessions before a collect starts (measured locally: 0 mismatches);
- with `--timestamped-windows` each event also carries `timestamp: … role: … content:` inside the
  window. Measured locally over the corpus: 2,281 events, all with a role and a timestamp, 142,359
  words, and the prefix adds 11,405, **+8.0%**.

## Arms

All three are C9 built in process at the served `recall_aml` (identical to `385c6074`), variant
`C9_routed_specialists_grounded_graph_atomic`, graph on, atomic stage active, each in a fresh
database of its own, with per-event input.

| arm | renderer | purpose |
| --- | --- | --- |
| **K0** | `message-content-only-v1`, as served | baseline |
| **K0b** | the same | replicate of K0: the Add-time compiler is not deterministic, so this is the noise floor |
| **K1** | `timestamp-role-content-v1` (`content_only_windows=False` only) | candidate |

## Corpus and scoring

- Agent Memory Bench coding corpus, `corpus/manifest.json` SHA256 `58055df1…`, 196 sessions whose
  hashes the loader checks, 34 task prompts from `tasks/*/task.json`, synthetic only per the
  corpus `data-policy.json`. One user per arm; sessions added in sorted path order.
- One Search per task prompt, top_k 100. Relevance is at session level: a returned item is
  relevant when its `session_id` lies under `sessions/<task_id>/`, exactly as the C6 and C7
  qualifications scored (`compare_coding_rankings`).
- Metrics: MRR of the first relevant rank; recall@10 and recall@100 as counts of the 34 tasks;
  Search p95; raw windows written; paired MRR deltas by task with a 10,000 resample interval, seed
  0; mean top 100 id overlap.

## What I predict

| quantity | prediction |
| --- | --- |
| K0 raw windows | exactly 1,220 |
| K0 recall@10 | 34 of 34 |
| K0 MRR | 0.78 to 0.90 (C6 and C7 measured 0.8611 on the flattened input) |
| K0b minus K0, MRR | -0.02 to +0.02 |
| K1 raw windows | 1,290 to 1,350 |
| **K1 minus K0, MRR** | **-0.05 to +0.01** |
| K1 recall@10 | 33 or 34 |
| K1 recall@100 | 34 |
| K1 Search p95 over K0 | 0.9x to 1.2x |
| routes | all 34 prompts on `code` in every arm |

Reasoning. The task prompts carry no dates, so timestamps cannot help retrieval here. The prefix
puts identical tokens (`timestamp`, `role`, `assistant`, a date) into nearly every window. BM25
gives those tokens little weight, since they are in most documents; the dense Code4 embedding may
drift slightly toward boilerplate, and 8% more words mean about 8% more windows, so a relevant
passage can split across a boundary. I expect a small loss, or none. Following
[[i-over-predict-effect-magnitudes]], the band is narrow and near zero.

## Decision rule, fixed now

N is the noise floor, |K0b minus K0| on MRR.

1. **Apparatus failure.** If any check below fails, report and decide nothing.
2. **Non-inferior.** If ALL of these hold:
   - K1 minus K0 on MRR is at least -(0.02 + N), and K1 minus K0b is also;
   - K1 recall@10 is at least the smaller of K0's and K0b's, and the same for recall@100;
   - K1 Search p95 is at most 1.25 times the larger of K0's and K0b's;

   then C9 with `content_only_windows=False` may serve both tracks. The recommendation to the user
   becomes one build for the next Textual and Coding runs. It is still the user's decision
   ([[official-textual-coding-config-graph-on]]), and it rests on a retrieval screen, not on
   executable Task Solve, which this check does not run.
3. **Otherwise, inferior.** Do not flip the flag for the shared endpoint. The next candidate is a
   renderer that keeps timestamped windows for conversational Adds and content-only windows for
   coding trajectories, under its own pre-registration.

## What would falsify this

- Any quantity outside its band.
- In particular, K1 minus K0 below -0.05 means the prefix costs Code4 retrieval more than I think;
  above +0.01 means it helps, which I have no mechanism for.

## How it will be measured

- **Script:** `scripts/aml_c9_coding_window_check.py`, committed with this record:
  `collect --amb-root <corpus root> --arm K0|K0b|K1 --out <arm>.json.gz [--timestamped-windows]`,
  then `report --arms K0.json.gz K0b.json.gz K1.json.gz`.
- **Where:** VPS3, `/home/sentiment/reader-dates`, the same venv and code checkout as the
  reader-dates run, moved to this record's commit. The three arms run at the same time, each in its
  own database (`codingcheck_k0_20260924`, `codingcheck_k0b_20260924`, `codingcheck_k1_20260924`)
  with its own embedding cache and lock. The service runs at the served limits (Add 8, Search 3).
  Adds of one user are serialised by the tenant lock anyway, so each arm adds sequentially.
- **Corpus copy:** `corpus/manifest.json`, the 196 manifest sessions and `tasks/*/task.json` from
  the local Agent Memory Bench checkout, verified on VPS3 by the loader's manifest and per-session
  hashes.
- **Cost cap:** 5 USD of OpenRouter for the three arms' Add-time compiles (the C9 Coding ingest was
  estimated at 0.45 USD per pass on 2026-09-23). Voyage embedding is extra and small.

## Apparatus checks

1. Every session's event texts rejoin to the flattened session (enforced before collect).
2. K0 and K0b write exactly 1,220 raw windows, the historical screen's count.
3. `/version` reports `message-content-only-v1` for K0 and K0b and `timestamp-role-content-v1` for
   K1 (enforced).
4. No Add and no Search failure in any arm.
5. The three new tests are red by mutation (`tests/test_aml_c9_coding_window_check.py`, docstring
   names each mutation): event text rejoins, role and Unix milliseconds, one-based first rank.

## Confounds I can name now

- **This is not AML's own Coding payload.** The real trajectories are private; per-event messages
  with roles and timestamps are the closest public stand-in. If AML sends tool calls in extra
  fields rather than in `content`, the service drops those fields, and the real text differs.
- **34 queries.** One task moving across rank 10 is 3% of recall@10, and MRR moves in steps. The
  replicate arm is what makes a small delta readable.
- **Retrieval only.** The C6 decision also rested on executable Task Solve; a retrieval
  non-inferiority does not prove Task Solve non-inferiority.
- **Scoring is at session level**, as before, so a window split that moves the relevant passage
  within its session does not register.
- **Voyage embeds the corpus** as in every earlier C6, C7 and C9 screen, although the corpus
  `data-policy.json` lists only `openrouter` among its providers.

## What I already know

- `2026-09-24-aml-c9-reader-dates.md`: the Textual result this check gates.
- `docs/results/2026-09-20-aml-code4-exact-parity.md`: C6 MRR 0.8611, recall@10 34/34, on the
  flattened input.
- `2026-09-23-aml-c9-add-time-atomizer.md`: the C9 Coding ingest, about 51 minutes with one worker
  and 0.45 USD estimated.
