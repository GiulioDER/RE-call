# Bench'd (benchd.ai): official rules, and the plan for an official RE-call run

Everything in this document was read from the sources on 2026-08-23, not remembered:
benchd.ai/benchmarks, benchd.ai/docs, benchd.ai/methodology, and a clone of
[github.com/benchdai/harness](https://github.com/benchdai/harness), which is the code that
actually produced their leaderboard's signed manifests (the clone ships them under `runs-all/`).
Where the website and the code disagree, this document says so and sides with the code, because
the signed manifests agree with the code.

**Status: nothing has been run.** This is preparation. The RE-call configuration is to be
discussed before the smoke test, and a pre-registration must be written and committed before the
full measured run (`/preregister`; the guard denies measurement commands while anything under
`docs/preregistrations/` is uncommitted).

## 1. What Bench'd is

A leaderboard for AI **memory systems** (not RAG over document corpora): LongMemEval, LoCoMo, and
a built-in smoke set, run through one neutral open-source harness, with Ed25519-signed result
manifests. Current published LongMemEval column (May 2026, from their README): LlamaIndex 59.0,
LangChain 59.0, LLM Baseline 57.6, Mem0 OSS 32.4. This is the direct competitor comparison the
run is for.

## 2. The official pipeline, from the code

Per benchmark item, `benchd_harness/runner.py` does exactly:

1. `adapter.reset()`
2. `adapter.ingest(turns)` (the item's whole conversation history, one call)
3. `adapter.recall(query)` (returns one plain string; latency is measured over ingest + recall)
4. answerer LLM turns that string into an answer
5. judge LLM scores the answer CORRECT or INCORRECT
6. trace recorded; manifest built, signed, saved as `runs/<run_id>/manifest.signed.json`

Consequences that decide our configuration:

- **The recall string is the only channel.** Whatever RE-call wants the answerer to see
  (memory text, session dates for temporal questions) must be serialized into that one string.
  The harness's own generic MCP adapter drops timestamps entirely; a custom adapter may use
  everything in the `turns` dicts, which every adapter receives equally.
- **Reset plus full re-ingest happens per question.** LoCoMo has ~1,540 questions over only 10
  conversations, so a naive adapter re-embeds the same ~30k-token conversation ~154 times each.
  Their own adapters pay the same cost. An ingest cache keyed on the hash of the turns is
  behaviour-preserving for RE-call (deterministic ingestion, identical index state) and is the
  difference between hours and days on LoCoMo. Flagged as a fairness call to make out loud.
- **Abstention is scored as wrong.** Judge prompt, verbatim: "If the Given Answer says
  "insufficient information" or equivalent, judge it as INCORRECT." An empty recall string leads
  the answerer to say exactly that. RE-call's abstention can only lose points here (LoCoMo
  category 5, the adversarial one, is excluded by their loader).
- **Efficiency is 30% of the headline BMI** and is computed from tokens per correct answer,
  where tokens means the recall string's estimated tokens (`len(text.split()) * 4 // 3`).
  Their top systems return ~25 to 30 tokens per recall. A 1,500-token recall string would crater
  BMI even at high accuracy: `Efficiency = 100 - min(tokens_per_correct / 100, 100)`,
  `BMI = 0.70 * Accuracy + 0.30 * Efficiency`. Top-k and formatting are the levers.
- **With `--judge` every question is judged by the LLM** and the leaderboard number is the
  `scores.nuance.overall` percentage. The deterministic "verified" track is only populated in
  no-judge runs (verified nulls in every judged manifest they ship).

## 3. Official models and prompts

| Component | Value | Where |
|---|---|---|
| Answerer | `openai/gpt-4o-mini` via OpenRouter | `LLMJudgeConfig`, `scoring/llm_judge.py`; confirmed in their signed manifests |
| Judge | `openai/gpt-4o-mini` via OpenRouter | same |
| Temperature | 0.0 | same |
| Max answer tokens | 256 | same |
| Max judge tokens | 128 | same |
| API base | `https://openrouter.ai/api/v1`, key from `OPENROUTER_API_KEY` (falls back to `OPENAI_API_KEY`) | same |

⚠️ **The methodology page contradicts this.** benchd.ai/methodology (fetched 2026-08-23) claims
the locked judge is `claude-sonnet-4-20250514`, prompt v2.4.1; their PROTOCOL.md and their actual
signed leaderboard manifests both say `openai/gpt-4o-mini`, prompt unversioned in code. For an
official run we use the harness defaults, which is what their leaderboard rows were produced
with. Re-measure: `python - <<'EOF'` over any `runs-all/*/manifest.signed.json` in their repo,
read `manifest.judge`.

Answerer prompt, verbatim from `scoring/llm_judge.py` (do not restyle, it is evidence):

```
You are answering a question using ONLY the retrieved memories below.

Retrieved memories:
{retrieved_memories}

Question: {question}

Instructions:
- Answer the question using ONLY information from the retrieved memories above.
- If the retrieved memories do not contain enough information to answer, say "Insufficient information in memory."
- Be concise and direct. Give the specific answer, not an explanation.
- Do not make up information not present in the retrieved memories.
```

Judge prompt, verbatim from the same file:

```
You are a judge evaluating whether an answer is correct.

Question: {question}
Expected Answer: {expected_answer}
Given Answer: {given_answer}

Instructions:
- Determine if the Given Answer is semantically equivalent to the Expected Answer.
- The Given Answer does not need to match word-for-word. It needs to contain the correct information.
- If the Given Answer contains the correct fact even with extra context, judge it as CORRECT.
- If the Given Answer says "insufficient information" or equivalent, judge it as INCORRECT.
- If the Given Answer is partially correct but missing key information, judge it as INCORRECT.

Respond with EXACTLY one line in this format:
JUDGMENT: CORRECT
or
JUDGMENT: INCORRECT

Then on the next line, briefly explain your reasoning in one sentence.
```

## 4. Datasets, from the loaders

| Slug | Source | Size | Scoring split |
|---|---|---|---|
| `longmemeval-v1` | HuggingFace `xiaowu0162/longmemeval-cleaned`, `longmemeval_oracle.json`, auto-downloaded to `~/.cache/benchd/longmemeval/` | 500 questions | recall and temporal dimensions exact-match, reasoning LLM (all LLM under `--judge`) |
| `locomo-v1` | GitHub `snap-research/locomo`, `data/locomo10.json`, auto-downloaded to `~/.cache/benchd/locomo/` | ~1,540 QA (category 5 adversarial excluded) | single_hop and temporal exact, multi_hop and open_domain LLM (all LLM under `--judge`) |
| `smoke-memory-v0` | built into the harness | 10 questions | mixed |

Notes that matter:

- LongMemEval is the **oracle** variant: only the evidence sessions, mean ~5,700 estimated
  ingest tokens per item. Not the 115k-token haystack variant.
- LongMemEval ingest turns carry `metadata.has_answer` (an oracle evidence flag). **An adapter
  must not read it.** Ours refuses to.
- LoCoMo turns carry `metadata.session_date_time` and `metadata.speaker`; LongMemEval turns
  carry no dates (only `session_index`). Temporal questions on LoCoMo are winnable only if the
  adapter surfaces those dates in the recall string.
- `--max-items N` takes a seeded (42) stratified sample by question type.
- ⚠️ **Their leaderboard LoCoMo rows are n=49 samples**, not the full 1,540: both LoCoMo
  manifests in their repo (`llamaindex-memory` 54.8, `llm-baseline` 50.4) have 49 traces.
  Their LongMemEval rows are the full 500. For a comparable LoCoMo cell we either run the same
  `--max-items` sample (cheap, weak n) or the full set (stronger, and it subsumes their cell).

## 5. Submission and trust tiers

From PROTOCOL.md and the CLI: `benchd keys generate`, run with `--key`, then
`benchd submit ./runs/run_xxx/manifest.signed.json` (POSTs to benchd.ai/api/submit; web upload at
benchd.ai/submit as fallback). A run signed with our key on our infrastructure lands as
**Self-Reported**. Community-Verified requires Bench'd to run it themselves; Vendor-Verified
requires their co-signature against our endpoint. For visibility the practical path is: submit a
self-reported run with full traces, then invite them to reproduce it (the custom adapter makes
that a one-command rerun for them, and there is a GitHub issue template for new systems in their
`scripts/`).

Registering RE-call as a system means a ~50-line adapter in the harness fork, registered in
`benchd_harness/adapters/__init__.py`. Draft lives here: `recall_adapter.py`. Their
`benchd adapter validate` checks the interface.

## 6. Cost and time model (estimates, to be replaced by smoke-test measurements)

LLM spend is small because gpt-4o-mini is cheap and their pipeline is two short calls per
question. Computed from their own LlamaIndex LongMemEval manifest (real trace text, their token
estimator, list prices $0.15/M in, $0.60/M out):

| Run | LLM calls | Estimated spend |
|---|---|---|
| Their LongMemEval 500 (30-token recalls) | 1,000 | ~$0.04 |
| RE-call LongMemEval 500, if ~1,000-token recalls | 1,000 | ~$0.15 |
| RE-call LoCoMo full 1,540, if ~1,000-token recalls | 3,080 | ~$0.45 |

The real budget is **wall time and embedding**, both on the ingest side, both zero-LLM:

- LongMemEval 500: ~2.9M ingest tokens total. Fine anywhere.
- LoCoMo full: ~30,600 estimated ingest tokens per item times 1,540 items is ~47M tokens
  embedded naively; the ingest cache reduces it to 10 conversations (~310k tokens) plus
  1,530 cache hits.
- Embedding location is governed by the standing rule: **embedding runs on VPS2, one process,
  bounded** (flock plus the systemd-run caps), or a hosted embedder so no local model runs at
  all. The workstation is not an option for the full run.

The token counter for the smoke test is `count_tokens.py` here. It recomputes answerer and judge
tokens exactly from a run's manifest (tiktoken when installed, their word estimator otherwise),
prints tokens per question, and projects full-run cost; it can also snapshot the OpenRouter key's
real usage counter before and after a run, which is the measured number the artifact should carry.

## 7. Configuration decisions to make before the smoke test

Each is an env knob on the draft adapter, so the discussion maps one-to-one onto a config:

1. **Abstention** (`RECALL_BENCHD_ABSTAIN`, honour or suppress). Honouring it is RE-call's real
   behaviour and costs points by construction (see section 2). The mem0-harness run honoured it
   on principle. Same call to make here, out loud.
2. **Top-k and recall formatting** (`RECALL_BENCHD_TOP_K`). Accuracy wants more context, BMI
   efficiency wants ~30 tokens. Their top systems sit at 25 to 30 tokens per recall.
3. **Ingest granularity** (`RECALL_BENCHD_GRANULARITY`, turn or session). Turn granularity
   guarantees every retrieved chunk carries its session date; session granularity gives the
   retriever more context per chunk but a chunk split can strand the date header.
4. **Embedder** (`RECALL_BENCHD_EMBEDDER`). fastembed bge-small is $0 and matches every
   published RE-call number; voyage-4 is the hosted route that sidesteps the VPS2 local-model
   caps entirely.
5. **Where it runs.** VPS2 under the embedding caps (enterprise-rag-run precedent, own Postgres
   in `~/enterprise-rag-run/pgdata`), or workstation with a hosted embedder, or a session
   container for the smoke test only.
6. **LoCoMo cell**: full 1,540, or their n=49 sample, or both.
7. **Ingest cache** (`RECALL_BENCHD_INGEST_CACHE`): on for LoCoMo or strict per-item re-ingest.

## 8. Smoke test procedure (after the config discussion)

```bash
git clone https://github.com/benchdai/harness benchd-harness && cd benchd-harness
pip install -e .
cp /path/to/recall/benchmarks/benchd/recall_adapter.py benchd_harness/adapters/
# register: in benchd_harness/adapters/__init__.py add
#   from benchd_harness.adapters.recall_adapter import RecallAdapter
#   and the "re-call" entry in _BUILTIN_ADAPTERS
benchd keys generate --out ./keys
export OPENROUTER_API_KEY=sk-or-...
python /path/to/recall/benchmarks/benchd/count_tokens.py openrouter   # usage snapshot before
benchd run -a re-call -b smoke-memory-v0 --judge --key ./keys/private.key
python /path/to/recall/benchmarks/benchd/count_tokens.py openrouter   # usage snapshot after
python /path/to/recall/benchmarks/benchd/count_tokens.py manifest ./runs/run_*/manifest.signed.json --project 500
```

Operational lessons from the first smoke runs (2026-08-23, all fixed in the adapter or noted):

- **One database per embedder dimension.** The default `chunks` table is dimension-typed at
  first `ensure_schema`; switching hashing (64) to voyage-4 (1024) in one database fails with
  "table 'chunks' uses vector(64)". Wipe the schema between embedder switches, or use a fresh
  session container per configuration.
- The adapter must bootstrap the default `chunks` table before its bench table: global
  generation migrations refuse to apply through a custom table first.
- The synthesis digest must be complete declarative sentences. A bare "Marcus Chen." digest
  made the locked answerer answer "Insufficient information in memory.", which the judge
  scores INCORRECT. The prompt now forbids fragments; smoke went 10/10 with mean recall 45
  tokens per question.
- Arms share one tenant and table, so tuning runs are strictly sequential per database.

Then a 20-question dress rehearsal on the real target
(`benchd run -a re-call -b longmemeval-v1 -n 20 --judge ...`) to check dataset download,
per-dimension behaviour, and the token projection, before pre-registering and paying for the
full 500.
