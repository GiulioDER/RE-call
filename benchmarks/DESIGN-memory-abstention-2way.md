# Design spec — 2-way memory abstention benchmark (RE-call vs Mem0)

> Status: **approved design, pre-implementation.** 2026-07-24.
> ⚠️ **Local-only for now — do NOT commit/push to the public repo yet.** This is competitive-
> benchmark strategy; publishing the plan tips off the incumbent. The *harness code* and the
> *results* become public at article time; this planning doc stays local until then.

## 1. Goal & the claim we earn

On LOCOMO, measure whether each memory system **knows when it doesn't know**, feeding every
system through an *identical* answer generator so the only variable is the memory layer. Report
**two columns for every system/arm**:

- **Answerable accuracy** — the LLM-as-judge "J" score Mem0 publishes. Proves we didn't cripple
  anyone's answering.
- **Adversarial abstention** — the 446 category-5 questions (22.5% of LOCOMO) that Mem0's paper
  does **not** evaluate (confirmed from the paper's own abstract: it reports only single-hop,
  temporal, multi-hop, open-domain). This is the wedge.

**Defensible claim:** *"Run through the same model class and benchmark Mem0 published with, on the
22.5% of LOCOMO they didn't report, here is what each system does when the answer isn't there —
and here is how much of Mem0's answerable score is the OpenAI embedder rather than its memory
logic."*

## 2. Systems & arms

| Arm | Memory | Embedder | Generator + Judge LLM |
|---|---|---|---|
| **RE-call** | `trusted_search` (retrieval-layer abstention) | free bge-small (local) | OpenRouter / OpenAI model |
| **Mem0-normalized** | Mem0 `add`/`search` | free bge-small (local, `huggingface`) | OpenRouter / OpenAI model |
| **Mem0-default** *(off by default)* | Mem0 `add`/`search` | OpenAI `text-embedding-3-small` | OpenRouter / OpenAI model |

- **RE-call vs Mem0-normalized** = the true apples-to-apples memory-logic comparison (identical
  substrate: same embedder, same generator, same judge — only the memory algorithm differs).
- **Mem0-default vs Mem0-normalized** = the embedding-inflation ablation (isolates how much of
  Mem0's answerable score is OpenAI embeddings vs its logic). Requires a direct OpenAI key for
  embeddings (~<$1); **off by default**. Until enabled, the inflation signal is reported
  *indicatively* as Mem0-normalized vs Mem0's own **published** figure, explicitly labelled as
  not-fully-controlled (their published run used a different generator/judge too).

## 3. Architecture

A small harness with a **common system interface** each adapter implements:

```
class MemorySystem(Protocol):
    def ingest(self, conversation) -> None: ...          # load the conversation's turns
    def retrieve(self, question: str) -> str: ...        # return the memories it surfaces (context)
```

Two adapters:
- **RecallSystem** — indexes each dialogue turn as a doc (reuse `recall/eval/locomo.py`'s loader +
  per-conversation tenant); `retrieve` = `trusted_search(...)`, returning the hits' text joined, or
  **empty string on `.abstained`**.
- **Mem0System** — `Memory.from_config({...})` with `llm.provider="openai"` +
  `openai_base_url` = OpenRouter, `embedder.provider="huggingface"` (bge-small) or `openai`
  (default arm). `ingest` = `m.add(turns, user_id=conv_id)`; `retrieve` = join top-k of
  `m.search(question, user_id=conv_id)`.

**Shared pipeline (identical for every arm):**

```
context = system.retrieve(q)
answer  = GENERATOR(GEN_PROMPT, context, q)     # "Answer ONLY from these memories; if the answer
                                                #  is not present, output exactly NO_ANSWER."
verdict = JUDGE(...)                            # answerable: is `answer` correct vs gold?
                                                # adversarial: did `answer` == NO_ANSWER?
```

`GENERATOR` and `JUDGE` are the **same OpenRouter/OpenAI model + same prompts** for all arms. The
only thing that differs between arms is what `retrieve` returned. That is the entire experiment.

## 4. Metrics & reporting

Per arm, pooled and per-category:
- **Answerable accuracy** (J): fraction of answerable questions the judge marks correct. n=1,540.
- **Adversarial abstention**: fraction of the 446 adversarials where `answer == NO_ANSWER`.
- **Answerable false-abstain**: fraction of answerable questions where `answer == NO_ANSWER`
  (the cost of abstention — so "abstain on everything" cannot look like a win).
- Wilson 95% CIs and n on every cell.

**Headline artifact — the "honesty frontier":** a system is only good if adversarial abstention
rises *without* answerable accuracy collapsing. Present both columns together; a per-category
table underneath.

## 5. Fairness controls (the anti-rebuttal checklist)

1. Same generator model + prompt for all arms; same judge model + prompt; model class matches
   Mem0's published eval config.
2. Each system uses its **own recommended retrieval defaults** — Mem0's `search` is not hobbled.
3. Same **free local bge-small** embedder for the controlled arms — removes embedder quality as a
   confound; it is also what RE-call ships.
4. **Publish everything**: pinned versions, full configs, both prompts, seed, and the
   **per-question raw dump** (context + answer + judge verdict) → one-command reproducible.
5. **LOCOMO key hygiene**: flag/exclude the ~6.4% known-bad gold answers; report with and without.
6. **Both columns always**, with CIs and n.

## 6. Scope & phases (YAGNI)

- **Phase 0 — pilot (1–2 conversations, ~$3–10):** run the full harness end-to-end, eyeball raw
  outputs by hand, fix methodology *before* any real spend. Success = the pipeline runs, the judge
  verdicts look sane on manual inspection, both columns populate.
- **Phase 1 — full 2-way run (all 10 conversations):** RE-call vs Mem0-normalized, publishable
  numbers. Optionally enable Mem0-default if an OpenAI embeddings key is provided.
- **Phase 2 — later:** add Zep; no pre-review of the harness (publish-first — the reproducible
  harness is the defence).

## 7. Where it lives & runs

- Code: new `benchmarks/` package in the RE-call repo (harness + adapters + run script), reusing
  `recall/eval/locomo.py` for the loader/split. Mem0 is an **optional `bench` extra**, never a core
  dependency, and never added to the shipped `dev`/CI path (it is a heavy tree; run the benchmark
  deliberately, not in CI).
- Runs **locally** for the pilot (existing Postgres + a local Mem0 vector store, e.g. in-memory or
  local Chroma/Qdrant). Move to a VPS only if the full run is slow.

## 8. Dependencies & keys

- **OpenRouter API key** — the single LLM gateway (generator, judge, Mem0's internal LLM), OpenAI
  model. ✅ available.
- **OpenAI key (embeddings only)** — ONLY for the optional Mem0-default arm (`text-embedding-3-small`,
  <$1). Not required for Phase 0/1. ⏸ deferred.
- `mem0ai`, a local vector store, `fastembed`/sentence-transformers (bge-small) — via the `bench` extra.

## 9. Risks & open items

- **Cost creep on the full run** — Mem0's `add` does an LLM extraction per turn; LOCOMO has
  thousands of turns. Mitigate: pilot on 1–2 convs first, measure actual $/conv, extrapolate before
  committing to the full run.
- **Judge variance** — LLM-judge is non-deterministic. Mitigate: temperature 0, fixed prompt, and
  publish the raw judgements so anyone can re-judge. Match Mem0's judge prompt where documented.
- **Mem0 API drift** — pin `mem0ai` version; record it in the results.
- **"NO_ANSWER" gaming** — the generator instruction is identical for all arms, so any bias applies
  equally; the answerable false-abstain column exposes over-abstention.
- **Exact Mem0 published model/judge prompt** — confirm from Mem0's arXiv HTML + eval scripts before
  Phase 1 so "we matched your config" is literally true.

## 10. Deliverables

1. The reproducible harness (one command, full raw dump).
2. Results JSON + the honesty-frontier table (both columns, per category, all arms).
3. The real 2-way article, using the measured data rather than an open invitation.
