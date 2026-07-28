> ⛔ SUPERSEDED 2026-07-25 — DO NOT PUBLISH AS-IS. This draft says "I haven't run Mem0/Zep" /
> "open invitation", which is now FALSE: the full head-to-head was run (Mem0, full n, both
> OpenAI generators, both judges, metered cost). Use benchmarks/ARTICLE_DRAFT.md instead, or
> strip every "haven't run it" / "open invitation" line before this goes anywhere.

---
title: "The one question every AI-memory benchmark skips: does it know when it doesn't know?"
published: false
description: "Mem0 and Zep fight over LOCOMO accuracy. Both skip the 22.5% of the benchmark that tests whether a memory refuses when the answer isn't there. I measured that axis — including where my own system fails it."
tags: ai, llm, python, opensource
canonical_url:
cover_image:
---

<!--
REVIEW BEFORE PUBLISHING.
- published: false  -> imports as a DRAFT on dev.to. Flip to true (or hit Publish) when ready.
- canonical_url: leave EMPTY -> dev.to becomes the canonical home (you have no blog site yet).
  If you later stand up a real site, set canonical there and point this at it.
- cover_image: optional. The `docs/superseded-catch.png` in the repo works if you want one
  (upload it to dev.to and paste the URL), or leave blank.
- Every number below is measured and lives in the repo's results/ + FINDINGS.md. Nothing here
  claims a head-to-head I didn't run.
-->

There's a small benchmark war going on in AI memory. Mem0 published a paper putting its LOCOMO accuracy at 92.5% and LongMemEval at 94.4%, and reported a competitor, Zep, at ~66%. Zep published a rebuttal — *"Is Mem0 Really SOTA in Agent Memory?"* — self-reporting a corrected ~75% and disputing the methodology. It's a genuinely useful fight; it's how you know the numbers matter.

But both sides are fighting over the same axis: **of the questions the memory *can* answer, how many does it get right?**

LOCOMO ships a fifth category almost nobody reports on: **446 adversarial questions — 22.5% of the whole benchmark** — that look answerable but aren't. They name a real person and a real topic from the conversation, then ask about something that person never said (wrong-attribution). The correct answer is *"that's not in here."*

I went looking for a published LOCOMO result that scores those 446 questions. I couldn't find one. The accuracy leaderboards are computed on the answerable categories; the adversarial set gets dropped. (There's a known reason for the neglect: the original harness had a broken formatter for that category, and independent auditors have since found ~6.4% of LOCOMO's answer keys are wrong.)

So a quarter of the benchmark measures the thing that actually breaks agents in production — a confident wrong recall — and the leaderboards skip it.

## Why this is the thing that matters

An agent that acts on its memory doesn't fail politely. When it retrieves a fact that isn't really there, it doesn't return an error — it returns a confident, plausible, wrong answer, and then acts on it. For a memory layer, **a confident wrong recall is worse than "I don't know."** The whole value of memory is that the agent can trust it.

And the trap is geometric. When you reverse a decision, the *stale* memory of the old decision is often the highest-cosine hit in the entire result set. Similarity search returns it first, with maximum confidence. I measured this: on adversarially-worded queries, plain vector search returns the stale memory **100% of the time**.

## What I built, and what it actually scores

[RE-call](https://github.com/GiulioDER/RE-call) is a retrieval engine for an agent's own memory that returns a **verdict + confidence + provenance** with every hit instead of just a similarity score, and prefers an explicit abstention over a confident guess. No LLM and no graph database in the retrieval path — just pgvector and Postgres full-text, which is what makes it auditable.

Here's the honest scorecard on LOCOMO, with the free local embedder (no API key):

**Retrieval substrate** — evidence-turn hit@5 = **0.615** [0.59, 0.64], n=1536 answerable questions. This is *not* comparable to Mem0's/Zep's 66–92% "J" scores — those grade an LLM-as-a-judge over a *generator* RE-call doesn't ship. This number measures the retrieval layer underneath such a system. I'm careful about that boundary because a number placed next to an incomparable one is a lie of context.

**The abstention axis** — the 446 adversarials. And here's where I'll do something the leaderboards don't: show the whole trade-off, including where it's bad.

| Mode | Catches adversarials (want high) | Refuses real questions (want low) |
|---|---|---|
| Default (no calibration, no judge) | **0.00** | 0.00 |
| Calibrated threshold* | 0.53 | 0.37 |
| Entailment judge | 0.37 | 0.26 |
| Both | **0.76** | 0.56 |

\* Calibration here is fit *in-sample* — it sees the exact distribution it's later scored on. That's the **upper bound**, not a deployed operating point. If it can't cleanly separate the two distributions even with that advantage, no honestly-fit threshold can — which is the point: the adversarials are on-topic, so they sit cosine-close to real questions and a threshold alone can't part them.

Read that table straight: **out of the box, RE-call abstains on zero of the adversarials too.** A weak embedder can't tell an on-topic wrong-attribution question from a real one, because by cosine they *are* the same. The levers the library ships raise adversarial-catch to 0.37–0.76, but they pay for it by refusing 26–56% of legitimate questions. The residual gap is exactly the entity reasoning I deliberately left out (no graph, no LLM).

## Why publish the failure?

Because the failure *is* the result. The field's leaderboards report a single flattering accuracy number and drop the category that's hard. I'd rather show a curve that says "here's what refusal costs, and here's where it's still not good enough" than a 9X% headline that quietly measures only the easy half. (The repo takes this further — it keeps a running list of its *own* claims that didn't survive re-measurement, and withdraws them in public.)

If you're building an agent that acts on its own memory, the question isn't "what's your accuracy?" It's "what does your memory do when the answer isn't there?" Right now, as far as I can measure, the honest answer for most systems — including mine, by default — is *"it answers anyway."* At least now there's a number on it.

## Try it

MIT, Postgres + pgvector, 2-minute quickstart with a free local embedder:

```bash
pip install "recall-rag[fastembed]"
```

It drops into the frameworks you're probably already using — there's a **LangChain** retriever and a **LlamaIndex** retriever that both return *nothing* when the trust layer abstains, rather than handing your chain a stale memory:

```python
from recall.integrations.langchain import RecallRetriever
retriever = RecallRetriever.from_store(store, embedder, k=5)
docs = retriever.invoke("how many requests per second can a client make?")
```

Repo (methodology + every negative result): **https://github.com/GiulioDER/RE-call**

---

### An open invitation

I've scored RE-call on the adversarial set. The obvious next experiment is the head-to-head: run Mem0 and Zep through the *same* 446 questions and publish the abstention column for all three. I haven't run their systems yet (it needs their extraction-LLM keys and a fair harness), so I'm not going to quote numbers I didn't measure. If you maintain one of these systems, or want to help build a neutral harness, the LOCOMO adversarial split is right here and I'll publish whatever it says — including if RE-call loses.
