> ⛔ SUPERSEDED 2026-07-25 — DO NOT PUBLISH AS-IS. This draft says "I haven't run Mem0/Zep" /
> "open invitation", which is now FALSE: the full head-to-head was run (Mem0, full n, both
> OpenAI generators, both judges, metered cost). Use benchmarks/ARTICLE_DRAFT.md instead, or
> strip every "haven't run it" / "open invitation" line before this goes anywhere.

# X / Twitter thread draft

> Post after the dev.to post is live; the last tweet links to it. Keep each ≤280 chars.
> Post from your own account. Don't buy engagement. Reply to anyone who engages in the first
> couple hours. The 4th tweet (the table) is the one people screenshot.

**1/**
Mem0 and Zep are in a public benchmark war over AI-memory accuracy on LOCOMO (92% vs ~75%).

Both of them skip the same 22.5% of the benchmark.

It's the part that actually breaks agents. 🧵

**2/**
LOCOMO ships 446 "adversarial" questions — they name a real person + topic from the chat, then ask about something that person never said.

Correct answer: "that's not in here."

I couldn't find a single published LOCOMO result that scores them. They get dropped.

**3/**
Why it matters: an agent that acts on memory doesn't fail politely. Retrieve a fact that isn't there → it returns a confident, plausible, WRONG answer and acts on it.

For a memory layer, a confident wrong recall is worse than "I don't know."

**4/**
So I measured that axis for RE-call. Honestly — including where it's bad:

Mode            | catches adversarials | refuses REAL questions
default         | 0.00                 | 0.00
calibrated*     | 0.53                 | 0.37
entailment judge| 0.37                 | 0.26
both            | 0.76                 | 0.56

*in-sample = best case

**5/**
Read it straight: out of the box RE-call abstains on ZERO of them too.

A weak embedder can't tell an on-topic wrong-attribution question from a real one — by cosine they're the same. The levers help, but only by refusing 26–56% of legit questions.

That's the honest state of the art.

**6/**
I'm publishing the failure because the failure IS the result. The leaderboards report one flattering accuracy number and drop the hard category.

The question for agent memory isn't "what's your accuracy?" — it's "what does it do when the answer isn't there?"

**7/**
RE-call is MIT, Postgres + pgvector, no LLM/graph in the retrieval path. Drop-in LangChain + LlamaIndex retrievers that return nothing when they should, instead of a stale memory.

Full method + every negative result:
https://github.com/GiulioDER/RE-call
[dev.to post link]

**8/ (open invitation)**
Next: run Mem0 + Zep through the same 446 questions and publish the abstention column for all three. Haven't yet — needs their LLM keys + a fair harness.

If you maintain one of these, the split is right here. I'll publish whatever it says, incl. if RE-call loses.
