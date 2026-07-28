# Show HN draft

> ⛔ **DEFERRED (2026-07-24)** — HN posting isn't available on the account right now. This draft is
> kept ready for when that changes; the launch is running via dev.to + r/LocalLLaMA + X + Discord
> instead (see `PLAN.md` §4). Nothing below has changed; it's just on hold.
>
> Draft launch copy. Grounded in measured numbers from `results/` and `FINDINGS.md`.
> Do NOT post until: (1) 0.5.2 is live on PyPI so `pip install recall-rag` works, and
> (2) the GitHub README fix is on master. Post ~08:00–10:00 ET on a Tue/Wed/Thu.

## Title (pick one — finding-first beats product-first on HN)

**Recommended:**
> Show HN: RE-call – AI memory that abstains instead of returning a confident wrong hit

Alternatives:
> Show HN: I benchmarked AI memory on refusal — systems answer every unanswerable question
>
> Show HN: The one memory metric Mem0 and Zep don't report — knowing when you don't know

Keep it under ~80 chars. The recommended one names the project (Show HN etiquette) *and* leads
with the differentiator.

## Body (the first text box)

Long-running agents accumulate memory — decisions, closed experiments, incident notes — and then
they re-litigate settled decisions and build on facts that are no longer true. The nasty part:
when you reverse a decision, the *stale* memory of it is often the **highest-cosine hit in the
whole result**. Plain vector search serves it, confidently.

RE-call is a retrieval engine for an agent's own memory that returns **verdict + confidence +
provenance** with every hit — not just similarity — demotes memories that were superseded or
expired, and prefers an explicit **"I don't know"** over confident noise.

Two things I measured that I haven't seen reported elsewhere:

1. **Supersession beats recency.** On adversarially-worded queries, plain search returns the stale
   memory 100% of the time (it's the top cosine). RE-call's superseded-trust rate is 0.00
   [Wilson 95%: 0.00, 0.02], n=250 — it flags the stale hit and points at its successor.

2. **The abstention axis nobody publishes.** LOCOMO (the standard agent-memory benchmark) ships
   446 adversarial questions — 22.5% of the set — that *look* answerable but aren't. Every
   published LOCOMO result I could find scores only the answerable categories. Out of the box
   RE-call abstains on **zero** of them too (they're on-topic, wrong-attribution — high cosine).
   Its shipped levers (threshold calibration, an entailment judge) raise that to 0.37–0.77, but
   only by refusing a quarter to half of *legitimate* questions. I'm publishing that whole
   trade-off curve, including where it's bad, because a claims table without the failures is
   marketing.

Deliberately boring under the hood: **no LLM and no graph DB in the retrieval path** — pgvector +
Postgres full-text over a table you already know how to back up. That's the constraint that makes
it auditable, and also why there's no entity reasoning here.

It's MIT, `pip install recall-rag`, 2-minute quickstart with a free local embedder (no API key).
Retrieval hit@5 on LOCOMO is 0.615 [0.59, 0.64] with that free embedder — a substrate number, not
a leaderboard win against systems that ship a generator I don't.

Repo: https://github.com/GiulioDER/RE-call
Would love feedback on the abstention curve specifically — is refusing 25–55% of real questions to
catch 40–75% of the wrong-attribution ones a trade you'd take for an agent that acts on its memory?

## Your first comment (post immediately after, pre-empt the top objection)

The obvious pushback: "0.00 abstention out of the box sounds broken." It's the honest starting
point, and it's the same finding from two directions — a weak local embedder can't separate an
on-topic wrong-attribution question from a real one by cosine, because they *are* cosine-close.
The point of the repo isn't a magic number; it's (a) measuring the axis at all, and (b) showing
exactly what each lever buys and costs. Calibration here is fit in-sample — the upper bound, not a
deployed operating point — and I say so in the harness. Happy to go deep on methodology.

## Notes for the poster
- Post under your own handle. Be present in the comments for the first 3–4 hours — founder
  responsiveness is the single biggest driver of a Show HN thread staying alive.
- Do NOT ask for upvotes anywhere (HN auto-flags it). Just ship the link to a few people who'd
  genuinely find it interesting and let it ride.
- If it stalls, that's fine — the blog post (below) is the durable asset; HN is one shot at a spike.
