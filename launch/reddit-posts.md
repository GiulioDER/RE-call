# Reddit posts

> Two subreddits, two registers. Post the dev.to link (or the repo) as the target. Lead with the
> FINDING, never "check out my project". Post from your own account; read each sub's self-promo
> rules first (both allow substantive OSS/research posts; neither tolerates pure marketing).
> Space the two posts out by a day or two — don't blast both at once.

---

## r/LocalLLaMA  (self-post — this is the primary spike channel now that HN is out)

**Title:**
> Every AI-memory benchmark (Mem0, Zep) skips the 22.5% of LOCOMO that tests whether the memory knows when it *doesn't* know. I measured it.

**Body:**

Mem0 and Zep publish LOCOMO accuracy numbers (92% / ~75%) and are in a public methodology fight about them. Both compute those numbers on the *answerable* questions.

LOCOMO also ships 446 adversarial questions (22.5% of the set): they name a real person + topic from the conversation, then ask about something that person never said. The right answer is "not in here." I couldn't find a published result that scores them — the category gets dropped (the original harness had a broken formatter for it, and ~6.4% of LOCOMO's answer keys are wrong anyway).

That's the part that actually breaks agents: a memory that returns a confident wrong recall instead of abstaining. So I scored that axis for my open-source retriever (RE-call) and I'm posting the whole trade-off, including where it fails:

```
mode              catches adversarials   refuses REAL questions
default           0.00                   0.00
calibrated*       0.53                   0.37
entailment judge  0.37                   0.26
both              0.76                   0.56
(* in-sample = best case, not a deployed operating point)
```

Out of the box it abstains on zero of them too — a weak local embedder can't separate an on-topic wrong-attribution question from a real one by cosine, because they're cosine-close. The shipped levers help but pay for it by refusing 26–56% of legitimate questions.

Retrieval hit@5 on LOCOMO is 0.615 [0.59, 0.64] with a free local embedder — a substrate number, deliberately not put next to Mem0/Zep's LLM-judge scores over a generator I don't ship.

Stack: MIT, Postgres + pgvector, no LLM/graph in the retrieval path. Drop-in LangChain + LlamaIndex retrievers that return nothing when the trust layer abstains. Repo has the full methodology and a running list of the project's *own* withdrawn claims.

Repo: https://github.com/GiulioDER/RE-call
Write-up: [dev.to link]

Curious what people here think about the trade-off: is refusing 25–55% of real questions to catch 40–75% of the wrong-attribution ones a deal you'd take for an agent that acts on its memory?

---

## r/MachineLearning  (flair: [P] Project — more formal, methodology-first, zero hype)

**Title:**
> [P] Scoring the abstention axis of LOCOMO that agent-memory leaderboards drop (446 adversarial questions, 22.5%)

**Body:**

Agent-memory systems (Mem0, Zep) report LOCOMO accuracy on the answerable categories. LOCOMO's category-5 set — 446 adversarial, wrong-attribution questions whose correct response is abstention — is, as far as I can find, unscored in published results (broken category-5 formatter in the original harness; independent audits also put ~6.4% of the answer keys as incorrect).

I built an open-source retriever (RE-call) with an explicit abstention decision (calibrated cosine-gap threshold + optional QNLI entailment judge over hits) and evaluated it on that split. Reporting both sides, since a system that abstains on everything trivially scores 1.0 on adversarials:

- adversarial abstention (want high) vs answerable false-abstain (want low)
- default 0.00 / 0.00 · calibrated-in-sample 0.53 / 0.37 · entailment 0.37 / 0.26 · both 0.76 / 0.56
- retrieval hit@5 (evidence-turn id match) 0.615 [0.59, 0.64], n=1536

Calibration is fit in-sample deliberately (an upper bound): the adversarials are on-topic, so the answerable and adversarial top-cosine distributions overlap and a threshold alone can't separate them. Deliberately no LLM-judge (J) score — the library ships no generator, so a J number would grade a model it doesn't include and wouldn't be comparable to Mem0/Zep's.

Harness, per-category tables, and negative results are in the repo. Feedback on the evaluation design (particularly the in-sample calibration as an upper bound, and the entailment-judge stage) welcome.

Repo: https://github.com/GiulioDER/RE-call
