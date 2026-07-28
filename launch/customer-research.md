# RE-call — Customer / Market Research

> **Status:** internal GTM working doc. Keep OUT of the public repo (`launch/` is untracked but
> NOT gitignored — add `launch/` to `.gitignore` or keep adding by path). Contains competitor
> teardown.
> **Date:** 2026-07-24
> **Companions:** `PLAN.md` (Mem0 GTM playbook → cold-start sequencing) ·
> `blog-the-question-benchmarks-skip.md` (credibility artifact) · `show-hn.md`.

## Framing

RE-call's wedge is **calibrated trust**: return a hit *with a verdict*, or honestly abstain
("I don't know"), instead of confidently feeding an agent wrong context.

**Honesty anchor (shapes all positioning):** RE-call's own benchmarks show abstention has a
**bounded domain** — strong on far-off-topic gaps, weak on near-miss wrong-attribution
(LongMemEval: refused ~48% of questions it had just answered correctly; six signals measured,
all fail to separate near-miss). So the sellable promise is *"a memory layer honest about what it
knows,"* NOT *"we eliminated hallucinated memory."* That distinction IS the brand — and it is the
one axis the field's benchmark-credibility war leaves wide open.

**Evidence base:** two evidence-first research passes over GitHub issues, Hacker News, the Letta
forum, and one academic benchmark (HaluMem, arXiv 2511.03506). Reddit was a coverage gap in both
passes — a logged-in Reddit pass would strengthen segments + trust signal. Vendor benchmark numbers
(LOCOMO/LongMemEval) are DISPUTED between Mem0 and Zep — do not cite any single score as settled.

---

## 1. Who the potential customers are

Five segments, ranked by public loudness × fit for RE-call:

| Segment | Who | Fit |
|---|---|---|
| **Indie / solo devs & seed startups** on LangGraph/OpenAI SDK | Loudest voice in every issue tracker; want drop-in memory, hate cloud lock-in | **Best** — `pip install recall-rag`, self-host pgvector, local embeddings, LangChain/LlamaIndex shipped |
| **Coding-agent / IDE-memory builders** (Claude Code plugins) | Distinct vocal sub-community (Cognee, claude-mem, Recallium chase it) | **Strong** — MCP server + `USING_WITH_CLAUDE.md` already exist; underexploited |
| **Customer-support / CRM agents** | Recall past tickets; must NOT confidently apply a stale resolution | **Strong** — wrong memory = wrong customer answer, abstention has $ value |
| **Multi-tenant SaaS platform teams** | Memory into a product, per-user isolation | **Partial** — fits trust/self-host; multi-tenant isolation is a gap (§5) |
| **Regulated / enterprise (finance, healthcare, GDPR)** | Talked *about* by vendors more than they speak | **Aspirational** — evidence thin, vendor-authored. Hypothesis, not a launch target |

**Aim the launch at the first three.** They self-serve, live on HN/GitHub (free reach), and feel
the trust pain directly.

---

## 2. What they WANT (the desires)

- **Memory they can trust in production** — strongest under-served signal in the dataset.
  Production audit: *"What we found after auditing 10,134 mem0 entries: 97.8% were junk"* —
  *"Any hallucination that gets stored once will be re-extracted indefinitely,"*
  *"indiscriminate memory storage performs worse than using no memory at all."*
  (github.com/mem0ai/mem0/issues/4573)
- **Escape from mandatory OpenAI** — local/offline embeddings + local LLM. Universal.
- **No LLM in the read/write path** — *"every memory solution (Mem0, Zep, Letta) routes data
  through an LLM on every operation… 200-500ms latency, token costs, a runtime dependency you don't
  control."* (HN 47260077). **RE-call already on the right side** — cosine + optional entailment
  judge (off by default), no LLM in CRUD.
- **See WHY a memory was returned** — inspectability/observability. Real but not yet a crisp
  request chorus (an opening).

RE-call satisfies 3 of 4 out of the box (trust, local embeddings, no-LLM path). That's the pitch.

---

## 3. What they EXPECT as table stakes (fail these = out)

1. Self-hosting / data ownership / air-gapped — *the* deciding factor. ✅ pgvector, local
2. Local embeddings, non-OpenAI models — ✅ fastembed/bge-small
3. Low latency, memory never blocks response — ✅ (no LLM in path)
4. Transparent, controllable cost — ✅ (no per-op token cost) — direct counter to pricing pain
5. Framework integration & trivial install — ✅ LangChain + LlamaIndex + MCP; `pip install`

**RE-call clears table stakes. The gap is ~1 GitHub star — a distribution problem, not a
capability problem.**

---

## 4. What they DISLIKE about competitors → the opening

Rule: a lost competitor customer is a potential one for us. Each grievance + whether RE-call answers it.

**Mem0**
- **Pricing cliff $19→$249/mo, graph memory paywalled** — most-cited production complaint →
  RE-call OSS, no cliff. **Wins.**
- **"Semantic drift"; confidently repeats stale facts** — *"a user contradicts something stored
  three sessions ago and the agent confidently repeats the stale version."* (particula) →
  **The exact failure abstention exists to flag. Headline use-case.**
- **OSS-neglect fear** — users publicly ask founders *"will you support the OSS version as a
  first-class citizen long term?"* → RE-call is OSS-first with a "withdrawn claims" ethos. **Wins.**

**Zep**
- **Community Edition deprecated → self-hosting is a real operational burden** (run Graphiti +
  Neo4j yourself). Strongest Zep complaint → RE-call self-host = `pip install` + Postgres. **Wins.**
- **Background ingestion delay — retrieval misses right after write** → RE-call indexes
  synchronously. **Wins.**
- **Per-episode LLM cost at scale** → RE-call: no LLM at ingest. **Wins.**

**Cross-cutting — biggest single opening**
- **Neither leader tells you when a memory is untrustworthy** — *"Both systems require custom work
  to implement human review workflows, which neither natively supports."* (fountaincity)
- **Mem0↔Zep benchmark war** (Zep claimed 84% LOCOMO → Mem0 recomputed 58% → Zep counter-claimed
  75%) left buyers unable to trust anyone's numbers. → **An independent, honest, abstention-aware
  benchmark that publishes its own failure modes is un-attackable** (can't be caught inflating a
  number you deliberately reported as bad). Ship the LOCOMO-adversarial credibility artifact.
- **Academic backing:** HaluMem (arXiv 2511.03506) tests Mem0/Zep → *all systems <70% accuracy,
  can't recognize their own "memory boundary."* External validation the wedge is a named, unsolved
  problem.

**Honesty caveat:** wrong-retrieval evidence is far stronger for **Mem0** than Zep. Don't overclaim
Zep's trust failures — attack Zep on self-host deprecation + ingestion delay instead.

---

## 5. What NEW FEATURES they're asking for

Ranked by cross-project repetition (real filed issues):

1. **Forgetting / decay / consolidation of stale memory** — single most cross-cutting request,
   independently filed in THREE competitor repos (letta #3116, graphiti #1300, cognee #996).
   *"redundant passages, semantic duplicates… so agents run indefinitely without degradation."*
   **Pairs naturally with RE-call's trust framing** — a trust layer that also demotes/expires stale
   evidence is a coherent story no one else tells.
2. **Multi-tenant / per-user / per-agent isolation** — repeated; mem0 has a real bug where "Agent
   A's memories are recalled for Agent B." A gap for RE-call today — scope it before selling to SaaS.
3. **Hybrid temporal + vector retrieval** — "vector for semantic, structured for 'all conversations
   from last week'." Zep's whole pitch; demand validated.
4. **Preference / behavior learning, not just fact storage** — *"Mem0 stores memories but doesn't
   learn user patterns; user corrections are the highest-signal data."* Loud, few voices.
5. **Observability — "why was this retrieved?"** — thin as explicit ask, adjacent to trust.
   **RE-call uniquely positioned** (already computes verdict/separability) — surfacing *why* a hit
   was trusted/refused could be a signature feature.

**Roadmap read:** #1 (decay) and #5 (why-was-this-retrieved) reinforce the trust wedge instead of
chasing competitors on their turf. #2 (multi-tenant) is the table-stakes gap to close before SaaS.

---

## Bottom line

RE-call already clears table stakes and holds the one axis the two leaders structurally can't
(honest, calibrated trust) — and they're mired in a benchmark-credibility war that makes honesty
MORE valuable. The problem is **distribution, not product.** First customer to chase: the
indie / coding-agent / support-bot builder who has personally been burned by a confidently-wrong
memory.

## Open follow-ups
- Direct **Reddit** pass (r/LangChain, r/AI_Agents, r/LocalLLaMA) — coverage gap in both research
  passes; would strengthen segments (§1) + trust signal (§2, §4).
- Pull **per-system hallucination numbers out of HaluMem** (arXiv 2511.03506 results tables) for a
  citable, neutral benchmark blog.
- `gh api` pass to quantify 👍 reactions on the decay/multi-tenant feature-request issues to
  prioritize §5.
