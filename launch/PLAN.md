# RE-call launch plan

Reverse-engineered from Mem0's actual playbook (verified: Show HN 2024-09-04, 201 pts; rebrand of
Embedchain's 8k⭐/2M downloads; category ownership; benchmark-as-marketing; integration loops),
adapted for a cold start with no inherited audience.

**The core adaptation:** Mem0 did *not* go 0→breakout — they inherited Embedchain's audience. You
can't copy that. Your substitute for an inherited audience is a **contrarian, measured finding**
(the abstention axis the field skips) that earns its own attention. Credibility replaces the batch
badge; the finding replaces the inherited stars.

---

## Gate 0 — FRONT DOOR (blocking; nothing else converts until done)

- [x] PyPI publish unblocked (you did this).
- [x] README merge-conflict markers resolved locally (branch `fix/readme-merge-markers-052`,
      commit 12fd890). **0.5.1 shipped the markers to GitHub + PyPI — verified live.**
- [ ] **Push the fix branch + merge to master** → fixes the GitHub landing page. (needs your OK)
- [ ] **Publish 0.5.2 to PyPI** → fixes the pip page (0.5.1's description is frozen with the
      markers forever; only a new version repairs it). Rebuild from the bumped `pyproject.toml`.
- [ ] Sanity-check: open the PyPI 0.5.2 page and the GitHub landing page in a fresh browser; confirm
      the banner renders, no markers, `pip install recall-rag` works in a clean venv.

## 2 — POSITIONING (largely DONE — do not rewrite)

Your README already nails the wedge: *"Trustworthy retrieval for an AI agent's own memory… or the
honest answer is 'I don't know.'"* plus a competitor table (Mem0/Zep/Graphiti) and a
"claims that were withdrawn" section most projects would never dare publish. That honesty *is* the
brand. Leave it.

- [ ] One small enhancement: add a one-line category claim near the very top that a first-time
      visitor arriving from a "Mem0 alternative" search will grok in 3 seconds — e.g.
      *"the memory layer that knows when it doesn't know."* You own an axis; say so above the fold.

## 3 — CREDIBILITY ARTIFACT (the crown jewel — data already exists)

- [x] Abstention ablation is measured (`results/locomo_abstention.json`): the honest 0.00 → 0.37–0.76
      trade-off curve. This is a *better* story than a vanity number.
- [x] Launch blog drafted → `launch/blog-the-question-benchmarks-skip.md`.
- [ ] Consider committing the eval harness + results (currently untracked) so the numbers are
      citable and reproducible from the repo. Run your usual honesty/CCA pass first.
- [ ] (Stretch, high-value) The real head-to-head: run Mem0 + Zep through the *same* 446 adversarial
      questions and publish the abstention column for all three. Needs their extraction-LLM keys and
      a neutral harness — scoped as a follow-up, framed in the blog as an open invitation so you
      never quote a number you didn't measure.
- [ ] (Stretch) An arXiv note on "abstention on LOCOMO's adversarial split" — this is your J-score
      substitute, the credibility artifact that replaces the YC stamp.

## 4 — LAUNCH MOMENT

⛔ **HN deferred** — account can't post right now (`launch/show-hn.md` kept for when that changes).
With HN out, **r/LocalLLaMA is the primary spike channel** and the dev.to post is the durable home.

Drafts ready for review (all in `launch/`, nothing published):
- [x] **dev.to post** → `launch/devto-post.md` (`published: false` = imports as a draft; the canonical home)
- [x] **X/Twitter thread** → `launch/x-thread.md` (last tweet links the dev.to post)
- [x] **Reddit** → `launch/reddit-posts.md` (r/LocalLLaMA self-post + r/MachineLearning `[P]`)
- [x] **Discord** → `launch/discord-messages.md` (LangChain + LlamaIndex #showcase channels)
- [ ] Show HN → `launch/show-hn.md` (DEFERRED until HN posting is available)

Sequence (one weekday morning, ~08:00–10:00 ET, Tue–Thu):
1. Publish the dev.to post (flip `published: true`). Grab its URL.
2. Post the X thread + the r/LocalLLaMA post, both linking the dev.to URL.
3. Drop the Discord messages in the right #showcase/#tools channels.
4. Space the r/MachineLearning `[P]` post a day or two later (don't blast every channel at once).
5. Be present to reply for the first few hours everywhere. Never solicit upvotes.

## 5 — DISTRIBUTION LOOP (the only thing that compounds)

Mem0's durable growth was integrations, not stars. Each framework you plug into imports its
audience into yours.

- [x] MCP server already ships (`recall_mcp/`) — you're already on a 2026-native distribution
      surface Mem0 barely occupies. Lead with this: *"AI memory as an MCP server."*
- [ ] Ship a **LangChain `BaseMemory` / retriever adapter** so RE-call is a drop-in `memory=`
      backend. Highest-leverage single integration. Do it as a proper TDD task (repo runs real
      pgvector + type/coverage gates) — worth its own focused pass.
- [ ] Then a **LlamaIndex** vector-store/memory adapter.
- [ ] Get listed in `awesome-llm`, `awesome-ai-agents`, `awesome-mcp-servers` (free, durable SEO).
- [ ] Add a "Using with Claude / MCP" quickstart to the docs front-and-center (you have
      `docs/USING_WITH_CLAUDE.md` — surface it).

## 6 — AMPLIFICATION (only after 0–5 land)

- [ ] Don't spend attention or money until the front door is fixed and one integration exists.
- [ ] A short demo GIF of the superseded-catch (you have the asset) does more than any ad.

---

## What NOT to do (learned from the teardown)
- Don't compete on raw accuracy headline numbers — you'll lose the framing war to two funded teams
  and it contradicts your own honesty posture. Compete on the axis they don't report.
- Don't quote a 3-way benchmark you haven't run. The invitation framing is stronger than a claim.
- Don't buy amplification into a broken funnel. Gate 0 first, always.
