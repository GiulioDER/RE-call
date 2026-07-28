# X account setup — @REcallRAG

Copy-paste kit. Phase zero: profile first, then one launch post. No competitor
framing, no benchmark numbers — those are a later, separate move (`launch/x-thread.md`).

Assets live next to this file, in `launch/x/`.

---

## 1. Profile fields

**Name** (50 max)

```
RE-call
```

**Handle** — `@REcallRAG` (already registered)

**Bio** (146 / 160 — paste all three lines)

```
The memory layer that knows when it doesn't know.
Verdict · confidence · provenance — or an honest "I don't know."
MIT · Postgres + pgvector · MCP
```

Line 1 is the category claim. Line 2 is what it does. Line 3 is the "is this real"
check a technical visitor makes in about two seconds.

**Website**

```
https://github.com/GiulioDER/RE-call
```

The repo, not PyPI — the repo has the banner, the demo output and the docs. PyPI is a
place to install from once someone has already decided.

**Location** — leave empty. On a project account it reads as filler.

**Birth date** — leave unset, or set it to visible-to-nobody. X uses it for ad targeting.

---

## 2. Images

| Slot | File | Size |
|---|---|---|
| Header | `header-1500x500.png` | 1500 × 500 |
| Avatar — option A | `avatar-monogram-400.png` | 400 × 400 |
| Avatar — option B | `avatar-nodes-400.png` | 400 × 400 |

**Header** is the README banner rescaled and cropped to the band that keeps the "RE-call"
wordmark clear of the avatar. A straight crop of the original would have buried the
wordmark under the profile picture — X punches the avatar through the header's
bottom-left corner.

**Avatar — pick one.** The monogram (A) is the safer choice: X renders avatars at roughly
48 px in the timeline, and at that size the node-cluster (B) collapses into a coloured
glow. B is more distinctive on the profile page itself and matches the header seamlessly.
A if you care most about being recognisable in a busy feed, B if you care most about the
profile page looking like one designed object.

---

## 3. The launch post

Pin whichever one you post. Character counts include X's rule that any URL costs 23
characters regardless of length.

### Option A — the contrast punch *(recommended, 250 / 280)*

```
Most RAG hands back the closest vector match.

RE-call hands back a verdict.

Trustworthy retrieval for an AI agent's own memory — confidence, provenance, and an honest "I don't know."

MIT. pip install "recall-rag[fastembed]"
github.com/GiulioDER/RE-call
```

Two short lines set up a contrast anyone doing RAG feels immediately, and the payoff
lands before the fold. This is your README's own opening move, compressed.

### Option B — the failure-mode hook (257 / 280)

```
Your agent's memory has a failure mode nobody names: when you reverse a decision, the stale version is often the highest-cosine hit in the whole result.

RE-call demotes it — and says "I don't know" when it should.

Open source, MIT.
github.com/GiulioDER/RE-call
```

Leads with the problem instead of the product. Stronger for people who have been bitten
by exactly this; slower to reach the point for everyone else.

### Option C — the plain ship note (244 / 280)

```
Shipped RE-call — the memory layer that knows when it doesn't know.

Retrieval for AI agents that returns a verdict, not just the nearest vector. Postgres + pgvector, MCP server, MIT.

pip install "recall-rag[fastembed]"
github.com/GiulioDER/RE-call
```

Least risky, least memorable.

**Attach `docs/banner.png`** to whichever you pick — 1280 × 640 crops cleanly in the
timeline. Do *not* attach `header-1500x500.png`; at 3:1 it letterboxes badly in a post.

**Then pin it.** ⋯ menu on the post → *Pin to your profile*.

---

## 4. Before you post — check the front door

You already lost a launch window to a broken landing page (0.5.1 shipped merge-conflict
markers to both GitHub and PyPI). Same check, ninety seconds:

- [ ] Open `https://github.com/GiulioDER/RE-call` in a **logged-out / private** window.
      Banner renders, no stray markup, badges green.
- [ ] Open `https://pypi.org/project/recall-rag/`. Description renders, version reads
      0.5.3, install command matches the post.
- [ ] Run `pip install "recall-rag[fastembed]"` in a clean venv. The post's command has
      to be the one that works — plain `pip install recall-rag` installs no embedder.
- [ ] Open the X profile itself once saved: avatar not cropping the monogram, header
      wordmark not sitting under the avatar, bio not truncated with an ellipsis.

Anyone arriving from the post hits these pages within seconds of reading it.

---

## 5. After the post

Not part of phase zero — noted so it isn't lost:

- Reply to everyone who engages, for the first few hours. Founder presence in-thread is
  most of what converts a launch post.
- The LOCOMO-adversarial thread (`launch/x-thread.md`) is the *second* move, and it wants
  the dev.to post live first so its last tweet has somewhere to point.
- Never solicit engagement, never buy it.
