# Obsidian: teardown, and an honest comparison with RE-call

**Researched 2026-08-19.** Every external number below is dated to that day and attributed to the
source that published it. Obsidian is a closed-source commercial product that discloses very little,
so several figures are third party estimates and are labelled as such. Nothing here was measured by
me except the RE-call facts, which cite files in this repository.

---

## 1. What Obsidian actually is

A Markdown editor over a folder, with a plugin runtime, sold as free software funded by two
subscription services. That sentence is the whole product, and almost every strategic consequence
follows from it.

| Dimension | Fact | Source |
|---|---|---|
| Desktop runtime | Electron | [DeepWiki, obsidian-help](https://deepwiki.com/obsidianmd/obsidian-help/1.1-application-architecture-and-data-storage) |
| Mobile runtime | Capacitor | same |
| Source | Closed, but the JS bundle is unobfuscated and inspectable | [HN thread](https://news.ycombinator.com/item?id=48180795) |
| Core price | Free, no account, no telemetry, free for commercial use since 2025-02-20 | [obsidian.md/blog/free-for-work](https://obsidian.md/blog/free-for-work/) |
| Sync | 4 USD per user per month annual, 5 USD monthly | [obsidian.md/pricing](https://obsidian.md/pricing) |
| Publish | 8 USD per site per month annual, 10 USD monthly | same |
| Company | Bootstrapped, no VC, roughly 18 staff, grown out of Dynalist | [Fueler](https://fueler.io/blog/obsidian-usage-revenue-valuation-growth-statistics), [OperatorBook](https://www.operatorbook.dev/stories/obsidian-revenue-estimates-2m-to-25m) |
| Revenue | Never disclosed. Estimates run from about 2M USD ARR to a viral 25M USD claim | [OperatorBook](https://www.operatorbook.dev/stories/obsidian-revenue-estimates-2m-to-25m) |
| Reach | About 1.5M monthly active users cited; over 10,000 organizations | [Fueler](https://fueler.io/blog/obsidian-usage-revenue-valuation-growth-statistics), [free-for-work](https://obsidian.md/blog/free-for-work/) |

The revenue spread is wide enough that I would not build a plan on either end of it. What is solid
is the shape: consumer scale, subscription services, no investors to satisfy, and a deliberate
refusal to put anything into the core that is not an editor.

### 1.1 The storage model: files over app

There is no database of record. The filesystem is the database.

- A **vault** is an ordinary folder. Obsidian watches it and picks up external edits.
- **Notes** are UTF-8 `.md` files. Attachments sit beside them as ordinary files.
- **Structured data** is YAML frontmatter, surfaced in the UI as "Properties".
- **Canvas** files are `.canvas`, in [JSON Canvas 1.0](https://jsoncanvas.org/), an open spec that
  Obsidian published under MIT with libraries in C, Dart, Go, Python, React, Rust and TypeScript.
- **Bases** are `.base` files: YAML with five sections (views, filters, formulas, properties,
  summaries). Shipped as a core plugin in 1.9.0, extended through 1.10.0 with a public Bases API.
  See the [Bases syntax reference](https://github.com/obsidianmd/obsidian-help/blob/master/en/Bases/Bases%20syntax.md).
- **Vault config** lives in `.obsidian/` at the vault root: `app.json`, `workspace.json`,
  `hotkeys.json`, `plugins/`, `themes/`, `snippets/`.
- **Global config** lives outside every vault, at `%APPDATA%\Obsidian\` on Windows.

Giving away JSON Canvas as an MIT spec is worth noting as a competitive move, not just an
engineering one. It buys credibility with exactly the audience that distrusts lock in, and it costs
Obsidian nothing, because their moat was never the format.

### 1.2 The derived layer, which is the part that matters here

Everything Obsidian knows beyond raw bytes is a rebuildable cache, and Obsidian treats it that way.

- **`MetadataCache`** holds parsed frontmatter, headings, links, tags and embeds per file. It is
  persisted to **IndexedDB** between sessions and powers graph view, outline, backlinks, Properties
  filtering and Bases.
- **The search index** is a separate in memory lexical index.
- **File Recovery** snapshots `.md` and `.canvas` every five minutes for seven days. It snapshots
  content, never the index, because the index is disposable by design.

Search syntax is genuinely rich: boolean operators, quoted phrases, negation, JavaScript flavoured
regex between slashes, and field operators (`file:`, `path:`, `content:`, `tag:`, `line:`, `block:`,
`section:`, `task:`, `task-todo:`, `task-done:`, `match-case:`, and bracketed property queries such
as `[duration:<5]`). See the [search reference](https://obsidian.md/help/Plugins/Search).

**And it has no relevance ranking.** "Sort search results by relevance" sits in the *Planned*
section of the [public roadmap](https://obsidian.md/roadmap/) as of 2026-08-19, six years after
launch. There is also no semantic search anywhere in core, and Obsidian's stated philosophy is that
there should not be: the core is an editor, everything else is opt in, and every AI feature in
Obsidian today comes from a community plugin.

This is the single most important finding in the whole teardown, and I come back to it in section 4.

### 1.3 The plugin API

- TypeScript compiled to JavaScript, loaded from `.obsidian/plugins/<id>/` with a `manifest.json`
  and a `main.js`, enabled per vault by explicit user action.
- The `App` object exposes `vault`, `metadataCache`, `workspace`, `fileManager`, `keymap`, `scope`,
  plus `renderContext` (1.10.0+) and `secretStorage` (1.11.4+). See the
  [App reference](https://docs.obsidian.md/Reference/TypeScript+API/App).
- `Vault` sits on a `DataAdapter`; `FileSystemAdapter` on desktop exposes `getBasePath()` and
  `getFullPath()`. Events are `create`, `modify`, `delete`, `rename`. Reads are `read` (disk) or
  `cachedRead` (cache). Atomic edits go through `Vault.process` and
  `FileManager.processFrontMatter`. See the
  [Vault and file system reference](https://deepwiki.com/obsidianmd/obsidian-api/2.2-vault-and-file-system).
- **`isDesktopOnly: true`** in the manifest is the escape hatch for anything needing Node or
  Electron. Mobile has neither: no `fs`, no `path`, no `child_process`, no native modules. Plugins
  that support both must gate behind `Platform.isDesktopApp` and `require()` lazily.

### 1.4 The plugin market, and how it just changed

| Metric | Figure | Source |
|---|---|---|
| Published plugins during 2025 | 2,713, with 101,487,612 downloads that year | [Obsidian Stats, 2025 wrapped](https://www.obsidianstats.com/posts/2025-12-04-wrapped-2025) |
| Plugins at the directory relaunch | "over 4,000", 120M downloads | [The future of Obsidian plugins](https://obsidian.md/blog/future-of-plugins/) |
| Plugins listed by a third party tracker | 6,715 plugins, 695 themes | [Obsidian Stats](https://www.obsidianstats.com/) |

Those three counts disagree because they count different things (published in a year, live in the
directory, ever listed). I would quote the range, not a point.

The 2026 directory relaunch changed the rules in ways that bear directly on any plan to ship here:

- **Automated security and quality scans on every version**, not just first submission. Failing a
  scan removes a plugin from search within 24 hours.
- **New submissions must be open source.** Closed source submissions are temporarily blocked.
- **GitHub hosting is currently required.**
- Plugins must self label as **Free**, **Optional payments**, or **Paid**.
- **There is no built in payment system.** Paid plugins do their own license keys, API keys or
  login gates.

So monetising inside the ecosystem is permitted and increasingly normalised, but Obsidian will not
collect money for you, and the source must be public.

### 1.5 Sync, and the door it opened in February 2026

Sync is AES-256 end to end encrypted with a user chosen password, over WebSockets, with
deterministic file hash encryption so identical content deduplicates without the server seeing
plaintext. Limits are 4 GB per vault (attachments and version history included), up to 5 remote
vaults, and one year of version history. The
[reverse engineered server](https://github.com/GiganticThirstyHerald/rev-obsidian-sync) confirms the
transport and the module layout but does not document the wire format.

In **February 2026 Obsidian shipped a headless CLI Sync client**, which runs as a daemon on Linux,
macOS and Windows with no GUI, in a container or on a Raspberry Pi, with the same E2EE. That is on
the roadmap's launched list, and it matters more to RE-call than anything else Obsidian has shipped
recently: it means **a vault can be kept continuously current on a server that RE-call also runs
on.**

### 1.6 Where Obsidian is heading

From the [roadmap](https://obsidian.md/roadmap/) as of 2026-08-19:

- **In progress:** Kanban view for Bases; **Obsidian for Work** ("workplace configuration options to
  control access to plugins"); opening individual `.md` files outside a vault.
- **Planned:** background sync on mobile; Bases in Publish; calendar view for Bases; Canvas in
  Publish; **Multiplayer**; PDF annotation; **sort search results by relevance**.
- **Recently launched:** Airtable import (Aug 2026), iOS Share Sheet and settings search (Jul 2026),
  community directory (May 2026), Obsidian Reader (Mar 2026), headless Sync client and Obsidian CLI
  (Feb 2026), Siri and Shortcuts, mobile widgets (Jan 2026).

Read as a whole: they are building **structure** (Bases), **capture** (Reader, Web Clipper, Share
Sheet), **automation** (CLI, headless) and **the enterprise on ramp** (Obsidian for Work). They are
not building retrieval, and they are not building AI.

---

## 2. The AI and search layer inside Obsidian, which is the real competitive set

RE-call does not compete with Obsidian. It competes with these. All-time downloads from
[Obsidian Stats](https://www.obsidianstats.com/most-downloaded) on 2026-08-19:

| Rank | Plugin | Downloads | What it does | Where it stops |
|---|---|---|---|---|
| 13 | **Claudian** | 1,823,783 | Embeds Claude Code and other coding agents in the vault: chat panel, tag and wikilink suggestions, Mermaid concept maps | No retrieval layer of its own. The agent reads files. |
| 15 | **Omnisearch** | 1,759,407 | BM25 ranking over content, filenames and headings; PDF and OCR through a companion Text Extractor plugin | Lexical only. No vectors, no fusion, no ranking beyond BM25. |
| 17 | **Copilot** | 1,713,125 | Chat with the vault: lexical search plus optional semantic search | Semantic is **off by default**. Partitioned index that errors on large vaults. Indexing disabled on mobile. The docs describe no hybrid fusion and no reranking. |
| 22 | **Smart Connections** | 1,155,348 | Local transformers.js embeddings, chunked by `smart-blocks`, vectors stored in `.smart-env/` inside the vault | Pure cosine similarity. No BM25 fusion, no rerank, no calibration, no abstention. |

For scale, the top of the chart is Excalidraw at 7.34M, Templater at 5.29M, Dataview at 4.80M.

Two things jump out.

**First, the search gap is real, and people pay attention to it.** A plugin whose entire value
proposition is "BM25 ranking" has 1.76M downloads. That is what happens when the host application
still has relevance sorting on the roadmap.

**Second, nobody in this ecosystem ships a trust layer.** Not one of the four can say "I do not
know". Not one models the fact that a note stopped being true. Every one of them returns the nearest
match and lets the language model narrate it. That is precisely the failure mode RE-call was
extracted from.

---

## 3. RE-call as it actually stands today

Facts from this repository at `fa6fe04a`:

| Dimension | Fact |
|---|---|
| Language and size | Python 3.11+, about 54,000 lines across `recall`, `recall_mcp`, `recall_interop`, `recall_consistency` |
| Distribution | `recall-rag` 0.9.5 on PyPI, Apache-2.0 (`pyproject.toml`) |
| Hard dependency | PostgreSQL with pgvector (`psycopg`, `pgvector`) |
| Retrieval | Dense plus Postgres full text plus optional SPLADE, fused by RRF, optional cross encoder rerank, calibrated gap check, trust layer, optional entailment judge |
| Verdicts | Ten, in `recall/types.py:22`: `ok`, `superseded`, `expired`, `not_yet_valid`, `not_yet_known`, `low_confidence`, `invalid_metadata`, `ambiguous_supersession`, `not_entailed`, `unverified` |
| Validity model | `valid_from`, `valid_until`, `supersedes` in frontmatter (`recall/frontmatter.py:17`), with wikilink tolerant key normalisation |
| Temporal model | Bi-temporal: validity time from frontmatter, transaction time from `first_indexed_at` and `indexed_at`, so `known_as_of` answers "what did we hold on Tuesday" |
| Ingestion | Markdown, text, HTML, plus PDF, DOCX, XLSX, PPTX, MSG and legacy Office (`recall/extraction.py:94`) |
| Lifecycle | Immutable generations: build, validate, calibrate, promote, rollback, gc |
| Multi tenancy | Tenant IDs, row level security, serving and migration DSN split, scoped bearer tokens, quotas, erasure |
| Agent surface | MCP server with `recall_search`, `recall_evidence`, `recall_index`, `recall_ingest`, `recall_forget`, `recall_stats`, `recall_tenants`, `recall_job_status`, plus calibration and reasoning tools |
| Human surface | CLI, plus a PySide6 Windows desktop shell (`recall/desktop/ui.py`, about 1,880 lines) that drives a Docker Postgres |

And the limits this repository already publishes about itself, which are the ones that decide
whether the Obsidian market is reachable:

- Authored supersession has **terrible coverage**. On the reference corpus, 2 of 792 memos declared
  `supersedes:` while 60 closed a decision only in prose, and `lint --fix` could safely infer zero
  of those 60 (`docs/PRIOR_ART.md`).
- On LOCOMO's 446 adversarial questions, RE-call abstains on **zero** out of the box; calibration
  and the entailment judge raise that to 0.37 to 0.77 only by refusing a quarter to half of
  legitimate questions (`docs/PRIOR_ART.md`).
- Truth extraction, the mechanism that could infer the missing edges, is **off unless
  `RECALL_TRUTH_EXTRACTION` is set** (`recall/truth_extraction/_engine.py:164`).

---

## 4. The comparison

### 4.1 Category honesty first

Obsidian is a consumer application with an editor, a mobile client, a sync business and roughly 1.5M
monthly users. RE-call is a retrieval library. They are not the same kind of object, and a feature
by feature scoreboard between them would be theatre.

**The useful framing is that Obsidian is a corpus and a distribution channel, not a rival.** The
competitors are Smart Connections, Copilot and Omnisearch. That is the comparison worth running, and
it is a far more favourable one.

### 4.2 What Obsidian does better, where RE-call has no answer

1. **Zero install, zero config, no account.** Download, open a folder, start typing. RE-call needs
   PostgreSQL, pgvector, a schema migration, an embedder download, and a calibration run before
   strict mode will answer at all. For a consumer that is not a friction gap, it is a different
   universe.
2. **A user interface.** Obsidian sells an editor people live in for eight hours a day. RE-call has
   a CLI, an MCP server, and a Windows launcher for a Docker container.
3. **Mobile.** Capacitor on iOS and Android. RE-call cannot run there at all, and neither can
   Postgres or Python. This is not a roadmap item, it is a structural exclusion.
4. **Durability of the artefact.** An Obsidian vault outlives Obsidian, because the notes are just
   files. A RE-call index is derived state inside a database somebody has to operate and back up.
5. **Ecosystem gravity.** Thousands of plugins, over 100M downloads a year, a directory with
   automated review and a developer dashboard. RE-call has none of this and cannot manufacture it.
6. **Sync as a business model.** E2EE sync at 4 USD per user per month, sold into an installed base
   that already trusts them. RE-call has no distribution at all.
7. **Open formats given away.** JSON Canvas under MIT. That is a trust asset in this audience.
8. **The philosophy is already theirs.** Local first, no lock in, no account, privacy by default.
   Those are the values RE-call claims. Obsidian is the reference implementation of them, and it got
   there in 2020.

### 4.3 What RE-call does better, where the whole Obsidian ecosystem has nothing

1. **Ranking at all.** Obsidian core still has no relevance sort. Omnisearch adds BM25. Smart
   Connections adds cosine. **Nobody fuses them.** RE-call ships dense plus full text plus optional
   learned sparse, fused with RRF, with optional cross encoder reranking on top.
2. **Abstention as a returned value.** No tool in that ecosystem can say "no result here is good
   enough to answer from". RE-call returns a calibrated abstention with a reason.
3. **Supersession.** Nothing in Obsidian, core or plugin, models "this note replaced that one". The
   metadata cache has links and tags; it has no notion of a claim being retired.
4. **Bi-temporal query.** "What did I believe in March" is unanswerable in Obsidian even in
   principle: the cache holds current state only, and version history is a one year Sync artefact,
   not a queryable axis. RE-call's `known_as_of` answers it directly.
5. **Per hit provenance.** Verdict, confidence, source, validity metadata. The plugins return a list
   of notes and a similarity score.
6. **Non Markdown corpora as first class.** PDF, DOCX, XLSX, PPTX, MSG. Obsidian search does not
   read PDFs at all; Omnisearch bolts it on through a second plugin.
7. **A versioned index.** Immutable generations with build, validate, calibrate, promote and
   rollback. No plugin versions its index; they rebuild and hope.
8. **Tenancy, RLS, scoped tokens, quotas, erasure.** Irrelevant to a solo note taker. Decisive for
   Obsidian for Work, which is in progress on their roadmap right now and has no retrieval story.
9. **Agent native by construction.** A typed MCP surface with an explicit evidence tool, rather than
   a chat sidebar that pastes note text into a prompt.

### 4.4 What RE-call is missing to be viable in this market

Ordered by how much each one hurts.

1. **The authored edge coverage problem, restated for consumers.** RE-call's differentiator depends
   on authors writing `supersedes:` in frontmatter. Obsidian users will do this approximately never;
   RE-call's own corpus managed 2 in 792. **In a consumer vault the validity layer has close to zero
   coverage, and RE-call degrades to a well ranked hybrid search.** That is still worth something
   (Omnisearch proves BM25 alone is worth 1.76M downloads), but it is not the thesis.
2. **No zero install path.** Postgres plus pgvector is disqualifying for a consumer plugin. An
   embedded store (SQLite with a vector extension, or a bundled single binary) would be a
   prerequisite, and that is a substantial engineering programme, not an adapter.
3. **No JavaScript or WASM path.** Everything is Python. A plugin cannot ship Postgres, and the
   plugin runtime cannot call Python without spawning a process, which forces `isDesktopOnly`.
4. **No mobile story at all**, and no way to build one on the current stack.
5. **No plugin.** There is no `.obsidian/plugins/recall/` anything, and no UI in which a human could
   see a verdict. A verdict nobody can see is worthless.
6. **No file watching.** RE-call indexes a path. It does not subscribe to `create`, `modify`,
   `delete` and `rename`, which is how every Obsidian plugin stays current.
7. **The link graph is unused.** Wikilinks, backlinks and tags are the most distinctive structure in
   a vault, and the strongest freely available relevance signal. RE-call parses wikilinks only to
   normalise `supersedes:` keys; it does not use the graph for retrieval.
8. **`.canvas` and `.base` are not ingestible.** Both are first class Obsidian content, and both are
   structured formats RE-call's extractor will refuse.
9. **Nothing writes back into the editor.** The valuable move would be proposing a `supersedes:` line
   where RE-call detects a contradiction. `rewrite plan` and `rewrite apply` exist; an editor surface
   for them does not.

---

## 5. Strategic read

Three plays, and they are not equally good.

### Play A: ship an Obsidian plugin

Highest reach, worst fit. It requires either a retrieval core rewritten in TypeScript or WASM, or a
locally spawned sidecar process with `isDesktopOnly: true`, which forfeits mobile and adds an
install step to a market defined by not having one. Precedent exists (the Git plugin and Text
Extractor both use Node), but no precedent exists for a plugin that requires a database server, and
I do not think one would survive contact with the audience.

If this is ever attempted, the honest version is not "RE-call for Obsidian". It is a hybrid search
plus abstention plugin on an embedded store, with validity as an advanced feature most users never
touch.

### Play B: MCP server over a vault, starting now

Lowest effort, best fit with what already exists. RE-call is already an MCP server. Point it at a
vault folder, index `**/*.md`, and expose `recall_search` with verdicts to Claude Code, Claude
Desktop or Cursor. The headless Sync client shipped in February 2026 means a server can keep the
vault current with no GUI, which makes a continuously indexed remote vault a supported configuration
rather than a hack.

The audience is the intersection of Obsidian users and agent users. Much smaller than 1.5M, but it
is a population that installs Postgres without complaining, and Claudian's 1.82M downloads say it is
already large and growing fast. This is roughly a day of glue plus a good README, and it is the
cheapest possible test of whether the thesis lands with vault owners.

One piece of engineering is worth doing before that test: **file watching and incremental
reindexing**, because a stale index is the first thing that will kill the demo.

### Play C: aim at Obsidian for Work, not at consumers

Obsidian for Work is *in progress* on their roadmap, described as "workplace configuration options
to control access to plugins", and over 10,000 organizations already use Obsidian at work. Obsidian
has no enterprise retrieval story, no tenancy, no audit trail, no erasure guarantees and no
abstention. RE-call has all of those, and they are exactly the parts a consumer would never pay for.

In that setting the coverage problem also softens, because organisational notes have owners, reviews
and change processes, so declaring supersession becomes a plausible policy rather than a hope. And
"install Postgres" is a normal sentence in that room.

### Recommendation

**B as the wedge, C as the business, A only if B shows demand.**

One warning I want stated plainly before any of this starts. The pitch into this market cannot be
"validity aware retrieval" until the coverage problem has an answer, because in a vault nobody
declares supersession and the feature will not fire. Either truth extraction has to earn its way on
by default, or the honest pitch here is **ranking and abstention first, validity as the upsell**.
Leading with the thesis and delivering hybrid search would be the fastest way to burn the
credibility this project has spent a year building on measured claims.

---

## Sources

- [obsidian.md](https://obsidian.md/) and [pricing](https://obsidian.md/pricing)
- [Obsidian roadmap](https://obsidian.md/roadmap/)
- [Obsidian is now free for work](https://obsidian.md/blog/free-for-work/)
- [The future of Obsidian plugins](https://obsidian.md/blog/future-of-plugins/)
- [Obsidian search reference](https://obsidian.md/help/Plugins/Search)
- [How Obsidian stores data](https://obsidian.md/help/Files+and+folders/How+Obsidian+stores+data)
- [App API reference](https://docs.obsidian.md/Reference/TypeScript+API/App)
- [Build a plugin](https://docs.obsidian.md/Plugins/Getting+started/Build+a+plugin)
- [Mobile development](https://docs.obsidian.md/Plugins/Getting%20started/Mobile%20development)
- [Vault and file system, DeepWiki](https://deepwiki.com/obsidianmd/obsidian-api/2.2-vault-and-file-system)
- [Application architecture and data storage, DeepWiki](https://deepwiki.com/obsidianmd/obsidian-help/1.1-application-architecture-and-data-storage)
- [Bases syntax](https://github.com/obsidianmd/obsidian-help/blob/master/en/Bases/Bases%20syntax.md)
- [JSON Canvas](https://jsoncanvas.org/) and [the announcement](https://obsidian.md/blog/json-canvas/)
- [Reverse engineered Obsidian Sync server](https://github.com/GiganticThirstyHerald/rev-obsidian-sync)
- [Obsidian Stats, most downloaded](https://www.obsidianstats.com/most-downloaded) and [2025 wrapped](https://www.obsidianstats.com/posts/2025-12-04-wrapped-2025)
- [Smart Connections architecture, DeepWiki](https://deepwiki.com/brianpetro/obsidian-smart-connections)
- [Copilot for Obsidian, vault search and indexing](https://www.obsidiancopilot.com/docs/vault-qa)
- [Omnisearch](https://github.com/scambier/obsidian-omnisearch)
- [Claudian](https://github.com/YishenTu/claudian)
- [Obsidian revenue estimates, OperatorBook](https://www.operatorbook.dev/stories/obsidian-revenue-estimates-2m-to-25m)
- [Obsidian statistics, Fueler](https://fueler.io/blog/obsidian-usage-revenue-valuation-growth-statistics)
- [Headless Sync, DeepWiki](https://deepwiki.com/obsidianmd/obsidian-help/2.5-headless-sync)
