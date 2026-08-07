"""Emits docs/pipeline.svg — the full RE-call pipeline, ingest at the top, answer at the bottom.

Hand-drawn SVG in the same GitHub-dark card language as docs/banner.svg, but generated rather
than typed: every cell's height falls out of its wrapped body text, so editing a description
cannot silently push a box through the one below it.

The colour semantics are one rule applied to every cell, not decoration:

    light green   the thing RE-call actually ships
    light blue    the best-measured option, whatever it costs
    dark green    a free alternative that stays on your machines
    dark blue     an option whose price is your corpus leaving the building
    amber         opt-in, off by default
    red           measured and rejected, or refused outright

    python docs/make_pipeline_diagram.py > docs/pipeline.svg
"""
from __future__ import annotations

import html
import sys
import textwrap

# ---------------------------------------------------------------------------------------------
# palette — GitHub dark, same values as docs/banner.svg so the two read as one family
# ---------------------------------------------------------------------------------------------
BG = "#0d1117"
CARD_STROKE = "#30363d"
PANEL = "#161b22"
PANEL_STROKE = "#21262d"
INK = "#e6edf3"
INK_2 = "#adbac7"
INK_3 = "#8b949e"
MUTED = "#6e7681"
RULE = "#30363d"

#: (fill, stroke, chip) per semantic class. The fills are near-black on purpose: a saturated
#: panel would out-shout the text it exists to label.
STYLES = {
    "standard": ("#12241a", "#2ea043", "#56d364"),  # light green — shipped
    "best": ("#0e1e33", "#58a6ff", "#79c0ff"),  # light blue  — best measured
    "free": ("#101a13", "#238636", "#238636"),  # dark green  — free, local
    "egress": ("#0d1524", "#1f6feb", "#1f6feb"),  # dark blue   — data leaves
    "optional": ("#221a0c", "#9e6a03", "#d29922"),  # amber       — opt-in
    "rejected": ("#24120f", "#8c2c25", "#f85149"),  # red         — rejected/refused
}

LEGEND = [
    ("standard", "shipped default", "what runs if you change nothing"),
    ("best", "best measured", "highest quality on the numbers we have"),
    ("free", "free · stays local", "no API key, no egress, $0 marginal"),
    ("egress", "cloud · data leaves", "your corpus is sent to a vendor"),
    ("optional", "opt-in", "off by default; measured cost"),
    ("rejected", "rejected / refused", "measured and turned down, with the reason"),
]

# ---------------------------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------------------------
W = 1400
MAIN_X, MAIN_W = 196, 780
#: The side column starts 80px clear of the main one, not 24: the connectors into it carry
#: labels ("optional"), and a label needs a horizontal run long enough to sit on.
SIDE_X, SIDE_W = 1056, 304
PAD = 20
GAP = 40  # vertical space between two stacked cells, i.e. the arrow's room

TITLE_SIZE = 14.0
BODY_SIZE = 11.6
LINE_H = 16.4

FONT_SANS = 'font-family="-apple-system,\'Segoe UI\',Roboto,Helvetica,Arial,sans-serif"'
FONT_MONO = 'font-family="ui-monospace,\'SF Mono\',\'Cascadia Code\',Menlo,Consolas,monospace"'


def esc(text: str) -> str:
    return html.escape(text, quote=False)


def wrap(text: str, width_px: float, size: float = BODY_SIZE) -> list[str]:
    """Word-wrap to a pixel budget, using the average glyph width of this sans at `size`."""
    chars = max(8, int(width_px / (size * 0.512)))
    return textwrap.wrap(text, chars) or [""]


def body_lines(text: str, width_px: float) -> list[str]:
    """Wrap paragraphs separated by a literal newline, keeping the break."""
    out: list[str] = []
    for para in text.split("\n"):
        out.extend(wrap(para, width_px))
    return out


class Canvas:
    def __init__(self) -> None:
        self.parts: list[str] = []

    def add(self, markup: str) -> None:
        self.parts.append(markup)

    def text(
        self,
        x: float,
        y: float,
        content: str,
        *,
        size: float = BODY_SIZE,
        fill: str = INK_3,
        weight: str = "400",
        mono: bool = False,
        anchor: str = "start",
        opacity: float = 1.0,
        style: str = "",
    ) -> None:
        font = FONT_MONO if mono else FONT_SANS
        extra = f' font-style="{style}"' if style else ""
        op = f' opacity="{opacity}"' if opacity != 1.0 else ""
        self.add(
            f'<text x="{x:.1f}" y="{y:.1f}" {font} font-size="{size}" fill="{fill}" '
            f'font-weight="{weight}" text-anchor="{anchor}"{extra}{op}>{esc(content)}</text>'
        )


def tag_stacked(title: str, tag: str, width: float) -> bool:
    """True when the tag will not fit beside the title, so it has to go on its own line.

    Measured rather than eyeballed, because the narrow side column is exactly where a title and
    its right-aligned tag overlap — and an overlap in generated SVG is silent: the file is still
    valid, it just reads as one word made of two.
    """
    if not tag:
        return False
    return (len(title) * TITLE_SIZE * 0.58 + len(tag) * 10.2 * 0.62 + 30) > (width - 2 * PAD)


def cell_height(title: str, body: str, width: float, tag: str = "", extra: float = 0.0) -> float:
    lines = body_lines(body, width - 2 * PAD - 26)
    stacked = 15 if tag_stacked(title, tag, width) else 0
    return PAD + 18 + 6 + stacked + len(lines) * LINE_H + PAD - 4 + extra


def draw_cell(
    c: Canvas,
    x: float,
    y: float,
    width: float,
    title: str,
    body: str,
    kind: str,
    *,
    tag: str = "",
) -> float:
    """One pipeline cell. Returns its height."""
    fill, stroke, chip = STYLES[kind]
    text_w = width - 2 * PAD - 26
    lines = body_lines(body, text_w)
    height = cell_height(title, body, width, tag)
    stacked = tag_stacked(title, tag, width)

    c.add(
        f'<rect x="{x}" y="{y:.1f}" width="{width}" height="{height:.1f}" rx="10" '
        f'fill="{fill}" stroke="{stroke}" stroke-opacity="0.85"/>'
    )
    # the class chip: a 4px rail down the left edge, so the colour is legible at thumbnail size
    c.add(
        f'<rect x="{x + 1:.1f}" y="{y + 9:.1f}" width="4" height="{height - 18:.1f}" rx="2" '
        f'fill="{chip}"/>'
    )
    c.text(x + PAD + 6, y + PAD + 12, title, size=TITLE_SIZE, fill=INK, weight="700")
    if tag and not stacked:
        c.text(
            x + width - PAD,
            y + PAD + 11,
            tag,
            size=10.2,
            fill=chip,
            mono=True,
            anchor="end",
        )
    elif tag:
        c.text(x + PAD + 6, y + PAD + 28, tag, size=10.2, fill=chip, mono=True)
    ty = y + PAD + 12 + 20 + (15 if stacked else 0)
    for line in lines:
        c.text(x + PAD + 6, ty, line, fill=INK_3)
        ty += LINE_H
    return height


def arrow(
    c: Canvas,
    x: float,
    y0: float,
    y1: float,
    *,
    label: str = "",
    colour: str = "#484f58",
    dashed: bool = False,
) -> None:
    dash = ' stroke-dasharray="5 4"' if dashed else ""
    c.add(
        f'<line x1="{x:.1f}" y1="{y0:.1f}" x2="{x:.1f}" y2="{y1 - 9:.1f}" stroke="{colour}" '
        f'stroke-width="1.6"{dash}/>'
    )
    c.add(
        f'<path d="M {x - 5:.1f} {y1 - 9:.1f} L {x + 5:.1f} {y1 - 9:.1f} L {x:.1f} {y1:.1f} Z" '
        f'fill="{colour}"/>'
    )
    if label:
        c.text(x + 12, (y0 + y1) / 2 + 4, label, size=10.6, fill=colour, mono=True)


def elbow(
    c: Canvas,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    *,
    label: str = "",
    colour: str = "#9e6a03",
    dashed: bool = True,
    head: str = "right",
) -> None:
    """A right-angle connector from the main column out to the side column (or back)."""
    dash = ' stroke-dasharray="5 4"' if dashed else ""
    mid = x0 + (x1 - x0) * 0.5
    c.add(
        f'<path d="M {x0:.1f} {y0:.1f} H {mid:.1f} V {y1:.1f} H {x1 - (9 if head == "right" else -9):.1f}" '
        f'fill="none" stroke="{colour}" stroke-width="1.6"{dash}/>'
    )
    tip = 1 if head == "right" else -1
    c.add(
        f'<path d="M {x1 - 9 * tip:.1f} {y1 - 5:.1f} L {x1 - 9 * tip:.1f} {y1 + 5:.1f} '
        f'L {x1:.1f} {y1:.1f} Z" fill="{colour}"/>'
    )
    if label:
        # On the horizontal run leaving the main column, never at the elbow: the corner is where
        # the arrowhead and the target box both are, and a label there lands on top of both.
        c.text(x0 + (14 if head == "right" else -14), y0 - 9, label, size=10.6, fill=colour,
               mono=True, anchor="start" if head == "right" else "end")


# ---------------------------------------------------------------------------------------------
# content — every phase, top to bottom
# ---------------------------------------------------------------------------------------------
INGEST = [
    (
        "1 · Memos on disk",
        "Plain markdown with YAML frontmatter. Validity is DECLARED by the author — `supersedes:`, "
        "`valid_from:`, `valid_until:` — never inferred by an LLM at ingest. That is the whole reason "
        "writing a memory costs $0.",
        "standard",
        "corpus",
    ),
    (
        "2 · recall lint — the write-time guard",
        "Checks the supersession graph before anything is indexed: dangling, self- and cyclic edges, "
        "malformed dates, version-sibling and closed-in-prose smells. No database needed, exit 1 on "
        "error, CI-ready.\n"
        "`--semantic` (opt-in, DB-backed) adds the missing-edge check: a new memo highly similar to a "
        "prior closed decision it never references.",
        "standard",
        "CI gate",
    ),
    (
        "3 · Walk · confine · content-hash skip",
        "Root-confined file walk; every file keyed by its ROOT-RELATIVE path so a/notes.md and "
        "b/notes.md cannot collide in provenance. Unchanged files are skipped on content hash, files "
        "gone from disk are pruned behind a guard. 5,100 chunks / 1,120 files: full 7.4 s, unchanged "
        "re-index 0.22 s.",
        "standard",
        "incremental",
    ),
    (
        "4 · Chunking",
        "`chunk_text` packs markdown blocks to ~800 chars with overlap and splits oversized ones; "
        "`chunk_code` breaks on def / class / decorator boundaries. 400 / 800 / 1600 were measured — "
        "the shipped 800 was already best on MRR. Re-indexing REPLACES a file's rows wholesale, so a "
        "withdrawn `supersedes:` leaves no stale chunk behind.",
        "standard",
        "800 chars",
    ),
    (
        "5 · Contextualisation",
        "How much of a chunk's surroundings is folded into the text that gets embedded: `none` "
        "(shipped) / `document` / `section` / `neighbor`. The mode is derived into the profile's "
        "`context_version`, never declared twice — so a context change is a NEW embedding identity "
        "and cannot silently disagree with vectors already stored.",
        "optional",
        "none by default",
    ),
]

EMBEDDERS = [
    (
        "voyage-3",
        "best",
        "cloud · Voyage",
        "The best retrieval quality measured here. +0.282 hit@5 over bge-small on an idiosyncratic "
        "private corpus (0.348 → 0.630) and it wins 16 of 17 BEIR / CQADupStack corpora, median "
        "≈ +0.06 hit@5, with the gap growing with the haystack. Price: 45 ms → 246 ms per "
        "query, an API dependency, and your corpus leaves your infrastructure.",
    ),
    (
        "voyage-4-large",
        "egress",
        "cloud · Voyage",
        "The current Voyage generation; used for the BEAM k-sweep (k=45). Its cosines sit in an "
        "entirely different regime — the shipped 0.50 floor starves 23.3 % of questions on it. No "
        "head-to-head quality number against the locals yet.",
    ),
    (
        "voyage-finance-2",
        "egress",
        "cloud · domain",
        "FinanceBench, 150 questions over 72,151 chunks: dense + reranker went 0.393 → 0.527 "
        "(p<0.001) when the candidate pool grew 40 → 100 with ef_search corrected. Measured only on "
        "its own domain, so it ranks here on evidence rather than above it.",
    ),
    (
        "bge-large-en-v1.5",
        "free",
        "local · free",
        "The strongest LOCAL result: on LOCOMO's same-embedder control it scored judged accuracy "
        "0.478 against text-embedding-3-small's 0.412–0.422 on the same benchmark. Cosine median "
        "0.827 (memory) / 0.782 (BEAM), starves nothing at the shipped floor.",
    ),
    (
        "bge-small-en-v1.5",
        "standard",
        "★ shipped default",
        "384-dim, FastEmbed, offline, no API key, $0 forever. Dense-only nDCG@10 0.97 where the "
        "corpus is easy; hit@5 0.705 on public technical prose and 0.348 on a jargon-heavy private "
        "one — that spread IS the finding. Highest cosine median of everything measured "
        "(0.852 / 0.819) and 0 % starved at the 0.50 floor.",
    ),
    (
        "text-embedding-3-small",
        "egress",
        "cloud · OpenAI",
        "Mem0's documented default, carried so the head-to-head runs on the competitor's own "
        "embedder: RE-call 0.412 / 0.422 judged accuracy vs Mem0's 0.366 (n=1,540). It is also the "
        "one model the shipped 0.50 gap floor mis-fits — median 0.710 / 0.608, 16 % of BEAM "
        "questions starved.",
    ),
    (
        "gte-base",
        "free",
        "local · free",
        "768-dim, a DIFFERENT architecture family — run as the ladder-v3 generality arm, where 90 % "
        "of the questions monotone under bge-small stayed monotone under it. Explicitly NOT a quality "
        "comparison: nothing measured says it is better or worse.",
    ),
    (
        "all-MiniLM-L6-v2",
        "free",
        "local · free",
        "The fine-tuning base. Recall@5 1.00 held-out BEFORE and AFTER contrastive fine-tuning on "
        "(query, gold-chunk) pairs — an honest null: a modern small embedder already retrieved the "
        "right chunk for every held-out query, so there was nothing left to win.",
    ),
    (
        "Qwen3-Embedding-0.6B",
        "rejected",
        "rejected 2026-08-03",
        "Rejected on CPU SERVING LATENCY alone, at a four-thread budget: query p95 5.8 s and a "
        "20-passage batch p50 of 41 s against a 250 ms profile budget. Its retrieval quality was "
        "never measured, so nothing here says it is worse at retrieval — the rejection record is "
        "kept in the registry with its numbers so the next session cannot re-litigate it.",
    ),
    (
        "hashing-64",
        "free",
        "local · no model",
        "Dependency-free and non-semantic, so the whole test suite runs offline. Its answerable and "
        "unanswerable cosine distributions OVERLAP, so no threshold separates them: the floor case "
        "proving gap detection is bounded by the embedder, not by the guard.",
    ),
]

EMBEDDER_FOOTNOTE = (
    "Ordered best → worst by measured retrieval quality where a number exists. These numbers come "
    "from different corpora and are not one leaderboard — the two directly paired comparisons are "
    "voyage-3 vs bge-small (17 BEIR corpora, hit@5) and bge-large vs text-embedding-3-small (LOCOMO, "
    "judged accuracy)."
)

STORE = (
    "7 · Store — PostgreSQL + pgvector",
    "One transactional database holds both legs: dense vectors on an HNSW index and Postgres FTS "
    "`tsvector` side by side. No separate vector service to operate. Row-level security per tenant, "
    "`recall forget` for right-to-erasure, registered index generations with a shadow route, durable "
    "ordered outbox and a transactional cutover that refuses while events are pending. Measured at "
    "50,600 chunks: recall@5 1.00 filtered and unfiltered.",
    "standard",
    "pgvector · HNSW",
)

SPLADE = (
    "6b · Learned sparse (SPLADE)",
    "Encodes term weights over the vocabulary into the `recall_sparse_v1` sidecar at index time, and "
    "a query encoder at search time.\n"
    "REFUSED under RECALL_ENV=production: the sidecar has no foreign key to the chunk table and the "
    "erasure paths predate it, so a forgotten chunk would leave partially reconstructable term "
    "weights behind. Wiring erasure is the precondition for lifting the refusal.",
    "optional",
    # NOT "optional" — that word belongs on the arrow, and printing it twice reads as two
    # different claims.
    "sidecar table",
)

QUERY = [
    (
        "8 · Query → query vector",
        "The query is embedded with the profile's QUERY encoder (`query_embed`), not the passage "
        "encoder — the registry makes the two dispatch keys, so an asymmetric profile cannot "
        "silently produce a query vector from the passage side.\n"
        "`search_fused` optionally also embeds the recent conversation turns and fuses both: "
        "+0.0084 nDCG@5 and +0.0842 R@100 WITH a reranker, −0.0447 without — so it refuses to "
        "run unreranked rather than warn.",
        "standard",
        "asymmetric",
    ),
]

LEGS = [
    (
        "Dense",
        "standard",
        "pgvector cosine (`<=>`) over HNSW, top `candidate_k`. Catches paraphrase — the question "
        "worded nothing like the memo.",
    ),
    (
        "Sparse · lexical",
        "standard",
        "Postgres FTS: `websearch_to_tsquery` + `ts_rank`. Catches exact identifiers, error codes and "
        "rare tokens that embeddings smear.",
    ),
    (
        "Sparse · learned",
        "optional",
        "SPLADE weights. REPLACES the lexical leg by default rather than adding to it: a weak extra "
        "leg injects fusion noise at ranks 37–54.",
    ),
]

POST = [
    (
        "10 · Reciprocal Rank Fusion",
        "RRF at k=60 over every enabled leg's `candidate_k` candidates. Each surviving hit is "
        "re-scored to its TRUE dense cosine whichever leg found it, so one single basis reaches the "
        "trust layer — a cosine measured against a different string would silently mean something "
        "else downstream.",
        "standard",
        "k = 60",
    ),
    (
        "11 · Cross-encoder rerank",
        "ms-marco-MiniLM over the WHOLE fused pool before truncating to k — slicing to k first would "
        "hide from the cross-encoder exactly the document reranking exists to rescue. Worth +0.065 "
        "hit@5 on a weak-embedder corpus (n.s., 57× latency: 45 ms → 2,568 ms) and nothing where "
        "dense already scores nDCG@10 0.97. A reranker can only reorder what fusion already "
        "retrieved.",
        "optional",
        "off by default",
    ),
    (
        "12 · Gap warning",
        "If the best DENSE cosine is below the threshold, the result is flagged 'probable corpus gap "
        "— treat as noise' instead of serving nearest-noise as an answer. The published negative "
        "result: the 0.50 default sits at the 0th percentile of five measured distributions and the "
        "16th of the sixth. No fixed constant transfers across embedders, so calibrate against your "
        "own labelled set — do not ship the constant.",
        "standard",
        "fail-closed",
    ),
    (
        "13 · Trust layer — validity beats similarity",
        "A pure post-processing pass, no DB access and no clock read. Bi-temporal: `now` is valid "
        "time (expired / not_yet_valid from the memo's own window), `known_as_of` is transaction time "
        "(not_yet_known from `first_indexed_at`), and supersession edges rewind with it.\n"
        "Verdict precedence: invalid_metadata > not_yet_known > superseded > expired / not_yet_valid "
        "> low_confidence > ok. When a superseded hit scored above threshold — it WOULD have been "
        "the confident answer — its retrieved successor is promoted to `ok` even at a lower cosine, "
        "and hits reorder valid-first. That single decision is the thesis: cos 0.806 stale loses to "
        "cos 0.784 current.",
        "standard",
        "always on",
    ),
    (
        "14 · Entailment judge",
        "A QNLI cross-encoder asked, per verdict-ok hit, 'does this text answer this question?' — "
        "non-answering hits are demoted to `not_entailed`. It exists for the NEAR-MISS, an adjacent "
        "memory that clears any cosine threshold by construction, which the gap check cannot catch by "
        "definition. A decision at the judge's own model-fixed boundary, so there is nothing to "
        "recalibrate per embedder. Cost is one model pass per ok candidate.",
        "optional",
        "off by default",
    ),
    (
        "15 · Abstain, or serve",
        "No hit earned `ok` ⇒ an explicit abstention carrying its reason (corpus gap / superseded / "
        "expired / not entailed), rather than the nearest match. Every result also reports freshness: "
        "how old the newest indexed content is, so a stale index warns instead of silently serving "
        "rot.",
        "standard",
        "“I don't know”",
    ),
    (
        "16 · Evidence construction",
        "Generator-neutral by construction: a fixed system prompt with no interpolation site, every "
        "corpus byte JSON-escaped inside a delimiter whose own angle brackets are escaped, capped by "
        "`max_items` and an EXACT token budget. Citations must resolve to supplied chunk IDs — "
        "structural validation only, it does not claim the cited passage entails the answer. An "
        "abstention bypasses the generator entirely.",
        "standard",
        "no LLM shipped",
    ),
    (
        "17 · Surface",
        "`recall_search` · `recall_evidence` · `recall_index` · `recall_forget` · "
        "`recall_stats` over MCP (stdio), plus the CLI and drop-in LangChain / LlamaIndex retrievers. "
        "Every hit returns verdict + calibrated confidence + provenance + `indexed_at`, alongside "
        "diagnostics naming the exact tenant, generation, pipeline and corpus fingerprint it came "
        "from.",
        "standard",
        "MCP · CLI · SDK",
    ),
]

CALIBRATION = (
    "Calibration — fitted, not constant",
    "The confidence curve and the abstention threshold are fitted per (embedder, corpus) from a "
    "small labelled answerable / unanswerable set, and the result is bound to that generation's "
    "fingerprint. An uncalibrated retriever warns rather than pretending; a threshold copied from "
    "another embedder is the failure this whole box exists to prevent.",
    "standard",
    "per embedder",
)


def main() -> None:
    c = Canvas()

    # ---- header ------------------------------------------------------------------------------
    y = 46
    c.text(MAIN_X - 40, y + 18, "RE-call", size=40, fill=INK, weight="700", mono=True)
    c.text(
        MAIN_X + 138,
        y + 18,
        "— the full pipeline, corpus to answer",
        size=20,
        fill=INK_2,
    )
    c.text(
        MAIN_X - 40,
        y + 46,
        "every phase, top to bottom · colour says what a thing costs you, "
        "not how important it is",
        size=13,
        fill=MUTED,
    )
    y += 78

    # ---- legend ------------------------------------------------------------------------------
    leg_h = 84
    c.add(
        f'<rect x="{MAIN_X - 40}" y="{y}" width="{SIDE_X + SIDE_W - MAIN_X + 40}" '
        f'height="{leg_h}" rx="10" fill="{PANEL}" stroke="{PANEL_STROKE}"/>'
    )
    c.text(MAIN_X - 22, y + 24, "LEGEND", size=10.5, fill=MUTED, mono=True)
    col_w = (SIDE_X + SIDE_W - MAIN_X + 40 - 36) / 3
    for i, (kind, label, note) in enumerate(LEGEND):
        cx = MAIN_X - 22 + (i % 3) * col_w
        cy = y + 44 + (i // 3) * 24
        _, stroke, chip = STYLES[kind]
        c.add(
            f'<rect x="{cx:.1f}" y="{cy - 9:.1f}" width="13" height="13" rx="3" fill="{chip}" '
            f'stroke="{stroke}"/>'
        )
        c.text(cx + 21, cy + 2, label, size=11.4, fill=INK_2, weight="600")
        c.text(cx + 21 + len(label) * 6.6 + 8, cy + 2, note, size=10.6, fill=MUTED)
    y += leg_h + 34

    # ---- lane 1: ingest ----------------------------------------------------------------------
    def lane(label: str, note: str, y0: float) -> float:
        c.add(
            f'<line x1="{MAIN_X - 40}" y1="{y0}" x2="{SIDE_X + SIDE_W}" y2="{y0}" '
            f'stroke="{RULE}"/>'
        )
        c.text(MAIN_X - 40, y0 + 26, label, size=13.5, fill=INK, weight="700", mono=True)
        c.text(MAIN_X - 40 + len(label) * 8.6 + 16, y0 + 26, note, size=12, fill=MUTED)
        return y0 + 46

    y = lane("INGEST", "— the write path. No LLM anywhere on it, which is why it is free.", y)

    cx_main = MAIN_X + MAIN_W / 2
    for title, body, kind, tag in INGEST:
        h = draw_cell(c, MAIN_X, y, MAIN_W, title, body, kind, tag=tag)
        arrow(c, cx_main, y + h, y + h + GAP)
        y += h + GAP

    # ---- the embedder cell -------------------------------------------------------------------
    row_gap = 6
    row_text_w = MAIN_W - 2 * PAD - 40
    rows: list[tuple[str, str, str, list[str], float]] = []
    for name, kind, tag, note in EMBEDDERS:
        lines = wrap(note, row_text_w, 11.0)
        rows.append((name, kind, tag, lines, 22 + len(lines) * 15.2 + 12))
    foot_lines = wrap(EMBEDDER_FOOTNOTE, MAIN_W - 2 * PAD - 12, 10.8)
    emb_h = (
        PAD
        + 18
        + 8
        + 34
        + sum(r[4] + row_gap for r in rows)
        + 10
        + len(foot_lines) * 14.6
        + PAD
        - 6
    )

    fill, stroke, chip = STYLES["standard"]
    c.add(
        f'<rect x="{MAIN_X}" y="{y:.1f}" width="{MAIN_W}" height="{emb_h:.1f}" rx="10" '
        f'fill="#0f1620" stroke="{CARD_STROKE}"/>'
    )
    c.add(f'<rect x="{MAIN_X + 1}" y="{y + 9:.1f}" width="4" height="{emb_h - 18:.1f}" rx="2" fill="{chip}"/>')
    c.text(MAIN_X + PAD + 6, y + PAD + 12, "6 · Embedding", size=TITLE_SIZE, fill=INK, weight="700")
    c.text(
        MAIN_X + MAIN_W - PAD,
        y + PAD + 11,
        "pluggable · one Embedder protocol",
        size=10.2,
        fill=chip,
        mono=True,
        anchor="end",
    )
    c.text(
        MAIN_X + PAD + 6,
        y + PAD + 34,
        "Every input that can change a stored vector — model, dimension, query/passage mode, "
        "normalisation, chunker, context — is one declared identity.",
        size=11.4,
        fill=INK_3,
    )

    ry = y + PAD + 50
    for name, kind, tag, lines, rh in rows:
        rfill, rstroke, rchip = STYLES[kind]
        c.add(
            f'<rect x="{MAIN_X + PAD - 6:.1f}" y="{ry:.1f}" width="{MAIN_W - 2 * PAD + 12:.1f}" '
            f'height="{rh:.1f}" rx="7" fill="{rfill}" stroke="{rstroke}" stroke-opacity="0.7"/>'
        )
        c.add(
            f'<circle cx="{MAIN_X + PAD + 10:.1f}" cy="{ry + 15:.1f}" r="4.6" fill="{rchip}"/>'
        )
        c.text(MAIN_X + PAD + 24, ry + 19, name, size=12.6, fill=INK, weight="700", mono=True)
        c.text(
            MAIN_X + MAIN_W - PAD - 6,
            ry + 19,
            tag,
            size=10.2,
            fill=rchip,
            mono=True,
            anchor="end",
        )
        ty = ry + 36
        for line in lines:
            c.text(MAIN_X + PAD + 24, ty, line, size=11.0, fill=INK_3)
            ty += 15.2
        ry += rh + row_gap

    c.add(
        f'<line x1="{MAIN_X + PAD:.1f}" y1="{ry + 2:.1f}" x2="{MAIN_X + MAIN_W - PAD:.1f}" '
        f'y2="{ry + 2:.1f}" stroke="{PANEL_STROKE}"/>'
    )
    fy = ry + 18
    for line in foot_lines:
        c.text(MAIN_X + PAD + 6, fy, line, size=10.8, fill=MUTED, style="italic")
        fy += 14.6

    # SPLADE hangs off the embedding stage
    sp_h = cell_height(SPLADE[0], SPLADE[1], SIDE_W, SPLADE[3])
    sp_y = y + 40
    draw_cell(c, SIDE_X, sp_y, SIDE_W, SPLADE[0], SPLADE[1], SPLADE[2], tag=SPLADE[3])
    elbow(
        c,
        MAIN_X + MAIN_W,
        y + 90,
        SIDE_X,
        sp_y + 46,
        label="optional",
        colour=STYLES["optional"][2],
    )

    y += emb_h
    arrow(c, cx_main, y, y + GAP)
    y += GAP

    # SPLADE also writes to the store
    elbow(
        c,
        SIDE_X,
        sp_y + sp_h - 20,
        MAIN_X + MAIN_W,
        y + 40,
        label="",
        colour=STYLES["optional"][2],
        head="left",
    )

    # ---- store -------------------------------------------------------------------------------
    h = draw_cell(c, MAIN_X, y, MAIN_W, STORE[0], STORE[1], STORE[2], tag=STORE[3])
    y += h + GAP + 14

    # ---- lane 2: query -----------------------------------------------------------------------
    y = lane(
        "QUERY",
        "— the read path. Postgres does the searching; the guards decide what may be served.",
        y,
    )

    for title, body, kind, tag in QUERY:
        h = draw_cell(c, MAIN_X, y, MAIN_W, title, body, kind, tag=tag)
        arrow(c, cx_main, y + h, y + h + GAP)
        y += h + GAP

    # ---- three legs, side by side ------------------------------------------------------------
    leg_w = (MAIN_W - 2 * 14) / 3
    leg_bodies = [wrap(note, leg_w - 2 * 14, 11.0) for _, _, note in LEGS]
    legs_h = 20 + 16 + 8 + max(len(b) for b in leg_bodies) * 15.2 + 16
    c.text(MAIN_X, y - 12, "9 · Retrieval legs — run in parallel", size=12.4, fill=INK_2, weight="700")
    for i, ((name, kind, _), lines) in enumerate(zip(LEGS, leg_bodies)):
        lx = MAIN_X + i * (leg_w + 14)
        lfill, lstroke, lchip = STYLES[kind]
        c.add(
            f'<rect x="{lx:.1f}" y="{y:.1f}" width="{leg_w:.1f}" height="{legs_h:.1f}" rx="9" '
            f'fill="{lfill}" stroke="{lstroke}" stroke-opacity="0.85"/>'
        )
        c.add(f'<rect x="{lx + 1:.1f}" y="{y + 8:.1f}" width="4" height="{legs_h - 16:.1f}" rx="2" fill="{lchip}"/>')
        c.text(lx + 16, y + 26, name, size=12.6, fill=INK, weight="700")
        ty = y + 48
        for line in lines:
            c.text(lx + 16, ty, line, size=11.0, fill=INK_3)
            ty += 15.2
        # The "optional" label rides the ARROW, not the box: what is optional is whether this leg
        # contributes to fusion at all.
        arrow(
            c,
            lx + leg_w / 2,
            y + legs_h,
            y + legs_h + GAP,
            label="optional" if kind == "optional" else "",
            dashed=(kind == "optional"),
            colour=lchip if kind == "optional" else "#484f58",
        )
    y += legs_h + GAP

    # ---- everything after fusion -------------------------------------------------------------
    cal_drawn = False
    for title, body, kind, tag in POST:
        h = draw_cell(c, MAIN_X, y, MAIN_W, title, body, kind, tag=tag)
        if title.startswith("13") and not cal_drawn:
            cal_h = cell_height(CALIBRATION[0], CALIBRATION[1], SIDE_W, CALIBRATION[3])
            cal_y = y + h / 2 - cal_h / 2
            draw_cell(
                c, SIDE_X, cal_y, SIDE_W, CALIBRATION[0], CALIBRATION[1], CALIBRATION[2],
                tag=CALIBRATION[3],
            )
            elbow(
                c,
                SIDE_X,
                cal_y + cal_h / 2,
                MAIN_X + MAIN_W,
                y + h / 2,
                colour=STYLES["standard"][2],
                head="left",
            )
            cal_drawn = True
        if (title, body, kind, tag) != POST[-1]:
            arrow(c, cx_main, y + h, y + h + GAP)
        y += h + GAP

    # ---- footer ------------------------------------------------------------------------------
    y += 4
    c.add(f'<line x1="{MAIN_X - 40}" y1="{y}" x2="{SIDE_X + SIDE_W}" y2="{y}" stroke="{RULE}"/>')
    c.text(
        MAIN_X - 40,
        y + 26,
        "Every number on this diagram is measured and sourced in results/FINDINGS.md; where a cell "
        "carries no number, none was measured, and it says so.",
        size=11.6,
        fill=MUTED,
        style="italic",
    )
    total_h = y + 52

    header = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {total_h:.0f}" '
        f'width="{W}" height="{total_h:.0f}" role="img" aria-labelledby="t d">'
        f"<title id=\"t\">The RE-call pipeline, ingest to answer</title>"
        f"<desc id=\"d\">A top-to-bottom diagram of every phase of RE-call: lint, chunking, "
        f"contextualisation, embedding (with every embedder tested, ranked), an optional learned "
        f"sparse sidecar, the Postgres store, then query embedding, three retrieval legs, RRF "
        f"fusion, optional rerank, gap warning, the trust layer, an optional entailment judge, "
        f"abstention, evidence construction and the MCP surface. Colour encodes what an option "
        f"costs: shipped default, best measured, free and local, cloud egress, opt-in, or "
        f"rejected.</desc>"
        f'<rect width="{W}" height="{total_h:.0f}" rx="18" fill="{BG}" stroke="{CARD_STROKE}"/>'
    )
    print(header + "\n" + "\n".join(c.parts) + "\n</svg>")


if __name__ == "__main__":
    # The diagram is full of arrows, multiplication signs and curly quotes. On a console whose
    # default encoding is cp1252 (Windows) the whole run dies at the first one, having printed
    # nothing, so the encoding is set here rather than left to the environment.
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    main()
