"""Emits docs/mtrag_abstention.html, the source of docs/mtrag_abstention.png.

The scene: every MTRAG system plotted as answer quality against correct refusals. The
point of the picture is the shape of the field, not RE-call's dot. The systems that
answer best are the ones that almost never refuse, and the llama family makes that
monotone at fixed architecture.

Palette, type and the once-per-image rule are inherited from site/styles.css, so this
reads as the same object as the README banner and the setup site.

Build:
    1. python docs/make_mtrag_abstention.py
    2. chrome --headless=new --disable-gpu --hide-scrollbars --force-device-scale-factor=2 \
         --window-size=1600,900 --screenshot=docs/_mtrag_2x.png docs/mtrag_abstention.html
    3. python -c "from PIL import Image; Image.open('docs/_mtrag_2x.png').convert('RGB') \
         .resize((1600,900), Image.LANCZOS).save('docs/mtrag_abstention.png', optimize=True)"

Step 2 needs network: Geist is loaded from Google Fonts, the same way site/index.html
loads it. Without network the render silently falls back to Segoe UI and the type will
not match the site, so check the output rather than trusting the exit code.

Step 3 is not optional, for the reason given in docs/banner.html: shipping the 2x render
is a 4 MB file, and downsampling from 2x is what makes the type crisp and the grain
compressible.

⚠️ Do not draw a fitted trend line on this chart. Least squares over the nine baselines
gives r = -0.549, permutation p = 0.107 (n = 9), which does not support a line. The llama
family callout carries the same claim as fact instead of as a fit.
"""

from __future__ import annotations

import pathlib
import urllib.request

HERE = pathlib.Path(__file__).parent

# ---------------------------------------------------------------------------
# Data. docs/MTRAG_BENCHMARK.md section 5, recomputed per system from RAG.json.
# The human reference row (48/55, harmonic 0.7947) is excluded: it is not a system,
# and plotting it would flatten every real difference on both axes.
# (label, end-to-end harmonic, correct refusals of 55, anchor, dx, dy, kind)
# ---------------------------------------------------------------------------
POINTS = [
    ("llama-3.1-8b",     0.4710, 18, "start",  17,   5, "llama"),
    ("RE-call",          0.5527, 16, "start",  21,   6, "hero"),
    ("llama-3.1-70b",    0.5378, 16, "end",   -17,   6, "llama"),
    ("gpt-4o-mini",      0.5437, 13, "end",   -17,   5, "field"),
    ("c4ai-command-r+",  0.5502, 11, "start",  17,   5, "field"),
    ("gpt-4o",           0.5591,  7, "start",  17,   5, "field"),
    ("qwen-2.5-7b",      0.5447,  3, "end",   -17,   5, "field"),
    ("llama-3.1-405b",   0.5691,  3, "end",   -17,  -9, "llama"),
    ("qwen-2.5-72b",     0.5625,  1, "start",  15,   5, "field"),
    ("mixtral-8x22b",    0.5230,  0, "start",  16,   5, "field"),
]

TOTAL_UNANSWERABLE = 55

# Geometry -------------------------------------------------------------------
W, H = 1600, 900
PL, PR = 232, 1436          # plot left / right in px
PT, PB = 364, 726           # plot top / bottom in px
XMIN, XMAX = 0.4625, 0.5775
YMIN, YMAX = 0.0, 35.0


def px(harmonic: float) -> float:
    return PL + (harmonic - XMIN) / (XMAX - XMIN) * (PR - PL)


def py(pct: float) -> float:
    return PB - (pct - YMIN) / (YMAX - YMIN) * (PB - PT)


# Colours, from site/styles.css ---------------------------------------------
GROUND = "#04060c"
INK = "#f4f9ff"
INK_SOFT = "#93aac8"
INK_MUTED = "#5b6f8c"
HERO = "#22d3ee"            # the identity cyan, carrying one mark only
WARN = "#f5a524"            # the one signal colour, on the llama family
FIELD = "#5b6f8c"           # deliberately recessive: the field is context, not identity

COLOR = {"hero": HERO, "llama": WARN, "field": FIELD}

# The Glama rating badge, fetched live so it cannot drift from the real score.
BADGE_URL = "https://glama.ai/mcp/servers/GiulioDER/RE-call/badges/score.svg"
BADGE_CACHE = HERE / "glama_score_badge.svg"


def glama_badge() -> str:
    """Return the badge SVG, refreshing the cached copy when the network allows.

    Glama answers 403 to a request with no User-Agent, so send one. The cached copy
    is committed: it is the last known rating, and it keeps an offline rebuild
    producing the same image rather than a hole where the badge was.
    """
    try:
        request = urllib.request.Request(BADGE_URL, headers={"User-Agent": "RE-call-docs"})
        with urllib.request.urlopen(request, timeout=20) as response:
            svg = response.read().decode("utf-8")
        BADGE_CACHE.write_text(svg, encoding="utf-8", newline="\n")
    except OSError as exc:
        print(f"warning: could not refresh the Glama badge ({exc}), using the cached copy")
        svg = BADGE_CACHE.read_text(encoding="utf-8")
    return svg.replace('width="110" height="20"', 'width="176" height="32"', 1)


# ---------------------------------------------------------------------------
# Chart body
# ---------------------------------------------------------------------------
def chart() -> str:
    parts: list[str] = []

    # Y gridlines and ticks, every 10 percentage points.
    for v in (0, 10, 20, 30):
        y = py(v)
        alpha = 0.16 if v == 0 else 0.075
        parts.append(
            f'<line x1="{PL}" y1="{y:.1f}" x2="{PR}" y2="{y:.1f}" '
            f'stroke="rgba(147,170,200,{alpha})" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{PL - 22}" y="{y + 6:.1f}" text-anchor="end" font-size="17" '
            f'font-weight="500" fill="{INK_MUTED}" letter-spacing="0.01em">{v}%</text>'
        )

    # X ticks.
    for v in (0.47, 0.49, 0.51, 0.53, 0.55, 0.57):
        x = px(v)
        parts.append(
            f'<line x1="{x:.1f}" y1="{PB}" x2="{x:.1f}" y2="{PB + 9}" '
            f'stroke="rgba(147,170,200,0.22)" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{PB + 32}" text-anchor="middle" font-size="16" '
            f'font-weight="500" fill="{INK_MUTED}">{v:.2f}</text>'
        )

    # The marks. Field first, so the hero always sits on top of any overlap.
    order = sorted(POINTS, key=lambda p: {"field": 0, "llama": 1, "hero": 2}[p[6]])
    for label, harmonic, hits, anchor, dx, dy, kind in order:
        pct = hits / TOTAL_UNANSWERABLE * 100
        x, y = px(harmonic), py(pct)

        if kind == "hero":
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="26" fill="{HERO}" opacity="0.13"/>')
            parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="9.5" fill="{HERO}" '
                f'stroke="{GROUND}" stroke-width="3"/>'
            )
            weight, size, text_fill = 700, 21, INK
        elif kind == "llama":
            parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6.5" fill="{GROUND}" '
                f'stroke="{WARN}" stroke-width="2.4"/>'
            )
            weight, size, text_fill = 500, 16, "#c8b48a"
        else:
            parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{GROUND}" '
                f'stroke="{FIELD}" stroke-width="2"/>'
            )
            weight, size, text_fill = 500, 16, INK_SOFT

        # Every point is direct-labelled, so no identity rests on colour alone.
        parts.append(
            f'<text x="{x + dx:.1f}" y="{y + dy:.1f}" text-anchor="{anchor}" '
            f'font-size="{size}" font-weight="{weight}" fill="{text_fill}" '
            f'letter-spacing="-0.005em">{label}</text>'
        )
        if kind == "hero":
            parts.append(
                f'<text x="{x + dx:.1f}" y="{y + dy + 25:.1f}" text-anchor="{anchor}" '
                f'font-size="16" font-weight="500" fill="{HERO}" opacity="0.92">'
                f'29.1%, second of ten</text>'
            )

    # The llama family callout, placed in the empty lower left of the plot.
    cx0, cy0 = PL + 26, PT + 168
    parts.append(
        f'<rect x="{cx0}" y="{cy0}" width="298" height="152" rx="12" '
        f'fill="rgba(245,165,36,0.045)" stroke="rgba(245,165,36,0.24)" stroke-width="1"/>'
    )
    parts.append(
        f'<text x="{cx0 + 22}" y="{cy0 + 34}" font-size="15" font-weight="700" '
        f'fill="{WARN}" letter-spacing="0.09em">SAME FAMILY, MORE CAPABLE</text>'
    )
    family = (("llama-3.1-8b", "32.7%"), ("llama-3.1-70b", "29.1%"), ("llama-3.1-405b", "5.5%"))
    for i, (name, value) in enumerate(family):
        row_y = cy0 + 66 + i * 30
        parts.append(
            f'<text x="{cx0 + 22}" y="{row_y}" font-size="17" font-weight="500" '
            f'fill="{INK_SOFT}">{name}</text>'
        )
        parts.append(
            f'<text x="{cx0 + 276}" y="{row_y}" text-anchor="end" font-size="17" '
            f'font-weight="700" fill="{WARN if i == 2 else "#c8b48a"}">{value}</text>'
        )

    return "\n    ".join(parts)


HTML = f"""<!doctype html>
<!--
  Source of docs/mtrag_abstention.png. Generated by docs/make_mtrag_abstention.py,
  which holds the build steps and the data provenance. Do not hand edit: rerun the
  generator, or the next rebuild silently reverts your change.
-->
<html><head><meta charset="utf-8">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Geist:wght@300..900&display=swap">
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{width:{W}px;height:{H}px;overflow:hidden;background:{GROUND};
  font-family:'Geist',system-ui,'Segoe UI',sans-serif;color:{INK};
  -webkit-font-smoothing:antialiased;font-synthesis-weight:none}}
.stage{{position:relative;width:{W}px;height:{H}px;isolation:isolate}}
/* Atmosphere: two soft wells, teal low right and violet high right, as on the banner. */
.stage::before{{content:"";position:absolute;inset:0;z-index:0;
  background:
    radial-gradient(1100px 620px at 88% 86%, rgba(34,211,238,0.13), transparent 62%),
    radial-gradient(820px 520px at 96% 4%, rgba(139,92,246,0.14), transparent 60%),
    radial-gradient(900px 700px at 4% 0%, rgba(94,234,212,0.045), transparent 62%)}}
/* Film grain. 0.038 is the threshold where it is felt but does not touch legibility. */
.stage::after{{content:"";position:absolute;inset:0;z-index:9;pointer-events:none;opacity:0.038;
  background-image:url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='180' height='180'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='4' stitchTiles='stitch'/></filter><rect width='180' height='180' filter='url(%23n)'/></svg>")}}
.inner{{position:relative;z-index:2;padding:64px 76px 0}}
.top{{display:flex;align-items:flex-start;justify-content:space-between}}
.mark{{font-size:31px;font-weight:700;letter-spacing:-0.035em}}
/* The identity rule, used exactly once. */
.rule{{width:112px;height:3px;margin-top:11px;border-radius:2px;
  background:linear-gradient(90deg,#22d3ee 0%,#5eead4 42%,#8b5cf6 100%)}}
.badge{{display:flex;flex-direction:column;align-items:flex-end;gap:9px}}
.badge .cap{{font-size:12px;font-weight:700;letter-spacing:0.16em;color:{INK_MUTED}}}
h1{{margin-top:40px;font-size:53px;font-weight:700;letter-spacing:-0.032em;
  line-height:1.09;max-width:1180px}}
h1 em{{font-style:normal;color:{HERO}}}
.sub{{margin-top:16px;font-size:19.5px;font-weight:400;color:{INK_SOFT};max-width:1000px;
  letter-spacing:-0.004em;line-height:1.45}}
.axis-y{{position:absolute;left:0;top:0;transform-origin:0 0;
  transform:translate(78px,{PB}px) rotate(-90deg);
  font-size:12.5px;font-weight:700;letter-spacing:0.17em;color:{INK_MUTED};white-space:nowrap}}
.axis-x{{position:absolute;left:{PL}px;top:{PB + 54}px;
  font-size:12.5px;font-weight:700;letter-spacing:0.17em;color:{INK_MUTED}}}
.foot{{position:absolute;left:76px;right:76px;bottom:34px;z-index:3;
  display:flex;align-items:flex-end;justify-content:space-between;gap:44px}}
.foot .note{{font-size:14px;font-weight:400;color:{INK_MUTED};max-width:900px;line-height:1.5}}
.foot .url{{font-size:15px;font-weight:500;color:{INK_SOFT};white-space:nowrap}}
svg text{{font-family:'Geist',system-ui,'Segoe UI',sans-serif}}
</style></head>
<body>
<div class="stage">
  <div class="inner">
    <div class="top">
      <div>
        <div class="mark">RE-call</div>
        <div class="rule"></div>
      </div>
      <div class="badge">
        <div class="cap">MCP DIRECTORY RATING</div>
        {glama_badge()}
      </div>
    </div>
    <h1>The better a system scores,<br>the less it will <em>admit it does not know</em>.</h1>
    <div class="sub">IBM MTRAG, 842 human multi-turn tasks. Of 55 unanswerable ones, this is how
      often each system correctly refused, judged by the benchmark's own scorer.</div>
  </div>

  <svg width="{W}" height="{H}" style="position:absolute;inset:0;z-index:2" fill="none">
    {chart()}
  </svg>

  <div class="axis-y">CORRECT REFUSALS, OF 55 UNANSWERABLE</div>
  <div class="axis-x">END TO END ANSWER QUALITY, HARMONIC MEAN</div>

  <div class="foot">
    <div class="note">Every baseline recomputed from the MTRAG release through one harness, so
      these are anchored comparisons and not the published leaderboard. On MTRAG the refusal rate
      is set by the generator prompt, not by retrieval.</div>
    <div class="url">github.com/GiulioDER/RE-call</div>
  </div>
</div>
</body></html>
"""

if __name__ == "__main__":
    out = HERE / "mtrag_abstention.html"
    out.write_text(HTML, encoding="utf-8", newline="\n")
    print(f"wrote {out}")
