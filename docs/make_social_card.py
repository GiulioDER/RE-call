"""Emits docs/social_card.html, the source of docs/social_card.png.

This is the GitHub social preview: the card that renders whenever the repo is linked
in Discord, Slack or a tweet. It is a different job from docs/mtrag_abstention.png.
That figure is read at full size and can carry a ten point scatter; this one is often
seen 400px wide inside an embed, so it carries one claim, four bars and nothing else.

The four bars are the honest cut of the same table. RE-call is second, not first, and
the system that tops the benchmark on answer quality sits at the bottom on refusals.
Showing only RE-call against gpt-4o would read better and would be selective.

Data and the Glama badge are imported from make_mtrag_abstention rather than restated,
so the two assets cannot disagree. That import is safe precisely because that module
does all its work behind main().

Build:
    1. python docs/make_social_card.py
    2. chrome --headless=new --disable-gpu --hide-scrollbars --force-device-scale-factor=2 \
         --window-size=1280,640 --screenshot=docs/_social_2x.png docs/social_card.html
    3. python -c "import os; from PIL import Image; Image.open('docs/_social_2x.png') \
         .convert('RGB').resize((1280,640), Image.LANCZOS) \
         .save('docs/social_card.png', optimize=True); os.unlink('docs/_social_2x.png')"

    GitHub wants exactly 1280x640 and under 1 MB for a social preview, and step 3 is
    what keeps it under the limit. Upload the result at
    Settings > General > Social preview. It is a stored upload: committing this file
    does not change what GitHub serves.

    The unlink in step 3 is part of the step. docs/_social_2x.png is a multi-megabyte
    intermediate; it is covered by .gitignore now, but a failed step 3 still strands it.

    Step 2 needs network for Geist, as docs/make_mtrag_abstention.py explains.
"""

from __future__ import annotations

import html
import pathlib
import sys

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))

from make_mtrag_abstention import (  # noqa: E402  (path must be set first)
    GROUND,
    HERO,
    INK,
    INK_MUTED,
    INK_SOFT,
    WARN,
    glama_badge,
    label_for,
)

W, H = 1280, 640

# The four systems that carry the argument, top to bottom as plotted.
# `kind` picks the colour: the hero, the warning, or the field.
BARS = (
    ("llama-3.1-8b", "field", ""),
    ("RE-call", "hero", ""),
    ("gpt-4o", "field", ""),
    ("llama-3.1-405b", "warn", "tops the table on answer quality"),
)

BAR_X = 690
BAR_MAX_W = 400
VALUE_RIGHT = 1220          # values share one right-aligned column, clear of the track
ROW_TOP = 246
ROW_STEP = 82
BAR_H = 26
NOTE_DX = 158               # note offset from the label, which the label must not reach
FOOTER_TOP = 560            # where the footer note and URL begin
LABEL_PX_PER_CHAR = 9.8     # Geist 19px medium, close enough to catch a collision

COLOR = {"hero": HERO, "warn": WARN, "field": INK_MUTED}


def percent(name: str) -> float:
    """The plotted rate, parsed back from the shared formatter so both assets agree."""
    return float(label_for(name).rstrip("%"))


def _check_layout() -> float:
    """Refuse to draw a card whose rows do not fit, the way px()/py() refuse a stray mark.

    BARS is meant to be edited, and the canvas is `overflow:hidden`, so a fifth row
    overprints the footer and a sixth vanishes from the PNG with no error at all.
    """
    if not BARS:
        raise SystemExit("BARS is empty; the card needs at least one system")

    last_bottom = ROW_TOP + (len(BARS) - 1) * ROW_STEP + 12 + BAR_H
    if last_bottom > FOOTER_TOP:
        raise SystemExit(
            f"{len(BARS)} bars need {last_bottom}px but the footer starts at {FOOTER_TOP}px; "
            f"drop a row or reduce ROW_STEP"
        )

    for name, _kind, note in BARS:
        if note and len(name) * LABEL_PX_PER_CHAR + 12 > NOTE_DX:
            raise SystemExit(
                f"the note on {name!r} would collide with its label; "
                f"raise NOTE_DX above {NOTE_DX} or shorten the name"
            )

    top = max(percent(name) for name, _, _ in BARS)
    if top <= 0:
        raise SystemExit("every system in BARS refuses 0 of 55; there is nothing to scale against")
    return BAR_MAX_W / top


def bars() -> str:
    parts: list[str] = []
    scale = _check_layout()

    for i, (name, kind, note) in enumerate(BARS):
        value = percent(name)
        width = value * scale
        label_y = ROW_TOP + i * ROW_STEP
        bar_y = label_y + 12
        colour = COLOR[kind]
        hero = kind == "hero"

        parts.append(
            f'<text x="{BAR_X}" y="{label_y}" font-size="19" '
            f'font-weight="{700 if hero else 500}" '
            f'fill="{INK if hero else INK_SOFT}" letter-spacing="-0.01em">'
            f'{html.escape(name, quote=False)}</text>'
        )
        # The note rides on the label line. To the right of the value it would either
        # collide with the value column or run off the canvas.
        if note:
            parts.append(
                f'<text x="{BAR_X + NOTE_DX}" y="{label_y}" font-size="16" font-weight="500" '
                f'fill="{WARN}" opacity="0.85">· {html.escape(note, quote=False)}</text>'
            )
        # Track, so a short bar still reads as a share of the same span.
        parts.append(
            f'<rect x="{BAR_X}" y="{bar_y}" width="{BAR_MAX_W}" height="{BAR_H}" rx="6" '
            f'fill="rgba(147,170,200,0.07)"/>'
        )
        parts.append(
            f'<rect x="{BAR_X}" y="{bar_y}" width="{width:.1f}" height="{BAR_H}" rx="6" '
            f'fill="{colour}" opacity="{1 if hero else 0.72}"/>'
        )
        parts.append(
            f'<text x="{VALUE_RIGHT}" y="{bar_y + 20}" text-anchor="end" font-size="26" '
            f'font-weight="700" fill="{colour if kind != "field" else INK_SOFT}">'
            f'{value:.1f}%</text>'
        )
    return "\n    ".join(parts)


def build() -> str:
    return f"""<!doctype html>
<!--
  Source of docs/social_card.png, the GitHub social preview. Generated by
  docs/make_social_card.py, which holds the build steps. Do not hand edit.
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
.stage::before{{content:"";position:absolute;inset:0;z-index:0;
  background:
    radial-gradient(760px 470px at 84% 88%, rgba(34,211,238,0.15), transparent 62%),
    radial-gradient(620px 400px at 97% 3%, rgba(139,92,246,0.15), transparent 60%),
    radial-gradient(700px 520px at 2% 6%, rgba(94,234,212,0.05), transparent 62%)}}
.stage::after{{content:"";position:absolute;inset:0;z-index:9;pointer-events:none;opacity:0.038;
  background-image:url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='180' height='180'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='4' stitchTiles='stitch'/></filter><rect width='180' height='180' filter='url(%23n)'/></svg>")}}
.inner{{position:relative;z-index:2;padding:56px 60px 0}}
.top{{display:flex;align-items:flex-start;justify-content:space-between}}
.mark{{font-size:30px;font-weight:700;letter-spacing:-0.035em}}
.rule{{width:104px;height:3px;margin-top:10px;border-radius:2px;
  background:linear-gradient(90deg,#22d3ee 0%,#5eead4 42%,#8b5cf6 100%)}}
h1{{margin-top:62px;font-size:51px;font-weight:700;letter-spacing:-0.033em;
  line-height:1.09;max-width:604px}}
h1 em{{font-style:normal;color:{HERO}}}
.sub{{margin-top:22px;font-size:19px;font-weight:400;color:{INK_SOFT};max-width:520px;
  line-height:1.5;letter-spacing:-0.004em}}
.cap{{position:absolute;left:{BAR_X}px;top:196px;z-index:3;
  font-size:12.5px;font-weight:700;letter-spacing:0.15em;color:{INK_MUTED}}}
.foot{{position:absolute;left:60px;right:60px;bottom:34px;z-index:3;
  display:flex;align-items:flex-end;justify-content:space-between;gap:36px}}
.foot .note{{font-size:13.5px;font-weight:400;color:{INK_MUTED};max-width:620px;line-height:1.5}}
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
      {glama_badge()}
    </div>
    <h1>A memory layer that can say <em>it does not know</em>.</h1>
    <div class="sub">Validity-aware retrieval for agent memory. PostgreSQL plus pgvector,
      local by default, no LLM call in the memory layer.</div>
  </div>

  <div class="cap">CORRECT REFUSALS, OF 55 UNANSWERABLE &nbsp;·&nbsp; IBM MTRAG</div>

  <svg width="{W}" height="{H}" style="position:absolute;inset:0;z-index:2" fill="none">
    {bars()}
  </svg>

  <div class="foot">
    <div class="note">Baselines recomputed from the MTRAG release through one harness, so
      these are anchored comparisons and not the published leaderboard.</div>
    <div class="url">github.com/GiulioDER/RE-call</div>
  </div>
</div>
</body></html>
"""


def main() -> None:
    out = HERE / "social_card.html"
    out.write_text(build(), encoding="utf-8", newline="\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
