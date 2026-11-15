"""Render the README billboard and its static first card."""

from argparse import ArgumentParser
from functools import lru_cache
import json
from math import cos, exp, pi
from pathlib import Path
from shutil import copyfile

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont


ROOT = Path(__file__).parent
SIZE = (1280, 640)
INK = "#f5f1ed"
GOLD = "#efc78d"
MUTED = "#d9c7c9"
OUTLINE = "#000000"
HOLD_MS = 5000
DISSOLVE_FRAMES = 32
DISSOLVE_FRAME_MS = 50

CARDS = (
    ("Shared memory", "One memory,", "every agent.",
     "Agents can read and write to the same store."),
    ("SQLite & PostgreSQL", "Memory for one.", "Memory for a team.",
     "Start locally with SQLite or share a PostgreSQL store."),
    ("Hybrid RAG", "Two ways to search.", "One ranked list.",
     "Dense and full-text results fused with RRF."),
    ("Atomic rescue", "Find the fact", "inside the note.",
     "An opt-in retrieval path for individual facts."),
    ("Semantic graph", "Follow the links", "between memories.",
     "See which memos link to or replace each other."),
    ("Calibrated abstention", "Say when you", "don’t know.",
     "Abstains below a threshold calibrated on your store."),
    ("Supersession", "Keep the current", "decision in front.",
     "A declared successor outranks the memo it replaces."),
    ("Validity & stale reports", "Know when", "memory is stale.",
     "Flags expired and future-dated notes. Agents can report stale memory."),
    ("Forgetting", "Erase a memory", "from the store.",
     "Forget indexed notes and the fact values they alone support."),
)


@lru_cache
def _font(name: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(Path("C:/Windows/Fonts") / name), size)


def _artwork() -> Image.Image:
    wash = Image.new("RGB", (160, 80))
    colors = []
    for y in range(80):
        for x in range(160):
            rose = exp(-(((x - 113) / 48) ** 2 + ((y - 47) / 34) ** 2))
            ember = exp(-(((x - 129) / 32) ** 2 + ((y - 29) / 29) ** 2))
            colors.append((16 + int(31 * rose + 13 * ember),
                           13 + int(14 * rose + 9 * ember),
                           15 + int(19 * rose + 3 * ember)))
    wash.putdata(colors)
    image = wash.resize(SIZE, Image.Resampling.BICUBIC).convert("RGBA")

    texture = Image.new("RGBA", SIZE)
    draw = ImageDraw.Draw(texture)
    for y in range(78, 565, 17):
        for x in range(698, 1204, 17):
            strength = exp(-(((x - 975) / 225) ** 2 + ((y - 316) / 200) ** 2))
            draw.ellipse((x, y, x + 1, y + 1), fill=(236, 183, 133, int(31 * strength)))

    traces = (
        ((733, 231), (773, 231), (801, 203), (856, 203)),
        ((746, 342), (772, 342), (806, 309), (851, 309)),
        ((779, 452), (803, 452), (839, 416), (883, 416)),
        ((1125, 175), (1161, 175), (1183, 151), (1214, 151)),
        ((1146, 282), (1170, 282), (1195, 256), (1220, 256)),
        ((1103, 402), (1153, 402), (1197, 446), (1215, 446)),
    )
    for points in traces:
        draw.line(points, fill=(227, 167, 104, 68), width=1)
        x, y = points[0] if points[0][0] < 800 else points[-1]
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), outline=(240, 190, 131, 120))
    image.alpha_composite(texture)

    logo = Image.open(ROOT / "assets" / "re_call_logo.png").convert("RGBA")
    logo = logo.crop(logo.getbbox())
    logo.thumbnail((448, 448), Image.Resampling.LANCZOS)
    position = (980 - logo.width // 2, 304 - logo.height // 2)
    glow = Image.new("L", SIZE)
    glow.paste(logo.getchannel("A"), position)
    glow = glow.filter(ImageFilter.GaussianBlur(16))
    glow = glow.point(lambda value: int(value * .20))
    halo = Image.new("RGBA", SIZE, "#dc754f")
    halo.putalpha(glow)
    image.alpha_composite(halo)
    image.alpha_composite(logo, position)

    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((77, 138, 112, 140), radius=1, fill=GOLD)
    draw.rounded_rectangle((78, 379, 189, 380), radius=1, fill="#90736c")
    for index in range(len(CARDS)):
        x = 78 + index * 29
        draw.rounded_rectangle((x, 559, x + 19, 562), radius=2, fill="#58403f")
    return image


def _outlined_text(draw: ImageDraw.ImageDraw, x: int, baseline: int, text: str,
                   font: ImageFont.FreeTypeFont, color: str, stroke: int) -> None:
    left, _, _, _ = draw.textbbox((0, 0), text, font=font, anchor="ls", stroke_width=stroke)
    draw.text((x - left, baseline), text, font=font, fill=color, anchor="ls",
              stroke_width=stroke, stroke_fill=OUTLINE)


def _card(card: tuple[str, str, str, str]) -> Image.Image:
    label, line_one, line_two, subtitle = card
    layer = Image.new("RGBA", (700, 428))
    draw = ImageDraw.Draw(layer)
    _outlined_text(draw, 51, 35, label.upper(), _font("consola.ttf", 20), GOLD, 1)
    _outlined_text(draw, 2, 144, line_one.upper(), _font("segoeuib.ttf", 55), INK, 3)
    _outlined_text(draw, 2, 230, line_two.upper(), _font("georgiai.ttf", 57), "#f0cdd2", 3)

    font = _font("segoeui.ttf", 28)
    lines = [""]
    for word in subtitle.split():
        candidate = f"{lines[-1]} {word}".strip()
        if draw.textlength(candidate, font=font) <= 625:
            lines[-1] = candidate
        else:
            lines.append(word)
    for index, line in enumerate(lines):
        _outlined_text(draw, 2, 320 + 38 * index, line, font, MUTED, 2)
    return layer


def write_banner() -> None:
    base = _artwork()
    stills = []
    for index, card in enumerate(CARDS):
        frame = base.copy()
        frame.alpha_composite(_card(card), (76, 111))
        x = 78 + index * 29
        ImageDraw.Draw(frame).rounded_rectangle((x, 559, x + 19, 562), radius=2, fill=GOLD)
        stills.append(frame.convert("RGB"))
    stills[0].save(ROOT / "banner.png", optimize=True)
    copyfile(ROOT / "banner.png", ROOT / "social_card.png")

    # A shared palette keeps the logo and background from flickering between frames.
    palette = stills[0].quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    background = base.convert("RGB")
    indexed_background = background.quantize(palette=palette)

    def encode(frame: Image.Image) -> Image.Image:
        # Keep the dither in the static background fixed, so the GIF encodes only the copy.
        mask = ImageChops.difference(frame, background).convert("L").point(
            [0] + [255] * 255, mode="1"
        )
        indexed = indexed_background.copy()
        indexed.paste(frame.quantize(palette=palette), mask=mask)
        return indexed

    frames, durations = [], []
    for index, still in enumerate(stills):
        following = (index + 1) % len(stills)
        frames.append(encode(still))
        durations.append(HOLD_MS)
        for step in range(1, DISSOLVE_FRAMES + 1):
            phase = step / (DISSOLVE_FRAMES + 1)
            opacity = (1 + cos(2 * pi * phase)) / 2
            card = still if phase < .5 else stills[following]
            # Fade through the fixed background so two headlines never overlap.
            frames.append(encode(Image.blend(background, card, opacity)))
            durations.append(DISSOLVE_FRAME_MS)
    frames[0].save(ROOT / "banner.gif", save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, optimize=True)


def write_homepage_copy(homepage: Path) -> None:
    rotation = {
        "holdMs": HOLD_MS,
        "dissolveMs": DISSOLVE_FRAMES * DISSOLVE_FRAME_MS,
        "cards": CARDS,
    }
    (homepage / "assets" / "headline-rotation.js").write_text(
        "// Generated by docs/make_readme_banner.py --homepage.\n"
        + "const headlineRotation = " + json.dumps(rotation, ensure_ascii=False, indent=2) + ";\n",
        encoding="utf-8", newline="\n",
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--homepage", type=Path, help="Also export copy and timing into a homepage directory")
    args = parser.parse_args()
    write_banner()
    if args.homepage is not None:
        write_homepage_copy(args.homepage)
