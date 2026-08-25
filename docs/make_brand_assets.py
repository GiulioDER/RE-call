"""Build the RE-call README banner and GitHub social preview.

Both assets intentionally share the desktop UI palette and transparent logo. The
outputs are fixed at 1280 x 640 because that is the size GitHub expects for a
README banner and social preview.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont


ROOT = Path(__file__).parent
LOGO = ROOT / "assets" / "re_call_logo.png"
CANVAS = "#0E100F"
SURFACE = "#141714"
SURFACE_RAISED = "#171A17"
INK = "#F4F1E8"
MUTED = "#B6B7AC"
GOLD = "#D7A52A"
GOLD_BRIGHT = "#F0BE4A"
GREEN = "#63D39E"
LINE = "#465047"
W, H = 1280, 640


def font(size: int, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    if mono:
        path = Path("C:/Windows/Fonts/consola.ttf")
    elif bold:
        path = Path("C:/Windows/Fonts/seguisb.ttf")
    else:
        path = Path("C:/Windows/Fonts/segoeui.ttf")
    return ImageFont.truetype(path, size)


def background() -> Image.Image:
    image = Image.new("RGB", (W, H), CANVAS)
    px = image.load()
    for y in range(H):
        for x in range(W):
            glow = max(0.0, 1.0 - (((x - 930) / 520) ** 2 + ((y - 305) / 420) ** 2))
            warm = max(0.0, 1.0 - (((x - 1080) / 420) ** 2 + ((y - 120) / 250) ** 2))
            px[x, y] = (14 + int(5 * glow), 16 + int(4 * glow), 15 + int(2 * glow))
            if warm > 0:
                r, g, b = px[x, y]
                px[x, y] = (r + int(5 * warm), g + int(3 * warm), b)
    return image


def logo_layer(size: int, opacity: int = 220) -> Image.Image:
    logo = Image.open(LOGO).convert("RGBA")
    logo.thumbnail((size, size), Image.Resampling.LANCZOS)
    alpha = logo.getchannel("A").point(lambda value: value * opacity // 255)
    logo.putalpha(alpha)
    return logo


def add_halo(image: Image.Image, center: tuple[int, int], radius: int) -> None:
    halo = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(halo)
    x, y = center
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(215, 165, 42, 30))
    halo = halo.filter(ImageFilter.GaussianBlur(radius // 2))
    image.alpha_composite(halo)


def banner() -> Image.Image:
    image = background().convert("RGBA")
    draw = ImageDraw.Draw(image)

    draw.text((74, 112), "RE-call", font=font(76, bold=True), fill=INK)
    draw.text((78, 218), "MEMORY THAT KNOWS WHEN NOT TO GUESS", font=font(19, mono=True), fill=GOLD_BRIGHT)
    draw.line((78, 260, 545, 260), fill=LINE, width=2)
    draw.text((78, 292), "Validity-aware retrieval for agents", font=font(30), fill=INK)
    draw.text((78, 338), "PostgreSQL plus pgvector, local by default.", font=font(23), fill=MUTED)
    draw.text((78, 376), "A memory layer for decisions, notes, and project history.", font=font(23), fill=MUTED)

    draw.rounded_rectangle((78, 486, 404, 548), radius=8, fill=SURFACE_RAISED, outline=LINE, width=2)
    draw.text((104, 505), "VALIDITY-AWARE RETRIEVAL", font=font(16, mono=True), fill=GOLD_BRIGHT)
    draw.text((78, 578), "github.com/GiulioDER/RE-call", font=font(16, mono=True), fill=MUTED)

    add_halo(image, (1002, 317), 220)
    logo = logo_layer(445, 235)
    image.alpha_composite(logo, (790, 100))
    return image.convert("RGB")


def social_card() -> Image.Image:
    image = background().convert("RGBA")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, W, 12), fill=GOLD)
    draw.text((54, 46), "RE-call", font=font(40, bold=True), fill=INK)
    draw.text((56, 104), "A MEMORY LAYER FOR AGENTS", font=font(16, mono=True), fill=GOLD_BRIGHT)
    draw.line((56, 135, 300, 135), fill=LINE, width=2)
    draw.text((56, 184), "It can say", font=font(48, bold=True), fill=INK)
    draw.text((56, 244), "it does not know.", font=font(48, bold=True), fill=GOLD_BRIGHT)
    draw.text((58, 326), "Retrieval with calibrated confidence,", font=font(23), fill=MUTED)
    draw.text((58, 364), "explicit gaps, and no forced answer.", font=font(23), fill=MUTED)

    draw.rounded_rectangle((58, 458, 494, 522), radius=8, fill=SURFACE, outline=LINE, width=2)
    draw.ellipse((84, 480, 98, 494), fill=GREEN)
    draw.text((116, 472), "LOCAL BY DEFAULT", font=font(16, mono=True), fill=INK)
    draw.text((58, 579), "github.com/GiulioDER/RE-call", font=font(16, mono=True), fill=MUTED)

    add_halo(image, (1000, 318), 210)
    logo = logo_layer(450, 238)
    image.alpha_composite(logo, (785, 96))
    return image.convert("RGB")


def save(image: Image.Image, path: Path) -> None:
    image.save(path, format="PNG", optimize=True, compress_level=9)


if __name__ == "__main__":
    save(banner(), ROOT / "banner.png")
    save(social_card(), ROOT / "social_card.png")
    print("wrote docs/banner.png and docs/social_card.png")
