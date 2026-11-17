"""Build the RE-call logo variants, README banner, and GitHub social preview.

The README and social previews use one canonical design and transparent logo.
"""

from __future__ import annotations

from pathlib import Path
from shutil import copyfile

from PIL import Image

from make_readme_banner import write_banner

ROOT = Path(__file__).parent
LOGO = ROOT / "assets" / "re_call_logo.png"

def logo_layer(size: int, opacity: int = 220) -> Image.Image:
    logo = Image.open(LOGO).convert("RGBA")
    logo.thumbnail((size, size), Image.Resampling.LANCZOS)
    alpha = logo.getchannel("A").point(lambda value: value * opacity // 255)
    logo.putalpha(alpha)
    return logo


def save(image: Image.Image, path: Path) -> None:
    image.save(path, format="PNG", optimize=True, compress_level=9)


def sync_logos() -> None:
    project = ROOT.parent
    for destination in (
        project / "site/assets/re_call_logo.png",
        project / "recall/desktop/assets/re_call_watermark.png",
    ):
        copyfile(LOGO, destination)

    save(logo_layer(1024, opacity=255), project / "plugin/.claude-plugin/icon.png")
    logo_layer(192, opacity=255).save(
        project / "recall/dashboard/static/recall-logo.webp", format="WEBP", lossless=True,
    )


if __name__ == "__main__":
    sync_logos()
    write_banner()
    print("Updated logo variants, docs/banner.gif, docs/banner.png, and docs/social_card.png.")
