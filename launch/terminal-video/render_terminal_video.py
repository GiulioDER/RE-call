from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


WIDTH = 1920
HEIGHT = 1080
FPS = 12
SOURCE_DURATION_SECONDS = 90
SPEED_MULTIPLIER = 2
DURATION_SECONDS = SOURCE_DURATION_SECONDS // SPEED_MULTIPLIER
OUT_DIR = Path(__file__).with_name("out")
MP4_PATH = OUT_DIR / "re-call-terminal-demo-45s.mp4"
GIF_PATH = OUT_DIR / "re-call-terminal-demo-preview.gif"

BG = (9, 12, 18)
PANEL = (17, 24, 39)
PANEL_EDGE = (52, 64, 84)
TEXT = (226, 232, 240)
MUTED = (132, 146, 166)
GREEN = (86, 211, 100)
YELLOW = (245, 194, 66)
RED = (255, 106, 106)
BLUE = (105, 183, 255)
PURPLE = (194, 155, 255)
CYAN = (100, 214, 232)


@dataclass(frozen=True)
class Line:
    text: str
    color: tuple[int, int, int] = TEXT


@dataclass(frozen=True)
class Scene:
    start: float
    end: float
    title: str
    subtitle: str
    lines: tuple[Line, ...]


SCENES = (
    Scene(
        0,
        8,
        "Agent memory fails quietly",
        "Nearest neighbor search can make stale facts look authoritative.",
        (
            Line("$ recall demo --why", BLUE),
            Line("Most RAG returns the closest chunk."),
            Line("Agent memory needs a stricter contract.", YELLOW),
            Line("Is this memory current? Where did it come from? Should I answer?"),
        ),
    ),
    Scene(
        8,
        23,
        "Plain vector search chooses the stale memory",
        "The obsolete rule has the higher cosine.",
        (
            Line("$ naive-vector-search \"how many requests per second can a client make?\"", BLUE),
            Line("rank  source              cosine  text", MUTED),
            Line("1     rate_limits_v1.md   0.806   clients are limited to 100 rps", RED),
            Line("2     rate_limits_v2.md   0.784   clients are limited to 250 rps", GREEN),
            Line(""),
            Line("Agent answer: 100 requests per second", RED),
            Line("Problem: v1 was superseded. Similarity did not know.", YELLOW),
        ),
    ),
    Scene(
        23,
        45,
        "RE-call returns trust state",
        "Validity beats similarity when the corpus declares a correction.",
        (
            Line("$ recall search \"how many requests per second can a client make?\"", BLUE),
            Line("[trusted] query='how many requests per second can a client make?'", GREEN),
            Line("  ok          conf=1.00  cos=0.784  rate_limits_v2.md", GREEN),
            Line("  superseded  conf=1.00  cos=0.806  rate_limits_v1.md", YELLOW),
            Line("              successor=rate_limits_v2.md", YELLOW),
            Line(""),
            Line("Safe answer: 250 requests per second", GREEN),
            Line("The stale hit is visible, but it is not trusted.", MUTED),
        ),
    ),
    Scene(
        45,
        61,
        "When memory has no answer, it abstains",
        "No nearest-noise answer gets promoted into the agent.",
        (
            Line("$ recall search \"how do we handle penguins on mars?\"", BLUE),
            Line("[ABSTAIN gap] no hit above the calibrated confidence threshold", YELLOW),
            Line("  best_hit=browser_support.md  conf=0.18  verdict=low_confidence", MUTED),
            Line(""),
            Line("Agent behavior: say I do not know, ask for more evidence.", GREEN),
            Line("This is a return value, not an exception path.", MUTED),
        ),
    ),
    Scene(
        61,
        76,
        "The contract agents can build on",
        "Every result carries policy metadata before it reaches a model.",
        (
            Line("TrustedHit {", PURPLE),
            Line("  verdict:    ok | superseded | expired | low_confidence", TEXT),
            Line("  confidence: calibrated per corpus and embedder", TEXT),
            Line("  provenance: source file, chunk id, tenant, generation", TEXT),
            Line("  validity:   supersedes, valid_from, valid_until", TEXT),
            Line("}", PURPLE),
            Line("PostgreSQL plus pgvector. Local embeddings. MCP, LangChain, LlamaIndex.", CYAN),
        ),
    ),
    Scene(
        76,
        90,
        "Try it locally",
        "Trustworthy memory for AI agents.",
        (
            Line("$ docker compose up -d --wait", BLUE),
            Line("$ pip install \"recall-rag[fastembed]\"", BLUE),
            Line("$ python -m recall.cli setup", BLUE),
            Line("$ recall search \"what did we decide about caching?\"", BLUE),
            Line(""),
            Line("RE-call: verdict, confidence, provenance, validity, or abstain.", GREEN),
            Line("github.com/GiulioDER/RE-call", MUTED),
        ),
    ),
)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/CascadiaMono.ttf",
        "C:/Windows/Fonts/CascadiaMonoPL.ttf",
        "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


TITLE_FONT = font(52, bold=True)
SUBTITLE_FONT = font(30)
BODY_FONT = font(29)
SMALL_FONT = font(22)


def active_scene(t: float) -> Scene:
    for scene in SCENES:
        if scene.start <= t < scene.end:
            return scene
    return SCENES[-1]


def visible_text(scene: Scene, text: str, line_index: int, t: float) -> str:
    elapsed = max(0.0, t - scene.start)
    line_start = 0.4 + line_index * 1.0
    if elapsed < line_start:
        return ""
    reveal = elapsed - line_start
    chars = int(reveal * 44)
    if chars >= len(text):
        return text
    return text[: max(chars, 0)]


def rounded_rectangle(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    radius: int,
    fill: tuple[int, int, int],
    outline: tuple[int, int, int] | None = None,
    width: int = 1,
) -> None:
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def draw_terminal(draw: ImageDraw.ImageDraw) -> None:
    left, top, right, bottom = 120, 210, 1800, 930
    rounded_rectangle(draw, (left, top, right, bottom), 18, PANEL, PANEL_EDGE, 2)
    draw.rectangle((left, top, right, top + 58), fill=(23, 32, 50))
    for i, color in enumerate((RED, YELLOW, GREEN)):
        draw.ellipse((left + 28 + i * 34, top + 20, left + 46 + i * 34, top + 38), fill=color)
    draw.text((left + 150, top + 18), "RE-call terminal demo", font=SMALL_FONT, fill=MUTED)


def frame(t: float) -> Image.Image:
    source_t = t * SPEED_MULTIPLIER
    img = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(img)
    scene = active_scene(source_t)

    draw.text((120, 72), scene.title, font=TITLE_FONT, fill=TEXT)
    draw.text((122, 140), scene.subtitle, font=SUBTITLE_FONT, fill=MUTED)
    draw_terminal(draw)

    x, y = 160, 300
    for idx, line in enumerate(scene.lines):
        text = visible_text(scene, line.text, idx, source_t)
        if not text:
            continue
        draw.text((x, y), text, font=BODY_FONT, fill=line.color)
        y += 50

    progress_w = int((WIDTH - 240) * min(t / DURATION_SECONDS, 1.0))
    draw.rectangle((120, 984, WIDTH - 120, 990), fill=(38, 48, 68))
    draw.rectangle((120, 984, 120 + progress_w, 990), fill=BLUE)
    draw.text((120, 1010), "45s launch cut", font=SMALL_FONT, fill=MUTED)
    draw.text((WIDTH - 324, 1010), "recall-rag 0.9.1", font=SMALL_FONT, fill=MUTED)
    return img


def frames() -> Iterable[Image.Image]:
    total = DURATION_SECONDS * FPS
    for index in range(total):
        yield frame(index / FPS)


def write_mp4() -> None:
    import imageio.v2 as imageio
    import numpy as np

    with imageio.get_writer(
        MP4_PATH,
        fps=FPS,
        codec="libx264",
        quality=8,
        macro_block_size=1,
        ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    ) as writer:
        for img in frames():
            writer.append_data(np.asarray(img))


def write_gif_preview() -> None:
    preview_frames = [frame(i) for i in range(0, DURATION_SECONDS, 2)]
    preview_frames[0].save(
        GIF_PATH,
        save_all=True,
        append_images=preview_frames[1:],
        duration=2000,
        loop=0,
        optimize=True,
    )


MP4_HELP = """
The MP4 needs `imageio` and `imageio-ffmpeg`. The GIF above needed only Pillow.

    python -m pip install --user imageio imageio-ffmpeg

If the error above mentions a numpy DLL and an Application Control policy, the packages are fine
and their location is not: Windows refuses to load native extensions out of the ephemeral
environment `uv run` builds under AppData\\Local\\uv\\cache. Install into your normal interpreter
with the line above and run this script with plain `python` rather than `uv run`.

To skip the MP4 deliberately, pass --gif-only.
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render the RE-call terminal demo. Outputs go beside this script, in out/."
    )
    parser.add_argument(
        "--gif-only",
        action="store_true",
        help="skip the MP4, which needs imageio and ffmpeg. The GIF needs only Pillow.",
    )
    args = parser.parse_args(argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # The GIF goes first on purpose. It needs nothing beyond Pillow, which is already required to
    # draw a single frame, so rendering the MP4 first meant one missing optional dependency threw
    # away an asset that never depended on it.
    write_gif_preview()
    print(f"wrote {GIF_PATH}")

    if args.gif_only:
        return 0

    # Whether an MP4 was already on disk decides what a failure below actually means. Both
    # outputs are tracked in git, and the README tells you to re-run after copy changes, so the
    # dangerous outcome is not a missing file: it is a stale MP4 sitting beside the GIF that was
    # just regenerated, quietly disagreeing with it.
    mp4_existed = MP4_PATH.exists()

    try:
        write_mp4()
    except Exception as exc:
        # Report what was and was not produced. A caller that sees only a traceback cannot tell
        # that half the job succeeded, and the numpy DLL error this usually surfaces names
        # neither the real cause nor the fix.
        #
        # Flush first: stdout is block-buffered when piped, so without this the failure block
        # lands above the "wrote ... gif" line and the reader concludes nothing was produced.
        sys.stdout.flush()
        print(f"did NOT write {MP4_PATH}", file=sys.stderr)
        if mp4_existed:
            print(
                "  an earlier MP4 is still on disk. It predates the GIF written above, so the"
                " two assets may now disagree. Do not commit them as a pair.",
                file=sys.stderr,
            )
        print(f"  {type(exc).__name__}: {exc}", file=sys.stderr)
        print(MP4_HELP, file=sys.stderr)
        return 1

    print(f"wrote {MP4_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
