# RE-call 45 second terminal video

This folder contains the launch video storyboard and renderer for a short terminal-first demo.

## Render

The GIF needs only Pillow. The MP4 additionally needs `imageio` and `imageio-ffmpeg`.

```bash
python -m pip install --user pillow imageio imageio-ffmpeg
python launch/terminal-video/render_terminal_video.py
```

PowerShell:

```powershell
python -m pip install --user pillow imageio imageio-ffmpeg
python launch\terminal-video\render_terminal_video.py
```

Both forms assume the repository root as the working directory. The script writes its output beside
itself rather than beside you, so an absolute path works from anywhere:

```bash
python /path/to/RE-call/launch/terminal-video/render_terminal_video.py
```

## Outputs

* `launch/terminal-video/out/re-call-terminal-demo-45s.mp4`
* `launch/terminal-video/out/re-call-terminal-demo-preview.gif`

The GIF is written first, so a machine that cannot produce the MP4 still gets the preview. When the
MP4 fails, the script says which file it did not write, why, and how to fix it, then exits non-zero.

## When ffmpeg is unavailable

Pass `--gif-only` to skip the MP4 deliberately:

```bash
python launch/terminal-video/render_terminal_video.py --gif-only
```

## If you see a numpy DLL error

On Windows, `uv run --with imageio ...` can fail with a traceback ending in:

> `ImportError: DLL load failed while importing _multiarray_umath: An Application Control policy has blocked this file.`

The packages are fine. Their location is not. `uv run` builds an ephemeral environment under
`AppData\Local\uv\cache`, and Smart App Control and similar policies refuse to load native
extensions from that path. Install into your normal interpreter with the `pip` line above and run
the script with plain `python`. The traceback names numpy because numpy is the first native
extension imported, not because numpy is at fault.

## Why it is synthetic

The script renders a synthetic terminal capture instead of screen-recording a shell. That keeps the
timing deterministic, avoids leaking local paths or environment variables, and makes the asset easy
to regenerate after copy changes.
