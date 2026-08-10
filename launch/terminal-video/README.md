# RE-call 45 second terminal video

This folder contains the launch video storyboard and renderer for a short terminal-first demo.

Render:

```powershell
uv run --with pillow --with imageio --with imageio-ffmpeg python launch\terminal-video\render_terminal_video.py
```

Outputs:

* `launch/terminal-video/out/re-call-terminal-demo-45s.mp4`
* `launch/terminal-video/out/re-call-terminal-demo-preview.gif`

The script renders a synthetic terminal capture instead of screen-recording a shell. That keeps the
timing deterministic, avoids leaking local paths or environment variables, and makes the asset easy
to regenerate after copy changes.
