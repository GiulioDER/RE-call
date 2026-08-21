"""Entry point for the frozen graphical installer.

A separate module rather than pointing PyInstaller at `recall/desktop/main.py`, for one reason that
only shows up once the thing is frozen: **a frozen process must call
`multiprocessing.freeze_support()` before anything else can spawn one**, or a child process
re-executes the bundle from the top and the application starts again, recursively, until Windows
runs out of handles. Nothing in recall spawns a process today; `onnxruntime` and `tokenizers` both
can. The call is a no-op everywhere it is not needed, and the failure it prevents is a fork bomb
wearing the installer's icon.
"""

from __future__ import annotations

import multiprocessing
import sys


def main() -> int:
    multiprocessing.freeze_support()
    from recall.desktop.main import install_main

    return install_main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
