"""The hardware answers the desktop configuration page reads, without importing Qt.

Moved out of `recall.desktop.ui` (desktop UI split). `recall.desktop.ui` re-imports every name, so
`recall.desktop.ui.<name>` keeps resolving.
"""

from __future__ import annotations

from typing import Any


def _splade_hardware_probe() -> Any:
    """The two answers the configuration page's SPLADE check reads, and nothing else.

    `probe_system` also runs docker, wsl and systeminfo one after another (up to about 88 s),
    which kept the SPLADE checkbox disabled while it waited for answers the page never looks at.
    """
    from types import SimpleNamespace

    from recall.wizard.probe import probe_cuda_vram, probe_hardware

    return SimpleNamespace(hardware=probe_hardware(), cuda_vram_bytes=probe_cuda_vram())
