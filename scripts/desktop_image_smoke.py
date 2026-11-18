"""Check a built Desktop image (`docker/desktop/Dockerfile`) from inside it.

CI builds that image and runs this script in it, as the image's own user, mounted read-only:

    docker run --rm -v "$PWD/scripts/desktop_image_smoke.py:/smoke.py:ro" <image> python /smoke.py

A green build proves only that `pip install` finished. The desktop client then depends on four
more things, and each step here fails the way the client would:

1. The runtime is the fixed unprivileged user (uid 10001), because the image parses untrusted
   uploads with LibreOffice and the shared uploads volume is owned by that uid.
2. `python -m recall_mcp.server` and `recall schema apply` are what `recall/desktop/runtime.py`
   execs in the container, so the server module must import and the `recall` script must start.
3. `docker-compose.desktop.yml` selects `RECALL_EMBEDDER=fastembed`, so fastembed and onnxruntime
   must import. No model is downloaded.
4. An uploaded `.odt` goes through `recall.extraction.extract_document` to LibreOffice and its
   text must come back, from the uploads directory the user must be able to write.

Exit 0 when every step passed, 1 naming the first step that failed.

Red proof, recorded 2026-11-17 in GitHub Actions runs 44181236703 and 44182035657, against images
built from 7a9af4e55. Each line is one mutation of the image or of its runtime, and what failed;
the unmodified image passed all four steps in both runs.

- `USER recall` defeated with `docker run --user 0`: step 1, "runs as uid 0 gid 0, expected
  10001:10001".
- `/usr/lib/libreoffice` hidden under a tmpfs: step 4, DocumentExtractionError ".ODT needs
  LibreOffice for extraction".
- `/opt/recall/uploads` replaced by a root-owned tmpfs: step 4, PermissionError on
  `/opt/recall/uploads/smoke.odt`.
- `fastembed` dropped from the Dockerfile's `pip install` extras: step 3, ModuleNotFoundError
  "No module named 'fastembed'".
- Base image `python:3.15-slim` (Dependabot #1345): the build itself fails, "No matching
  distribution found for onnxruntime<2,>=1.17".
"""

from __future__ import annotations

import os
import subprocess
import sys
import zipfile
from io import BytesIO
from pathlib import Path

UPLOADS = Path("/opt/recall/uploads")
MARKER = "desktop image smoke marker 7f3a"


def _odt(text: str) -> bytes:
    # The smallest package LibreOffice opens: the `mimetype` entry must come first and be stored
    # uncompressed, as the OpenDocument packaging rules require.
    content = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
        'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" office:version="1.3">'
        f"<office:body><office:text><text:p>{text}</text:p></office:text></office:body>"
        "</office:document-content>"
    )
    manifest = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<manifest:manifest xmlns:manifest="urn:oasis:names:tc:opendocument:xmlns:manifest:1.0" '
        'manifest:version="1.3">'
        '<manifest:file-entry manifest:full-path="/" '
        'manifest:media-type="application/vnd.oasis.opendocument.text"/>'
        '<manifest:file-entry manifest:full-path="content.xml" manifest:media-type="text/xml"/>'
        "</manifest:manifest>"
    )
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as package:
        package.writestr("mimetype", "application/vnd.oasis.opendocument.text", zipfile.ZIP_STORED)
        package.writestr("META-INF/manifest.xml", manifest, zipfile.ZIP_DEFLATED)
        package.writestr("content.xml", content, zipfile.ZIP_DEFLATED)
    return buffer.getvalue()


def _check_user() -> None:
    if os.getuid() != 10001 or os.getgid() != 10001:
        raise AssertionError(f"runs as uid {os.getuid()} gid {os.getgid()}, expected 10001:10001")


def _check_entry_points() -> None:
    import recall_mcp.server  # noqa: F401

    # stderr stays on the console: a CalledProcessError says only the exit status.
    subprocess.run(["recall", "--help"], check=True, stdout=subprocess.DEVNULL, timeout=120)


def _check_embedder() -> None:
    import fastembed
    import onnxruntime

    print(f"  fastembed {fastembed.__version__}, onnxruntime {onnxruntime.__version__}")


def _check_libreoffice_extraction() -> None:
    from recall.extraction import extract_document

    upload = UPLOADS / "smoke.odt"
    upload.write_bytes(_odt(MARKER))
    try:
        document = extract_document(upload, upload.read_bytes())
    finally:
        upload.unlink()
    if MARKER not in document.text:
        raise AssertionError(f"extracted text does not contain the marker: {document.text[:200]!r}")


STEPS = (
    ("1 unprivileged user", _check_user),
    ("2 desktop entry points", _check_entry_points),
    ("3 fastembed embedder", _check_embedder),
    ("4 LibreOffice extraction from uploads", _check_libreoffice_extraction),
)


def main() -> int:
    print(f"Python {sys.version.split()[0]}")
    for name, step in STEPS:
        try:
            step()
        except Exception as exc:
            print(f"FAIL step {name}: {type(exc).__name__}: {exc}")
            return 1
        print(f"ok   step {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
