# RE-call desktop runtime

The Windows desktop prototype uses this compose project for a managed local runtime.
The database volume persists between application restarts. The `recall` container stays alive
so the desktop client can start a private stdio MCP process with `docker compose exec`.

The image includes native extraction for PDF, DOCX, XLSX/XLS, PPTX, HTML, CSV, EML, and RTF.
It also includes LibreOffice for legacy `.doc`, `.ppt`, `.odt`, `.ods`, and `.odp` files. Scanned
PDFs need an OCR layer and are reported as empty when no text layer is present.
