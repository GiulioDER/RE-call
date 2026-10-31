"""DOC-001: the copyable SDK script must compile, without executing its imports."""

import html
from pathlib import Path
import re


def test_copyable_sdk_example_compiles_as_a_script():
    """Red on 76a0b36ea: the extracted snippet contains top-level async for."""
    page = (Path(__file__).resolve().parents[1] / "site/integrations.html").read_text()
    section = page.split('<h2 id="agent-sdk">', 1)[1].split('<h2 id="mcp">', 1)[0]
    code = re.findall(r"<pre>(.*?)</pre>", section, re.S)[1]
    source = html.unescape(re.sub(r"<[^>]+>", "", code))
    error = None
    try:
        compile(source, "copied-agent-sdk.py", "exec")
    except SyntaxError as exc:
        error = str(exc)
    assert error is None, error
