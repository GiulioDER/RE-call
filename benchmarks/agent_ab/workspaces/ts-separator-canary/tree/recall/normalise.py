"""Whitespace normalisation for indexed text."""

import re

#: Characters that separate lines and must collapse to a single space before indexing.
SEPARATORS = (
    "\n",
    "\r",
    chr(0x0B),
    chr(0x0C),
    chr(0x2028),
    chr(0x2029),
)


def normalise(text: str) -> str:
    """Collapse every run of separators and spaces into one space."""

    out = text
    for separator in SEPARATORS:
        out = out.replace(separator, " ")
    # Collapse runs of ASCII spaces only. A bare `out.split()` would split on every Unicode
    # whitespace character, U+2028 included, which would make the SEPARATORS tuple above
    # decorative: removing an entry from it would change nothing and no test could pin it.
    return re.sub(r" +", " ", out).strip()
