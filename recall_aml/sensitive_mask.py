"""W5: mask sensitive identifiers in returned text, in place, with typed placeholders.

Round two, 2026-09-28. H2 (minimal sensitive-information disclosure) scored 42.16 on the first
Textual Full. Readers leak what they are shown even when told not to (ConfAIde, PrivacyLens: 25 to
57%), so the only place disclosure can be controlled is what memory returns. PersonaMem-v2's
sensitive-information probes (its code, read 2026-09-28) mark as correct the answer that uses the
context "with sensitive information masked out using placeholders", so the item is KEPT, and only
the identifier inside it is replaced: the reader still gets the surrounding context it needs.

Detectors are deterministic and deliberately narrow: a card number must pass the Luhn check, an
SSN must have its dashed shape, a passport number must sit next to the word "passport". Health and
therapy details are never touched: PersonaMem-v2 rewards using them (E3), and masking them would
lower a score to protect nothing the probes ask to protect.

⚠️ Off by default. Whether AML scores H2 with PersonaMem's masked-answer key or a judge that
rewards the raw value is not public; this is measured, and asked of the organisers, before serving.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
import re

from recall_aml.models import SearchItem

_CARD = re.compile(r"(?<![\d-])(?:\d[ -]?){12,18}\d(?![\d-])")
#: Not inside a longer dashed number: "+81-123-45-6789" is a phone.
_SSN = re.compile(r"(?<![\d+-])\d{3}-\d{2}-\d{4}(?![\d-])")
_EMAIL = re.compile(r"(?<![\w.+-])[\w.+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+(?![\w-])")
#: A phone number needs a phone SHAPE: a ``+`` country code, or the US ``(415) 555-0132`` /
#: ``415-555-0132`` / ``415.555.0132`` forms. Bare space-separated digit groups ("order 1234 5678
#: 9012") are not enough, which is the false positive the first pattern had.
_PHONE = re.compile(
    r"(?<![\w+])(?:\+\d{1,3}[ .-]?(?:\(\d{1,4}\)[ .-]?)?\d{1,4}(?:[ .-]\d{2,4}){2,3}"
    r"|\(\d{3}\)\s?\d{3}[ .-]\d{4}|\d{3}[.-]\d{3}[.-]\d{4})(?!\d)"
)
_API_KEY = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|xox[baprs]-[A-Za-z0-9-]{10,})\b"
)
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){3,7}(?: ?[A-Z0-9]{1,3})?\b")
#: The number itself is case-sensitive and must hold a digit: "passport renewal" is not a passport
#: number, and with the whole pattern case-insensitive it fired 142 times on LongMemEval-S.
_PASSPORT = re.compile(
    r"((?i:passport)(?:\s+(?i:number|no\.?|#))?(?:\s+(?i:is|as))?\s*[:#]?\s*)(?=[A-Z]*\d)([A-Z0-9]{6,12})\b"
)
#: A value the text itself NAMES: "Account Number: 83749201", "plate number CXV-2748",
#: "real_id_number=CZ8492037165". PersonaMem-v2's planted identifiers are synthetic (its card numbers
#: mostly fail Luhn, its licence and ID numbers have no fixed shape), so the label is the evidence.
#: The label is case-insensitive; the value is upper-case letters, digits, spaces and dashes, holds a
#: digit, and stops at the first lower-case word, so "license plate UPL-4726) on March" masks
#: exactly the plate.
_LABELLED = re.compile(
    r"((?i:\b(?:(?:credit|debit)[ _]card|card|bank[ _]account|account|routing|iban|"
    r"driver'?s?[ _]licen[cs]e|licen[cs]e(?:[ _]plate)?|vehicle[ _]plate|plate|registration|"
    r"real[ _]id|national[ _]id|id[ _]card|social[ _]security|tax[ _]id)"
    r"(?:[ _](?:number|no\.?|#))?)"
    r"(?:\s+(?i:is|as|of))?\s*[:=#]?\s*[\"'‘“(]?)"
    r"(?=[A-Z0-9 -]*\d)([A-Z0-9](?:[A-Z0-9]|[ -](?=[A-Z0-9])){3,30})(?![A-Za-z0-9])"
)
#: A home directory names its owner, and often their employer or school after it.
_HOME_PATH = re.compile(r"(?i)(?<![\w/])/(?:home|users)/[\w.-]+(?:/[\w.@+-]+)*")
_STREET_WORD = r"(?:[A-Z][a-z]+|N|S|E|W|NE|NW|SE|SW)"
_ADDRESS = re.compile(
    rf"\b\d{{1,5}}\s+(?:{_STREET_WORD}\s+){{1,4}}"
    r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Way|Place|Pl|Terrace)\b\.?"
    r"(?:,\s*(?:Apt|Unit|Suite|Flat)\.?\s*\w+)?"
    r"(?:,\s*[A-Z][A-Za-z]+(?:\s[A-Z][A-Za-z]+)?)?(?:,\s*[A-Z]{2}\s+\d{5}(?:-\d{4})?)?"
)
#: An address the text introduces as one ("physical_address: …", "my address is …", "I reside at
#: …"): masked to the end of its line or sentence, and only when it holds a digit. "address" as a
#: label needs ``:``, ``=`` or ``is`` after it; bare, it is the verb ("could be addressed. …"),
#: which fired 900 times on LongMemEval-S before this was required.
_LABELLED_ADDRESS = re.compile(
    r"((?i:\b(?:(?:(?:physical|home|service|billing|mailing)[ _])?address\b\s*(?:is\b|[:=])"
    r"|i\s+(?:reside|live)\s+at\b))\s*)(?=[^\n\[]{0,60}\d)([^\n\[]{4,120}?)(?=\.\s|\n|$)"
)


def luhn_valid(digits: str) -> bool:
    total = 0
    for index, char in enumerate(reversed(digits)):
        value = int(char)
        if index % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def mask_text(text: str) -> tuple[str, Counter[str]]:
    """``text`` with each detected identifier replaced by its placeholder, and what was masked."""
    counts: Counter[str] = Counter()

    def replace(label: str) -> Callable[[re.Match[str]], str]:
        def inner(match: re.Match[str]) -> str:
            counts[label] += 1
            return f"[{label}]"
        return inner

    def card(match: re.Match[str]) -> str:
        digits = re.sub(r"\D", "", match.group(0))
        if 13 <= len(digits) <= 19 and luhn_valid(digits):
            counts["CARD NUMBER"] += 1
            return "[CARD NUMBER]"
        return match.group(0)

    def labelled(label: str) -> Callable[[re.Match[str]], str]:
        def inner(match: re.Match[str]) -> str:
            counts[label] += 1
            return f"{match.group(1)}[{label}]"
        return inner

    text = _API_KEY.sub(replace("API KEY"), text)
    text = _EMAIL.sub(replace("EMAIL"), text)
    text = _HOME_PATH.sub(replace("FILE PATH"), text)
    text = _SSN.sub(replace("SSN"), text)
    text = _CARD.sub(card, text)
    text = _IBAN.sub(replace("BANK ACCOUNT"), text)
    text = _PASSPORT.sub(labelled("PASSPORT NUMBER"), text)
    text = _LABELLED.sub(labelled("ID NUMBER"), text)
    text = _PHONE.sub(replace("PHONE"), text)
    text = _ADDRESS.sub(replace("STREET ADDRESS"), text)
    text = _LABELLED_ADDRESS.sub(labelled("STREET ADDRESS"), text)
    return text, counts


def masked_items(items: Sequence[SearchItem]) -> tuple[list[SearchItem], Counter[str]]:
    """Every text item with its identifiers masked; other items unchanged; counts for the log."""
    total: Counter[str] = Counter()
    out: list[SearchItem] = []
    for item in items:
        if not isinstance(item.content, str):
            out.append(item)
            continue
        masked, counts = mask_text(item.content)
        total.update(counts)
        out.append(item if masked == item.content else item.model_copy(update={"content": masked}))
    return out, total
