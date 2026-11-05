"""Credential shapes every registered redactor must hide (M1).

CCA's `secret-exposure` class holds 31 confirmed RE-call findings, 20 of them P1 (measured
2026-11-04), and the most frequent shape is a redactor that missed one form: a libpq keyword DSN,
a password fragment in a driver error, a URL query credential, an env-style or bearer value, the
tail of a quoted value. Each redactor had been tested against the forms its author thought of.
This corpus is the one list, so a form learned once is checked against every redactor that
promises its kind (`tests/test_secret_redactors.py`).

A shape names its kind, the text, and the secrets in it. A redactor passes a shape when no
`WINDOW`-character run of any secret survives, because a redactor that hides a password's head
and prints its tail has still disclosed it.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Long enough for the 16-character floors bearer and token rules use, with digits in it.
SECRET = "S3cretPw9xQ7mZ2kL8vN"
WINDOW = 6

DSN_URL = f"postgresql://alice:{SECRET}@db.example:5432/recall"
DSN_KEYWORD = f"host=db.example user=alice password={SECRET} dbname=recall"


@dataclass(frozen=True)
class Shape:
    kind: str
    text: str
    secrets: tuple[str, ...] = (SECRET,)
    #: The DSN a `(text, dsn)` redactor is told about, for the shapes that carry one.
    dsn: str = ""


#: Kinds: `dsn` is a whole connection string; `dsn-in-text` is text holding one, or a fragment
#: libpq quotes from one, with the DSN known to the redactor; `assignment` is `NAME=value` or a
#: JSON field; `header` is an authorization header; `url` is a credential in a URL's query or
#: userinfo; `argv` is a launch command, one item per argument.
SHAPES: dict[str, Shape] = {
    "dsn-url": Shape("dsn", DSN_URL, dsn=DSN_URL),
    "dsn-url-encoded-password": Shape(
        "dsn",
        "postgresql://alice:S3cretPw9%40Q7mZ2kL8@db.example/recall",
        secrets=("S3cretPw9%40Q7mZ2kL8", "S3cretPw9@Q7mZ2kL8"),
        dsn="postgresql://alice:S3cretPw9%40Q7mZ2kL8@db.example/recall",
    ),
    "dsn-keyword": Shape("dsn", DSN_KEYWORD, dsn=DSN_KEYWORD),
    "dsn-keyword-quoted": Shape(
        "dsn",
        "host=db.example user=alice password='S3cret Pw9xQ7mZ2kL8vN' dbname=recall",
        secrets=("S3cret Pw9xQ7mZ2kL8vN",),
        dsn="host=db.example user=alice password='S3cret Pw9xQ7mZ2kL8vN' dbname=recall",
    ),
    "libpq-quotes-the-password": Shape(
        "dsn-in-text", f'missing "=" after "{SECRET}" in connection info string', dsn=DSN_KEYWORD
    ),
    "error-holding-a-url-dsn": Shape(
        "dsn-in-text", f"could not connect with {DSN_URL}: timeout", dsn=DSN_URL
    ),
    "log-holding-a-keyword-dsn": Shape(
        "dsn-in-text", f"registered recall with {DSN_KEYWORD}", dsn=DSN_KEYWORD
    ),
    "env-password": Shape("assignment", f"DB_PASSWORD={SECRET}"),
    "env-api-key": Shape("assignment", f"OPENAI_API_KEY={SECRET}"),
    "pgpassword": Shape("assignment", f"PGPASSWORD={SECRET} psql -h db.example"),
    "json-password": Shape("assignment", f'{{"password": "{SECRET}"}}'),
    "bearer-header": Shape("header", f"Authorization: Bearer {SECRET}"),
    "url-query-api-key": Shape("url", f"https://api.example.com/v1/x?api_key={SECRET}"),
    "url-query-access-token": Shape("url", f"https://h.example/cb?access_token={SECRET}&state=1"),
    "url-userinfo-token": Shape("url", f"https://{SECRET}@github.com/org/repo.git"),
    "argv-token-flag": Shape("argv", f"recall-mcp --auth-token {SECRET}"),
    "argv-token-equals": Shape("argv", f"recall-mcp --access-token={SECRET}"),
    "argv-bearer-header": Shape("argv", f"curl -H 'Authorization: Bearer {SECRET}' https://h.example"),
    "argv-pgpassword": Shape("argv", f"env PGPASSWORD={SECRET} psql -h db.example"),
    "argv-dsn": Shape("argv", f"psql {DSN_URL}"),
}


def leaked(output: str, shape: Shape) -> list[str]:
    """The runs of `shape`'s secrets that survive in `output`, case-insensitively."""
    folded = output.casefold()
    found = []
    for secret in shape.secrets:
        for start in range(len(secret) - WINDOW + 1):
            run = secret[start : start + WINDOW]
            if run.casefold() in folded:
                found.append(run)
    return found
