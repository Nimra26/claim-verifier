"""What a log line may say about people's text and errors (rule: user ids, never emails; no question, answer,
document or file text in the app log).

- text_ref(): a question or query as its length and a short hash, so two lines about the same question can be matched
  without the words being in the log.
- exc_brief(): an error as its kind only. A database error's text carries the statement's values (a saved chat's
  question and answer, the admin emails a check looked up, a personal file's name in its doc id), and an SMTP error's
  text carries the address it refused; the kind ("ProgrammingError(UndefinedTableError)") is enough to tell a missing
  table from a database that is down.
"""

from __future__ import annotations

import hashlib

HASH_CHARS = 10


def text_ref(text: object) -> str:
    """'<n> chars, sha256 <prefix>' for a piece of text; '0 chars' for None or empty."""
    s = "" if text is None else str(text)
    if not s:
        return "0 chars"
    return f"{len(s)} chars, sha256 {hashlib.sha256(s.encode('utf-8', 'replace')).hexdigest()[:HASH_CHARS]}"


def exc_brief(exc: BaseException) -> str:
    """The error's kind, plus the kinds of the driver errors it wraps (SQLAlchemy keeps them in .orig): never its text."""
    names = [type(exc).__name__]
    inner, seen = getattr(exc, "orig", None), {id(exc)}
    while inner is not None and id(inner) not in seen and len(names) < 4:
        seen.add(id(inner))
        names.append(type(inner).__name__)
        inner = getattr(inner, "orig", None) or inner.__cause__
    return names[0] + (f"({', '.join(names[1:])})" if len(names) > 1 else "")
