"""Text whose English words came out of a PDF reversed ("seY", "sititapeH", "sruoh 84 ni htiW" = "Within 48 hours").

Moved here from src/modules/workspace/service.py (18 Nov) so the PDF reader can use it at upload time too
(src/modules/workspace/extract.py: a page read this way is read again another way, or left out and reported). The
answer step still uses it to keep such passages out of the proof (service.py imports it from here).

17 Nov, owner's chat: passages of an Urdu survey form whose English words came out reversed inside the Urdu lines sat
at ~45% similarity to almost any short Urdu or Roman Urdu question, passed the refusal floor and were cited as proof.
A word counts as reversed when it reads as an English word only backwards: a capital at its END after small letters
("seY", "airalaM"), or a common English word spelled backwards ("ni", "rof", "htiw"). A text with REVERSED_MIN such
words and more of them than English words read forwards is broken. Measured on every passage of the Content, Public
demo and Public demo (internal) groups, 17 Nov.
"""

from __future__ import annotations

import re

REVERSED_MIN = 3

# The common English words of service._ENGLISH_WORDS as they were when this moved (18 Nov). Kept apart on purpose: the
# question-language check may grow its list; this check must stay what was measured.
ENGLISH_WORDS = frozenset(
    "the a an is are was were be been what which who whom whose when where why how does do did can could should would "
    "will shall may might must of for in on at by with about from into than and or not no this that these those there "
    "their its it they we you your our my i please tell explain describe list give summarize summarise show find any "
    "all many much more most some each every between after before during has have had".split())
# Reversed English words that are ordinary words read forwards: "woh" ("how" backwards) is Roman Urdu for "that".
_FORWARD_ALSO = frozenset({"woh"})

_LATIN_WORD = re.compile(r"[A-Za-z]{2,}")
_TRAILING_CAPITAL = re.compile(r"[a-z]{2,}[A-Z]")
_REVERSED_ENGLISH = frozenset(w[::-1] for w in ENGLISH_WORDS if len(w) >= 2) - ENGLISH_WORDS - _FORWARD_ALSO


def reversed_words(text: str) -> tuple[int, int]:
    """(words that read as English only backwards, common English words read forwards) in a text."""
    backwards = forwards = 0
    for w in _LATIN_WORD.findall(text or ""):
        low = w.lower()
        # Two capitals are an abbreviation, not a reversed word ("NA" in a table is "not available", not "an").
        if _TRAILING_CAPITAL.fullmatch(w) or (low in _REVERSED_ENGLISH and not (len(w) == 2 and w.isupper())):
            backwards += 1
        elif low in ENGLISH_WORDS:
            forwards += 1
    return backwards, forwards


def broken_extraction(text: str) -> bool:
    """True for a text whose English words came out of the PDF reversed (see REVERSED_MIN)."""
    backwards, forwards = reversed_words(text)
    return backwards >= REVERSED_MIN and backwards > forwards


# --- putting reversed English back the right way round (18 Nov) -----------------------------------------------------
# On a page that is mostly Urdu the PDF reader reads every line right to left (pdf_extractor._read_page, "rtl"), which
# puts Urdu right but turns each run of English and digits inside it backwards: ")REDLO DNA SRAEY 01 SELAM LLA ROF("
# for "(FOR ALL MALES 10 YEARS AND OLDER)", "31." for "13.". Turning each such run around again (brackets with it, as
# they were swapped too) gives the text as printed. Only for text with Arabic-script letters in it: a page with none
# (a mirrored cover) is not an Urdu page read right to left, and turning its runs would garble tables and numbers.

_ARABIC = "؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿"
_RUN = re.compile(rf"[()\[\]{{}}]*[A-Za-z0-9](?:[^\n{_ARABIC}]*[A-Za-z0-9])?[()\[\]{{}}]*")
_HAS_ARABIC = re.compile(f"[{_ARABIC}]")


def turn_reversed_runs(text: str) -> str:
    """text with each run of Latin letters and digits (and what lies between them on the line, outside Urdu) turned
    around; text with no Arabic-script letter is returned as it is."""
    if not _HAS_ARABIC.search(text or ""):
        return text
    return _RUN.sub(lambda m: m.group(0)[::-1], text)


# --- a single label set sideways (18 Nov) ---------------------------------------------------------------------------
# A chart's axis title set at 90 degrees comes out of the reader backwards inside otherwise good English: "nerdlihC%"
# (ASER 2023, 49 passages), "natsihcolaB", "bajnuP", "tnemlornE". Only such a word that is printed the right way round
# somewhere else in the same document is turned back ("Children" is there): the meaning is then certain. Others stay.

_SIDEWAYS = re.compile(r"(?<![A-Za-z])[a-z]{2,}[A-Z](?![A-Za-z])")


def turn_sideways_words(text: str, known: set[str]) -> str:
    """text with each word that reads backwards ("nerdlihC") turned around when its turned form (lower case) is in
    known, the words the document prints the right way round."""
    return _SIDEWAYS.sub(lambda m: m.group(0)[::-1] if m.group(0)[::-1].lower() in known else m.group(0), text or "")


def known_words(texts) -> set[str]:
    """The words of these texts (three letters or more, lower case) as written, for turn_sideways_words."""
    return {w.lower() for t in texts for w in re.findall(r"[A-Za-z]{3,}", t or "")
            if not _TRAILING_CAPITAL.fullmatch(w)}
