"""Turn an uploaded file into text the workspace can index, or say plainly why it can't.

extract(filename, data) never raises. It returns an Extraction whose status is one of:
  ok           text was found; `segments` holds it as (page or slide number, or None, text) pieces
  empty        the file has no words in it
  scanned      a PDF whose pages are pictures (a scan): nothing to read until OCR is added
  too_big      over WORKSPACE_UPLOAD_MAX_MB (default 25), or a PDF with more pages than RAG_PDF_MAX_PAGES
  unsupported  not one of .pdf .docx .xlsx .pptx .txt .md
  error        the file could not be opened (damaged, password-protected, or a reader isn't installed)
`message` is one plain-English sentence for the person who uploaded it.

PDFs are read with pdfplumber (the same reader as /rag/ingest/pdf), so every passage keeps its page number for the
proof under an answer. Word, Excel and PowerPoint go through Microsoft's markitdown, which keeps tables as markdown
tables; slides keep their slide numbers. Excel tables are then rewritten one row per line, each row carrying its
section and column labels ("PAKISTAN · Out of School Children (5-16): TOTAL 25373381; TOTAL FEMALE 13407075"), so a
passage of numbers still says what they are. Word and PowerPoint tables are left as markitdown wrote them.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import re
import sys
from dataclasses import asdict, dataclass, field

ALLOWED_TYPES = {
    ".pdf": "PDF",
    ".docx": "Word",
    ".xlsx": "Excel",
    ".pptx": "PowerPoint",
    ".txt": "Text",
    ".md": "Markdown",
}
OFFICE = {".docx", ".xlsx", ".pptx"}
OLD_OFFICE = {".doc": ".docx", ".xls": ".xlsx", ".ppt": ".pptx"}
DEFAULT_MAX_MB = 25
# A PDF page with fewer readable characters than this counts as a picture (a scanned page, a photo).
MIN_PAGE_CHARS = 25
# When fewer than this share of a PDF's pages have text, the whole file is reported as a scan.
MIN_TEXT_PAGE_SHARE = 0.5

_SLIDE_MARK = re.compile(r"<!--\s*Slide number:\s*(\d+)\s*-->", re.IGNORECASE)
_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_WORDS = re.compile(r"[\W_]+", re.UNICODE)

log = logging.getLogger(__name__)


@dataclass
class Extraction:
    status: str
    kind: str
    message: str
    text: str = ""
    segments: list[tuple[int | None, str]] = field(default_factory=list)
    pages: int | None = None
    warnings: list[str] = field(default_factory=list)
    # PDF pages left out because their text came out of the reader reversed and no other way read them cleanly.
    unreadable_pages: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def max_upload_mb() -> float:
    """WORKSPACE_UPLOAD_MAX_MB, default 25. A bad value falls back to the default rather than breaking uploads."""
    try:
        value = float(os.environ.get("WORKSPACE_UPLOAD_MAX_MB", DEFAULT_MAX_MB))
        return value if value > 0 else DEFAULT_MAX_MB
    except ValueError:
        return DEFAULT_MAX_MB


def extension_of(filename: str) -> str:
    return os.path.splitext(clean_name(filename))[1].lower()


def kind_of(filename: str) -> str:
    return ALLOWED_TYPES.get(extension_of(filename), "Other")


def clean_name(filename: str | None) -> str:
    """The file's own name: browsers may send a path (C:\\fakepath\\x.pdf); control characters are dropped."""
    name = re.split(r"[\\/]", filename or "")[-1]
    name = "".join(ch for ch in name if ch.isprintable()).strip()
    stem, ext = os.path.splitext(name)
    ext = ext[:20]
    return (stem[:200 - len(ext)] + ext) if name else "unnamed"  # trim the name, never the extension


def slug_for(filename: str) -> str:
    """Lower-case letters and digits joined by '-', as scripts/workspace_setup.py names files. A name with no Latin
    letters or digits (an Urdu file name) gets a stable short id from the name instead, so re-uploading it still
    finds the same document."""
    stem = os.path.splitext(clean_name(filename))[0]
    slug = re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-")[:120].strip("-")
    if not slug:
        slug = "file-" + hashlib.sha256(stem.encode("utf-8")).hexdigest()[:10]
    return slug


def has_words(text: str) -> bool:
    return len(_WORDS.sub("", text or "")) >= 3


# 20 Nov (documents scenario): every refusal says what to do, not only what is wrong.
EMPTY_FILE = "This file is empty. Check that you chose the right file, then add it again."
NO_PAGES = "This PDF has no pages. Open it, save a fresh copy (or print it to PDF), and try again."


def too_big_message(size: int, limit_mb: float) -> str:
    return (f"This file is {size / 1048576:.1f} MB; the limit is {limit_mb:g} MB. Split it into smaller files, "
            "or save it again with smaller pictures.")


def extract(filename: str, data: bytes, *, max_mb: float | None = None, max_pages: int | None = None,
            on_page=None) -> Extraction:
    """on_page(done, total): a PDF's pages read so far, for the page's progress (other kinds don't report)."""
    name = clean_name(filename)
    ext = extension_of(name)
    kind = ALLOWED_TYPES.get(ext, ext.lstrip(".").upper() or "Other")
    try:
        if ext not in ALLOWED_TYPES:
            if ext in OLD_OFFICE:
                return Extraction("unsupported", kind, f"This is the old {ext} format. Open it, choose Save As "
                                  f"{OLD_OFFICE[ext]}, and upload that copy.")
            allowed = ", ".join(ALLOWED_TYPES)
            return Extraction("unsupported", kind, f"This type of file can't be added. Use one of: {allowed}.")
        limit_mb = max_mb if max_mb is not None else max_upload_mb()
        if len(data) > limit_mb * 1048576:
            return Extraction("too_big", kind, too_big_message(len(data), limit_mb))
        if not data:
            return Extraction("empty", kind, EMPTY_FILE)
        if ext == ".pdf":
            return _pdf(data, kind, max_pages, on_page)
        if ext in OFFICE:
            return _office(data, ext, kind)
        return _plain(data, kind)
    except Exception:  # never let one odd file break the whole upload
        return Extraction("error", kind, "This file couldn't be read. It may be damaged. Open it, save a fresh "
                          "copy, and try again.")


def _done(kind: str, segments: list[tuple[int | None, str]], *, pages: int | None = None,
          warnings: list[str] | None = None, unreadable: list[int] | None = None) -> Extraction:
    # NUL characters (UTF-16 without a byte-order mark, some PDFs) are refused by Postgres.
    segments = [(page, text.replace("\x00", "").strip()) for page, text in segments if text and text.strip()]
    text = "\n\n".join(t for _, t in segments)
    if len(text) > max_chars():
        # One huge file would hold the embedding model (and everyone's searches) for minutes (review, 30 Oct).
        return Extraction("too_big", kind, f"This file has about {len(text):,} characters of text; the limit is "
                          f"{max_chars():,}. Split it into smaller files and upload those.", pages=pages)
    if not has_words(text):
        return Extraction("empty", kind, "No text was found in this file, so there is nothing to search. If it's "
                          "a scan or a photo, add a Word or text version instead.", pages=pages, warnings=warnings or [])
    return Extraction("ok", kind, "Read.", text=text, segments=segments, pages=pages, warnings=warnings or [],
                      unreadable_pages=unreadable or [])


def _plain(data: bytes, kind: str) -> Extraction:
    warnings = []
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        text = data.decode("utf-16")
    else:
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = data.decode("cp1252", errors="replace")
            warnings.append("The file isn't saved as UTF-8, so a few characters may look wrong. "
                            "Save it as UTF-8 and upload it again if they matter.")
    return _done(kind, [(None, text.replace("\r\n", "\n"))], warnings=warnings)


def classify_pdf_pages(chars_per_page: list[int]) -> tuple[str, list[int]]:
    """'scanned' when most pages have (almost) no text, else 'ok'; plus the 1-based pages that look like pictures."""
    blank = [i for i, n in enumerate(chars_per_page, start=1) if n < MIN_PAGE_CHARS]
    if not chars_per_page:
        return "scanned", []
    text_pages = len(chars_per_page) - len(blank)
    return ("scanned" if text_pages / len(chars_per_page) < MIN_TEXT_PAGE_SHARE else "ok"), blank


def _pdf(data: bytes, kind: str, max_pages: int | None, on_page=None) -> Extraction:
    try:
        from src.rag.ingestion.pdf_extractor import PDFPlumberExtractor, _pdfplumber
    except Exception:
        _pdfplumber = None
    if _pdfplumber is None:
        log.error("Upload: the PDF reader (pdfplumber) isn't installed; PDFs can't be read")
        return Extraction("error", kind, "PDFs can't be read right now. Tell your admin.")
    try:  # the same limits as /rag/ingest/pdf
        from src.settings.config import settings

        configured_pages, dedupe = settings.rag_pdf_max_pages, settings.rag_pdf_dedupe_threshold
        paragraph_chars = settings.rag_chunk_size  # text boxes are joined into pieces up to the chunk size
    except Exception:
        configured_pages, dedupe, paragraph_chars = 300, 96, 800
    max_pages = max_pages or configured_pages
    try:
        result = PDFPlumberExtractor(dedupe_threshold=dedupe, paragraph_chars=paragraph_chars)._extract_sync(
            pdf_bytes=data, max_pages=max_pages, on_page=on_page)
    except ValueError as exc:
        found = re.search(r"has (\d+) pages", str(exc))
        if found:
            return Extraction("too_big", kind, f"This PDF has {found.group(1)} pages; the limit is {max_pages}. "
                              "Split it into smaller files.", pages=int(found.group(1)))
        return Extraction("empty", kind, NO_PAGES)
    except Exception as exc:
        described = f"{type(exc).__name__} {exc!r}".lower()  # pdfplumber wraps pdfminer's PDFPasswordIncorrect
        if "password" in described or "encrypt" in described:
            return Extraction("error", kind, "This PDF is locked with a password. Remove the password, save a "
                              "copy, and upload that.")
        return Extraction("error", kind, "This PDF couldn't be opened. It may be damaged. Open it, save a fresh "
                          "copy (or print it to PDF), and try again.")

    chars = [0] * result.pages_total
    # Running headers and footers are left out of the passages but are still text on the page: a page with a title and
    # a running header is not a picture page (4 Nov: ASER 2023 p.47-51, "English Tools", would have been dropped).
    for seg in [*result.segments, *getattr(result, "running_headers", [])]:
        if 1 <= seg.page_number <= result.pages_total:
            chars[seg.page_number - 1] += len(_WORDS.sub("", seg.text))
    verdict, blank = classify_pdf_pages(chars)
    if verdict == "scanned":
        return Extraction("scanned", kind, "This PDF is a scan (pictures of pages), so there is no text to read. "
                          "This app can't read scans yet. If you have the original Word or text file, add that "
                          "instead.", pages=result.pages_total)
    blank_set = set(blank)
    segments = [(seg.page_number, seg.text) for seg in result.segments if seg.page_number not in blank_set]
    segments, unreadable = reread_reversed_pages(data, segments)
    warnings = [w for w in result.warnings if "no extractable text" not in w]
    if blank:
        shown = ", ".join(str(p) for p in blank[:12]) + (" and more" if len(blank) > 12 else "")
        if len(blank) == 1:  # 6 Nov: one page said "1 of 4 pages have ... were left out" (singular/plural)
            warnings.insert(0, f"1 of {result.pages_total} pages has no readable text (it may be a picture or a "
                            f"scan) and was left out: page {shown}.")
        else:
            warnings.insert(0, f"{len(blank)} of {result.pages_total} pages have no readable text (they may be "
                            f"pictures or scans) and were left out: pages {shown}.")
    if unreadable:
        warnings.insert(0, unreadable_note(unreadable, result.pages_total))
    return _done(kind, segments, pages=result.pages_total, warnings=warnings, unreadable=unreadable)


# --- pages whose words came out reversed (18 Nov) -------------------------------------------------------------------
# The live index held 110 passages of reversed English ("seY", "sititapeH", "3202 ihagaA-o-meelaT-e-aradI"). On an
# Urdu page the PDF reader now turns such runs back itself (pdf_extractor, reversed_text.turn_reversed_runs). A page
# that still reads reversed (a cover printed mirrored, a PDF whose text is stored backwards) is read a second time with
# PDFium (pypdfium2, already installed with pdfplumber); if that reads cleanly it is used, if not the page is left out
# and the upload says so in plain words, instead of filling search with text no question can match.


def _pdfium_pages_text(data: bytes, page_numbers: list[int]) -> dict[int, str]:
    """These pages' text as PDFium reads them; a page it can't read (or no PDFium at all) is left out."""
    try:
        import pypdfium2
    except ImportError:
        return {}
    out: dict[int, str] = {}
    try:
        pdf = pypdfium2.PdfDocument(data)
    except Exception:
        log.warning("Upload: PDFium could not open the PDF to read pages again")
        return {}
    try:
        for p in page_numbers:
            try:
                text = pdf[p - 1].get_textpage().get_text_bounded() or ""
            except Exception:
                log.warning("Upload: PDFium could not read page %s again", p)
                continue
            out[p] = re.sub(r"[ \t]+\n", "\n", text.replace("\r\n", "\n").replace("\r", "\n")).strip()
    finally:
        pdf.close()
    return out


def reread_reversed_pages(data: bytes, segments: list[tuple[int | None, str]], *,
                          reread=_pdfium_pages_text) -> tuple[list[tuple[int | None, str]], list[int]]:
    """segments with every page whose text reads reversed (reversed_text.broken_extraction) replaced by its text as
    read again by reread(data, pages) -> {page: text}, or left out when that is reversed (or empty) too; plus the
    pages left out. First, a sideways label ("nerdlihC%") whose word the document prints elsewhere is turned back."""
    from src.rag.reversed_text import broken_extraction, known_words, turn_sideways_words

    known = known_words(t for _, t in segments)
    segments = [(page, turn_sideways_words(text, known)) for page, text in segments]
    pages: dict[int, list[str]] = {}
    for page, text in segments:
        if page is not None:
            pages.setdefault(page, []).append(text)
    bad = [p for p, texts in pages.items() if broken_extraction("\n".join(texts))]
    if not bad:
        return segments, []
    again = reread(data, bad)
    replaced = {p: again[p] for p in bad if has_words(again.get(p, "")) and not broken_extraction(again[p])}
    unreadable = sorted(p for p in bad if p not in replaced)
    out: list[tuple[int | None, str]] = []
    written: set[int] = set()
    for page, text in segments:
        if page in replaced:
            if page not in written:  # the page's new text once, in place of all its pieces
                out.append((page, replaced[page]))
                written.add(page)
        elif page not in unreadable:
            out.append((page, text))
    return out, unreadable


def unreadable_note(pages: list[int], total: int) -> str:
    shown = ", ".join(str(p) for p in pages[:12]) + (" and more" if len(pages) > 12 else "")
    if len(pages) == 1:
        return (f"1 of {total} pages could not be read cleanly (its English words came out backwards) and was left "
                f"out of search: page {shown}.")
    return (f"{len(pages)} of {total} pages could not be read cleanly (their English words came out backwards) and "
            f"were left out of search: pages {shown}.")


MAX_UNPACKED_BYTES = 150 * 1024 * 1024  # an office file's contents once unzipped
MAX_RATIO = 100  # unzipped / zipped; real documents sit far below this, zip bombs far above


def _unpacked_size(data: bytes) -> int | None:
    """Total size of an office file's contents once unzipped, from the zip's own index; None when it isn't a zip."""
    import zipfile

    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            return sum(info.file_size for info in z.infolist())
    except (zipfile.BadZipFile, ValueError):
        return None


def max_chars() -> int:
    """WORKSPACE_UPLOAD_MAX_CHARS, default 2,000,000 (about 2,000 passages)."""
    try:
        value = int(os.environ.get("WORKSPACE_UPLOAD_MAX_CHARS", "2000000"))
        return value if value > 0 else 2_000_000
    except ValueError:
        return 2_000_000


def _markitdown_text(data: bytes, ext: str) -> str:
    # The one converter for the type, not MarkItDown's guessing: a damaged .docx must fail, not be read as plain text.
    try:
        from markitdown import StreamInfo
        from markitdown.converters import DocxConverter, PptxConverter, XlsxConverter
    except ImportError:  # markitdown before 0.1
        from markitdown import MarkItDown

        result = MarkItDown().convert_stream(io.BytesIO(data), file_extension=ext)
    else:
        converter = {".docx": DocxConverter, ".xlsx": XlsxConverter, ".pptx": PptxConverter}[ext]()
        result = converter.convert(io.BytesIO(data), StreamInfo(extension=ext))
    return getattr(result, "markdown", None) or getattr(result, "text_content", "") or ""


def _tidy_markdown(text: str) -> str:
    text = _IMAGE.sub(lambda m: m.group(1).strip(), text)  # pictures: keep any alt text, drop the file name / data
    text = _COMMENT.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


_TABLE_LINE = re.compile(r"^\|.*\|\s*$")
_TABLE_DIVIDER = re.compile(r"^\|(\s*:?-{3,}:?\s*\|)+\s*$")
_NOISE_CELL = re.compile(r"nan|unnamed: \d+", re.IGNORECASE)  # pandas' blank cell and blank header
_NUMBER = re.compile(r"[-+]?(\d[\d,]*(\.\d+)?|\.\d+)%?")


def _cells(line: str) -> list[str]:
    cells = re.split(r"(?<!\\)\|", line.strip())[1:-1]
    cells = [re.sub(r"\s+", " ", c.replace("\\n", " ").replace("\\|", "|")).strip() for c in cells]
    return ["" if _NOISE_CELL.fullmatch(c) else c for c in cells]


def _continues_header(head: list[str], row: list[str]) -> bool:
    """A second header row only under a group header spread over blank cells ("TOTAL", "", "", "RURAL"): the row
    fills one of those blanks ("MALE"). Any other row without numbers is a row of data (review 3 Nov: rows with no
    score yet were taken as headers and vanished)."""
    return any(not head[j] and any(head[1:j]) and row[j] for j in range(2, len(row)))


def _sheet_lines(table: list[str]) -> list[str]:
    """One markdown table from an Excel sheet as lines that each carry their labels (3 Nov, public eval Q5: a row of
    bare numbers, chunked away from its headers, matched no question). Rows with one filled cell are text: a title
    before the header, a section ("PAKISTAN") after it, carried into the rows below. The first row without numbers is
    the header; a row under it is a second header row only when it fills a group header's blank cells
    (_continues_header), the group header filling the blank cells after it; a 1, 2, 3... row is dropped. Every other
    row is a data row and is written out."""
    rows = [r for r in (_cells(line) for line in table if not _TABLE_DIVIDER.match(line)) if any(r)]
    if not rows:
        return []
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    kinds, heads = [], []
    for r in rows:
        filled = [c for c in r if c]
        if not any(r[1:]):
            kinds.append("text")
        elif "data" in kinds:
            kinds.append("data")
        elif len(filled) >= 3 and filled == [str(i) for i in range(1, len(filled) + 1)]:
            kinds.append("count")
        elif not any(_NUMBER.fullmatch(c) for c in r[1:]) and (not heads or _continues_header(heads[-1], r)):
            kinds.append("head")
            heads.append(r)
        else:
            kinds.append("data")
    data = [r for r, k in zip(rows, kinds) if k == "data"]
    if not heads or not any(_NUMBER.fullmatch(c) for r in data for c in r[1:]):
        return [" · ".join(c for c in r if c) for r in rows]  # nothing to label: each row's words, no table markup
    for r in heads[:-1]:  # forward-fill group headers (not the last header row, as pandas does)
        for j in range(2, width):
            r[j] = r[j] or r[j - 1]
    names = []
    for j in range(width):
        parts = [h[j] for h in heads if h[j]]
        names.append(" ".join(p for k, p in enumerate(parts) if k == 0 or p != parts[k - 1]))
    nlab = 0  # leading columns that name the row rather than hold a value
    while nlab < width - 1 and not any(_NUMBER.fullmatch(r[nlab]) for r in data):
        nlab += 1
    out, section, carry = [], "", [""] * nlab
    for i, (r, kind) in enumerate(zip(rows, kinds)):
        if kind == "text":
            out.append(r[0])
            if "head" in kinds[:i]:  # after the header it's a section; before it, the table's title
                section = r[0]
        elif kind == "data":
            labels = r[:nlab]
            filled = [i for i, c in enumerate(labels) if c]
            # A sub-row ("", "Of which: Loans") keeps its parent's label ("Total Bilateral ODA").
            carry = [labels[i] or (carry[i] if filled and i < filled[-1] else "") for i in range(nlab)]
            label = " · ".join(c for c in [section, *carry] if c)
            pairs = "; ".join(f"{names[j]} {r[j]}".strip() for j in range(nlab, width) if r[j])
            out.append(f"{label}: {pairs}" if label and pairs else label or pairs)
    return out


def _sheet_tables(text: str) -> str:
    """Excel only: every markdown table in markitdown's text rewritten by _sheet_lines; all other lines as they were."""
    out, table = [], []
    for line in text.split("\n") + [""]:
        if _TABLE_LINE.match(line):
            table.append(line)
            continue
        if table:
            out.extend(_sheet_lines(table))
            table = []
        out.append(line)
    return "\n".join(out).strip()


# Windows' "the module or one of its DLLs was not found" (126), "the procedure was not found" (127, an older DLL of
# the same name was loaded first) and "not a valid Win32 application" (193): a missing or wrong runtime part, not a
# damaged file.
_DLL_WINERRORS = {126, 127, 193}


def _reader_missing(exc: BaseException | None) -> BaseException | None:
    """The part of exc that says the reader itself can't load (not the file): an ImportError (a package, or a native
    file in it such as onnxruntime's or lxml's, "DLL load failed"), markitdown's MissingDependencyException (it
    keeps a converter's ImportError until a file of that type comes) or a Windows DLL OSError. None otherwise."""
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, ImportError) or type(exc).__name__ == "MissingDependencyException":
            return exc
        if isinstance(exc, OSError) and getattr(exc, "winerror", None) in _DLL_WINERRORS:
            return exc
        exc = exc.__cause__ or exc.__context__
    return None


def _reader_cannot_load(kind: str, exc: BaseException) -> Extraction:
    """A reader that can't start on this computer: the person's file is fine and nothing they do to it helps. The
    client sees one plain line; the log names the real cause for whoever fixes it."""
    cause = _reader_missing(exc) or exc
    chain, seen, e = [], set(), cause  # markitdown's exception hides the real ImportError ("DLL load failed") under it
    while e is not None and id(e) not in seen and len(chain) < 5:
        seen.add(id(e))
        chain.append(f"{type(e).__name__}: {e}")
        e = e.__cause__ or e.__context__
    detail = " <- ".join(chain)
    if isinstance(cause, ModuleNotFoundError) and (cause.name or "").split(".")[0] == "markitdown":
        log.error("Upload: markitdown isn't installed; %s files can't be read (%s)", kind, detail)
    else:
        log.error("Upload: the %s reader could not load on this computer (%s). On Windows this is almost always a "
                  "Visual C++ runtime DLL (msvcp140.dll, vcruntime140*.dll) missing next to python.exe or older than "
                  "the file that needs it: run Setup again (a repair puts the app's copies back), or install the "
                  "Microsoft Visual C++ 2015-2022 Redistributable (x64), which needs admin rights. "
                  "Check: windows/tools/vcruntime_check.py on the install folder", kind, detail)
    if sys.platform == "win32":
        return Extraction("error", kind, f"{kind} files can't be read on this computer yet. Your file is fine: ask "
                          "your admin to run Setup again, which repairs this.")
    return Extraction("error", kind, f"{kind} files can't be read right now. Tell your admin.")


def _office(data: bytes, ext: str, kind: str) -> Extraction:
    try:
        import markitdown  # noqa: F401
    except ImportError as e:  # not installed, or a native part of it (onnxruntime via magika) can't load
        return _reader_cannot_load(kind, e)
    unpacked = _unpacked_size(data)
    if unpacked is None:
        return Extraction("error", kind, f"This {kind} file couldn't be opened. It may be damaged. Open it, save a "
                          "fresh copy, and try again.")
    if unpacked > MAX_UNPACKED_BYTES or unpacked > MAX_RATIO * max(len(data), 1):
        # Office files are zips; one that unpacks to GBs would take the laptop's memory (and the model) down.
        return Extraction("too_big", kind, f"This {kind} file unpacks to far more data than its size suggests. "
                          "Split it into smaller files, or save a fresh copy, and try again.")
    try:
        raw = _markitdown_text(data, ext)
    except Exception as e:
        if _reader_missing(e) is not None:  # the converter's own reader (lxml, openpyxl, pptx) can't load: not the file
            return _reader_cannot_load(kind, e)
        return Extraction("error", kind, f"This {kind} file couldn't be opened. It may be damaged or locked with a "
                          "password. Open it, save a fresh copy, and try again.")
    if ext == ".pptx":
        parts = _SLIDE_MARK.split(raw)
        if len(parts) > 1:  # [before, number, slide text, number, slide text, ...]
            slides = [(int(parts[i]), _tidy_markdown(parts[i + 1])) for i in range(1, len(parts) - 1, 2)]
            return _done(kind, slides, pages=len(slides))
    text = _tidy_markdown(raw)
    return _done(kind, [(None, _sheet_tables(text) if ext == ".xlsx" else text)])
