"""Read any supported document as plain text.

.md / .txt / .csv are read as they are. PDF, Word (.docx), Excel (.xlsx) and PowerPoint (.pptx) go through
Ask the Workspace's own reader (src/modules/workspace/extract.py, copied unchanged): pdfplumber for PDFs, markitdown
for Office files, so tables stay tables. Older and open formats (.doc, .rtf, .odt, .xls, .ods, .ppt, .odp) are first
converted by LibreOffice, when it is installed, then read the same way.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

TEXT = {".md", ".txt", ".csv"}
READER = {".pdf", ".docx", ".xlsx", ".pptx"}
CONVERT = {".doc": "docx", ".rtf": "docx", ".odt": "docx",
           ".xls": "xlsx", ".ods": "xlsx",
           ".ppt": "pptx", ".odp": "pptx"}
SUPPORTED = TEXT | READER | set(CONVERT)
MAX_BYTES = 25 * 1024 * 1024
CONVERT_TIMEOUT = 120


class DocumentError(ValueError):
    """The file could not be turned into text; the message says why in plain words."""


def supported(name: str) -> bool:
    return Path(name).suffix.lower() in SUPPORTED


def _soffice() -> str | None:
    return shutil.which("soffice") or shutil.which("libreoffice")


def _convert(name: str, data: bytes, target: str) -> bytes:
    exe = _soffice()
    if exe is None:
        raise DocumentError(f"{name}: this format needs LibreOffice to read it. Install LibreOffice, or save the file "
                            f"as .{target} (or PDF) and add that instead.")
    with tempfile.TemporaryDirectory(prefix="claim-verifier-convert-") as tmp:
        tmpdir = Path(tmp)
        src = tmpdir / ("input" + Path(name).suffix.lower())
        src.write_bytes(data)
        cmd = [exe, f"-env:UserInstallation={(tmpdir / 'profile').as_uri()}", "--headless", "--norestore",
               "--convert-to", target, "--outdir", str(tmpdir / "out"), str(src)]
        try:
            subprocess.run(cmd, capture_output=True, timeout=CONVERT_TIMEOUT, check=False)
        except subprocess.TimeoutExpired:
            raise DocumentError(f"{name}: converting it took too long. Save it as .{target} or PDF and add that.")
        out = tmpdir / "out" / f"input.{target}"
        if not out.is_file():
            raise DocumentError(f"{name}: LibreOffice could not open it (it may be damaged or locked). "
                                f"Save it as .{target} or PDF and add that.")
        return out.read_bytes()


def read_text(name: str, data: bytes) -> str:
    """The document's words. Raises DocumentError (with a sentence for the person) when there are none to read."""
    ext = Path(name).suffix.lower()
    if ext not in SUPPORTED:
        raise DocumentError(f"{name}: this file type can't be read. Use PDF, Word, Excel, PowerPoint, "
                            f"OpenDocument, RTF, Markdown, CSV or plain text.")
    if len(data) > MAX_BYTES:
        raise DocumentError(f"{name}: the file is {len(data) / 1048576:.1f} MB; the limit is "
                            f"{MAX_BYTES // 1048576} MB.")
    if ext in TEXT:
        for enc in ("utf-8-sig", "cp1252"):
            try:
                text = data.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise DocumentError(f"{name}: the text is not in a readable encoding. Save it as UTF-8.")
        if not text.strip():
            raise DocumentError(f"{name}: the file is empty.")
        return text
    if ext in CONVERT:
        target = CONVERT[ext]
        data = _convert(name, data, target)
        name = f"{Path(name).stem}.{target}"

    from src.modules.workspace.extract import extract

    result = extract(name, data, max_mb=MAX_BYTES / 1048576)
    if result.status != "ok":
        raise DocumentError(f"{name}: {result.message}")
    text = (result.text or "").strip()
    if not text:
        raise DocumentError(f"{name}: no text was found in it.")
    return text


def read_file(path: Path) -> str:
    return read_text(path.name, Path(path).read_bytes())
