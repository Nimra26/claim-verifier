import io
import shutil
import subprocess
from pathlib import Path

import pytest

from verifier.documents import DocumentError, read_file, read_text, supported
from verifier.manifest import load_manifest

SENTENCE = "Full-time employees get 25 days of annual leave."
needs_soffice = pytest.mark.skipif(not (shutil.which("soffice") or shutil.which("libreoffice")),
                                   reason="LibreOffice is not installed")


def _xlsx() -> bytes:
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.append(["Policy", "Annual leave days"])
    ws.append(["Leave Policy 2025", 25])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _pptx() -> bytes:
    from pptx import Presentation

    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = "Leave Policy 2025"
    slide.placeholders[1].text = SENTENCE
    buf = io.BytesIO()
    prs.save(buf)
    return buf.getvalue()


def _office(tmp_path: Path, target: str) -> bytes:
    """A real file of type `target`, made by LibreOffice from a text file."""
    src = tmp_path / "src.txt"
    src.write_text("Leave Policy 2025\n\n" + SENTENCE + "\n", encoding="utf-8")
    exe = shutil.which("soffice") or shutil.which("libreoffice")
    subprocess.run([exe, f"-env:UserInstallation={(tmp_path / 'lo').as_uri()}", "--headless", "--norestore",
                    "--convert-to", target, "--outdir", str(tmp_path / "made"), str(src)],
                   capture_output=True, timeout=120, check=False)
    out = tmp_path / "made" / f"src.{target}"
    assert out.is_file(), f"LibreOffice could not make a .{target}"
    return out.read_bytes()


def test_text_files_are_read_as_they_are():
    assert read_text("a.md", ("# T\n\n" + SENTENCE).encode()) == "# T\n\n" + SENTENCE
    assert SENTENCE in read_text("a.txt", ("﻿" + SENTENCE).encode("utf-8"))
    assert "25" in read_text("a.csv", b"policy,days\n2025,25\n")
    assert "café" in read_text("a.txt", "café".encode("cp1252"))


def test_excel_and_powerpoint():
    xl = read_text("leave.xlsx", _xlsx())
    assert "Leave Policy 2025" in xl and "25" in xl
    assert SENTENCE in read_text("leave.pptx", _pptx())


@needs_soffice
@pytest.mark.parametrize("ext", ["pdf", "docx", "odt", "doc"])
def test_pdf_word_and_older_formats(tmp_path, ext):
    assert SENTENCE in read_text(f"leave.{ext}", _office(tmp_path, ext))


@pytest.mark.parametrize("name, data, words", [
    ("a.exe", b"MZ", "can't be read"),
    ("a.md", b"   \n", "empty"),
    ("a.docx", b"not really a word file", "a.docx"),
    ("a.pdf", b"%PDF-1.4 broken", "a.pdf"),
])
def test_unreadable_files_say_why(name, data, words):
    with pytest.raises(DocumentError) as e:
        read_text(name, data)
    assert words in str(e.value)


def test_supported_names():
    assert all(supported(n) for n in ["a.PDF", "b.docx", "c.xlsx", "d.pptx", "e.md", "f.txt", "g.odt", "h.doc"])
    assert not any(supported(n) for n in ["a.exe", "b.png", "c"])


def test_manifest_reads_office_files(tmp_path):
    (tmp_path / "slides.pptx").write_bytes(_pptx())
    (tmp_path / "manifest.yaml").write_text("question: q\ndocuments:\n  - file: slides.pptx\n    date: 2025-01-01\n")
    manifest, texts = load_manifest(tmp_path)
    assert manifest.documents[0].id == "slides" and SENTENCE in texts["slides"]
    assert SENTENCE in read_file(tmp_path / "slides.pptx")
