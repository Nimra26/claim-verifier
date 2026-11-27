from datetime import date

import pytest

from verifier.manifest import load_manifest


def _write(folder, name, text):
    (folder / name).write_text(text, encoding="utf-8")


def test_loads_question_dates_and_texts(tmp_path):
    _write(tmp_path, "manifest.yaml", (
        "question: How many leave days?\n"
        "documents:\n"
        "  - file: a.md\n"
        "    date: 2023-01-15\n"
        "  - file: b.txt\n"
    ))
    _write(tmp_path, "a.md", "Twenty days.")
    _write(tmp_path, "b.txt", "Thirty days.")
    manifest, texts = load_manifest(tmp_path)
    assert manifest.question == "How many leave days?"
    a, b = manifest.documents
    assert a.id == "a" and a.date == date(2023, 1, 15) and a.date_source == "manifest"
    assert b.id == "b" and b.date is None and b.date_source is None
    assert set(texts) == {"a", "b"}


def test_missing_file_raises(tmp_path):
    _write(tmp_path, "manifest.yaml", "question: q\ndocuments:\n  - file: nope.md\n")
    with pytest.raises(FileNotFoundError, match="nope.md"):
        load_manifest(tmp_path)


def test_title_from_heading(tmp_path):
    _write(tmp_path, "manifest.yaml", "question: q\ndocuments:\n  - file: p.md\n")
    _write(tmp_path, "p.md", "# Leave Policy 2025\n\nBody")
    manifest, _ = load_manifest(tmp_path)
    assert manifest.documents[0].title == "Leave Policy 2025"
