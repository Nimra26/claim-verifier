from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel

from verifier.documents import read_file, supported
from verifier.models import DocumentInfo


class Manifest(BaseModel):
    question: str
    documents: list[DocumentInfo]


def _title(text: str, stem: str) -> str:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return stem


def load_manifest(folder: Path) -> tuple[Manifest, dict[str, str]]:
    folder = Path(folder)
    raw = yaml.safe_load((folder / "manifest.yaml").read_text(encoding="utf-8")) or {}
    docs: list[DocumentInfo] = []
    texts: dict[str, str] = {}
    for entry in raw.get("documents", []):
        name = entry["file"]
        path = folder / name
        if not supported(name):
            raise ValueError(f"unsupported file type: {name} (use PDF, Word, Excel, PowerPoint, OpenDocument, RTF, "
                             f"Markdown, CSV or plain text)")
        if not path.is_file():
            raise FileNotFoundError(f"manifest names a file that does not exist: {name}")
        text = read_file(path)
        d = entry.get("date")
        if isinstance(d, str):
            d = date.fromisoformat(d)
        docs.append(DocumentInfo(
            id=path.stem,
            file=name,
            title=_title(text, path.stem),
            date=d,
            date_source="manifest" if d else None,
            replaces=[Path(r).stem for r in entry.get("replaces", [])],
        ))
        texts[path.stem] = text
    return Manifest(question=raw["question"], documents=docs), texts
