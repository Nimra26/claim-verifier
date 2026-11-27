"""Compare a finished report with a hand-written answer key.

Usage: python -m verifier.acceptance <folder>   (reads report.json and key.yaml)
"""

import sys
from pathlib import Path

import yaml

from verifier.models import Report


def check(report: Report, key: dict) -> list[str]:
    """Return mismatch lines; an empty list means the report matches the key."""
    lines: list[str] = []
    for failure in report.failures:
        lines.append(f"the run had a failure: {failure}")
    want = key.get("answer_status")
    if want is not None and report.answer.status != want:
        lines.append(f"answer status is '{report.answer.status}', expected '{want}'")
    labels = {v.claim_id: v.label for v in report.verdicts}
    for entry in key.get("claims") or []:
        doc, contains, expected = entry["doc"], entry["contains"], entry["label"]
        matches = [c for c in report.claims
                   if c.doc_id == doc and contains.lower() in c.text.lower()]
        if not matches:
            lines.append(f"no claim from {doc} containing '{contains}'")
            continue
        for c in matches:
            got = labels.get(c.id, "missing")
            if got != expected:
                lines.append(f"{doc}: '{c.text}' is '{got}', expected '{expected}'")
    return lines


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("Usage: python -m verifier.acceptance <folder>")
        return 2
    folder = Path(argv[0])
    try:
        report = Report.model_validate_json((folder / "report.json").read_text(encoding="utf-8"))
        key = yaml.safe_load((folder / "key.yaml").read_text(encoding="utf-8")) or {}
    except FileNotFoundError as e:
        print(f"Missing file: {e.filename}")
        return 2
    lines = check(report, key)
    for line in lines:
        print(line)
    if not lines:
        print("No mismatches.")
    return 1 if lines else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
