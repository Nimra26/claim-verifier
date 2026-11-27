"""Live check: one tiny Claude call per output schema. Costs a few cents; not run by the tests.

Usage: uv run python -m verifier.probe_model
VERIFIER_BACKEND=cli (default) uses the local Claude Code login; =api needs ANTHROPIC_API_KEY (never printed).\nModel: VERIFIER_MODEL (default claude-sonnet-5-5).
"""
from __future__ import annotations

import asyncio
import sys

from verifier.config import VerifierSettings
from verifier.models import Claim, DocumentInfo, Passage, Verdict
from verifier.reasoner import make_reasoner

QUESTION = "How many days of annual leave does a full-time employee get?"
TEXT = "# Leave Policy 2025\nIssued 1 March 2025. This replaces the 2023 policy.\nFull-time staff get 25 days of annual leave."
CLAIM = Claim(id="new#1", doc_id="new", text="Full-time staff get 25 days of annual leave.",
              quote="Full-time staff get 25 days of annual leave.", important=True, subject="annual leave days")
OTHER = Claim(id="old#1", doc_id="old", text="Full-time staff get 20 days of annual leave.",
              quote="Full-time staff get 20 days of annual leave.", important=True)
PASSAGE = Passage(id="old:0", doc_id="old", source="Leave Policy 2023",
                  text="Full-time staff get 20 days of annual leave.", score=0.9)


async def main() -> int:
    settings = VerifierSettings()
    try:
        r = make_reasoner(settings.backend, settings.model)
    except RuntimeError as e:
        print(e)
        return 2
    model = settings.model
    print(f"model: {model} via {settings.backend}")
    probes = {
        "DocFacts": lambda: r.doc_facts("Leave Policy 2025", TEXT, {"old": "Leave Policy 2023"}),
        "ExtractedClaims": lambda: r.extract_claims(
            QUESTION, DocumentInfo(id="new", file="new.md", title="Leave Policy 2025"), TEXT),
        "groups": lambda: r.group_claims(QUESTION, [CLAIM, OTHER]),
        "Judgment": lambda: r.judge(CLAIM, [PASSAGE]),
        "reword": lambda: r.reword(CLAIM, [CLAIM.text]),
        "answer": lambda: r.write_answer(
            QUESTION, [CLAIM], [Verdict(claim_id="new#1", label="single_source",
                                        reason="Only Leave Policy 2025 states this.", evidence=[])],
            "Only one document states this."),
    }
    failed = 0
    try:
        for name, call in probes.items():
            try:
                out = await call()
                print(f"OK   {name}: {out!r}")
            except Exception as exc:  # noqa: BLE001 - report every schema
                failed += 1
                print(f"FAIL {name}: {type(exc).__name__}: {str(exc)[:300]}")
                if type(exc).__name__ == "AuthenticationError":
                    print("Stopping: the API key was rejected, so no schema can be checked.")
                    break
    finally:
        await r.aclose()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
