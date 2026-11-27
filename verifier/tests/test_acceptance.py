from pathlib import Path

from verifier.acceptance import check
from verifier.manifest import load_manifest
from verifier.models import Answer, Claim, Report, Verdict

SETS = Path(__file__).resolve().parents[1] / "testsets"


def _report(label="outdated", status="answered") -> Report:
    claim = Claim(id="c1", doc_id="leave-policy-2023",
                  text="A full-time employee gets 20 days of annual leave per year.",
                  quote="20 days of annual leave", important=True)
    return Report(
        question="q", answer=Answer(status=status, text="t", claim_ids=[]),
        documents=[], claims=[claim], checks=[],
        verdicts=[Verdict(claim_id="c1", label=label, reason="r", evidence=[])],
        person_answers={}, dropped_non_factual=0)


KEY = {"answer_status": "answered",
       "claims": [{"doc": "leave-policy-2023", "contains": "20 DAYS", "label": "outdated"}]}


def test_check_passes():
    assert check(_report(), KEY) == []


def test_check_reports_mismatch():
    lines = check(_report(label="verified", status="not_enough_evidence"), KEY)
    assert len(lines) == 2
    assert any("answer status" in x for x in lines)
    assert any("'verified', expected 'outdated'" in x for x in lines)


def test_check_reports_missing_claim():
    key = {"claims": [{"doc": "leave-faq", "contains": "wellness", "label": "single_source"}]}
    assert check(_report(), key) == ["no claim from leave-faq containing 'wellness'"]


def test_testset_manifests_load():
    for name in ("leave-policy", "no-answer"):
        manifest, texts = load_manifest(SETS / name)
        ids = {d.id for d in manifest.documents}
        assert len(ids) == 5 and set(texts) == ids
        assert all(d.date is not None for d in manifest.documents)
        for d in manifest.documents:
            assert set(d.replaces) <= ids
    manifest, _ = load_manifest(SETS / "leave-policy")
    by_id = {d.id: d for d in manifest.documents}
    assert by_id["leave-policy-2025"].replaces == ["leave-policy-2023"]
    assert by_id["sick-leave-memo-2022"].replaces == ["leave-policy-2025"]
    assert by_id["leave-policy-2025"].title == "Leave Policy 2025"


def test_check_reports_failures():
    # I1: a run with model failures never passes the acceptance check.
    r = _report()
    r.failures = ["Could not read claims from Leave FAQ: BadRequestError"]
    lines = check(r, KEY)
    assert any("Could not read claims from Leave FAQ" in x for x in lines)
