import datetime as dt

from verifier.text import compare_dates, normalise, quote_in_document


def test_quote_match_ignores_case_space_quotes():
    assert quote_in_document(
        "Staff get  25 “days”",
        '... staff get 25 "days" of leave',
    )


def test_quote_absent():
    assert not quote_in_document("Staff get 30 days", "Staff get 25 days")


def test_empty_quote_false():
    assert not quote_in_document("", "anything at all")
    assert not quote_in_document("   ", "anything at all")


def test_normalise_maps_curly_quotes_and_case():
    assert normalise("  “Hello”   ‘World’ ") == "\"hello\" 'world'"


def test_compare_dates():
    early = dt.date(2025, 1, 1)
    late = dt.date(2026, 1, 1)
    assert compare_dates(late, early) == "a_newer"
    assert compare_dates(early, late) == "b_newer"
    assert compare_dates(late, late) == "same"
    assert compare_dates(None, late) == "unknown"
    assert compare_dates(late, None) == "unknown"
    assert compare_dates(None, None) == "unknown"
