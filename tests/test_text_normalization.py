import pytest

from onet.text_normalization import expand_abbreviations, normalize_text, pluralize_to_singular


@pytest.mark.parametrize("raw, expected", [
    ("RN", "registered nurse"),
    ("IT manager", "information technology manager"),
    ("fix it", "fix it"),               # ambiguous key, lowercase: left alone
    ("MA", "medical assistant"),
    ("teacher, ma", "teacher ma"),
    ("Na'cho", "na'cho"),               # whole tokens only
    ("(CNA)", "(certified nursing assistant)"),
])
def test_abbreviations(raw, expected):
    assert normalize_text(raw) == expected


def test_expansion_runs_once():
    once = expand_abbreviations("HR")
    assert expand_abbreviations(once) == once


@pytest.mark.parametrize("raw, expected", [
    ("  Sr.  Engineer, Q.A.  ", "sr engineer qa"),
    ("U. S. Army", "us army"),
    ("Sales / Marketing", "sales/marketing"),
    ("nurse\u2014manager", "nurse-manager"),
    ("caf\u00e9 owner", "cafe owner"),
])
def test_normalization(raw, expected):
    assert normalize_text(raw) == expected


@pytest.mark.parametrize("raw", [None, float("nan"), "", "   ", "nan"])
def test_blank_placeholder(raw):
    assert normalize_text(raw, keep_empty_as_placeholder=True) == "[empty]"
    assert normalize_text(raw) == ""


@pytest.mark.parametrize("raw, expected", [
    ("registered nurses", "registered nurse"),
    ("chief executives", "chief executive"),
    ("oil and gas", "oil and gas"),
    ("small business", "small business"),
    ("statistics", "statistics"),
])
def test_singular_never_invents_words(raw, expected):
    assert pluralize_to_singular(raw) == expected
