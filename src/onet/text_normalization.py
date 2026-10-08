"""Text normalization shared by the survey answers and the O*NET titles."""

import re
import string
import unicodedata
from functools import lru_cache

import inflect
import pandas as pd
import yaml

from onet.config import CONFIG_DIR

_INFLECT = inflect.engine()

# Keys that are also common words or US state codes. They are expanded only
# when the respondent wrote them in uppercase ("IT manager", not "fix it").
AMBIGUOUS_ABBREV_KEYS = {"it", "do", "na", "ma", "pa", "md", "ot", "pt", "ba"}

# Endings that look plural but are not ("business", "status", "physics").
_PROTECTED_SINGULAR_SUFFIXES = ("ss", "us", "is", "sis", "ous", "ics", "ness")

# Latin-1 readings of UTF-8 bytes found in survey exports.
_MOJIBAKE = {
    'â\x80\x99': "'",
    'â\x80\x98': "'",
    'â\x80\x9c': '"',
    'â\x80\x9d': '"',
    'â\x80\x93': '-',
    'â\x80\x94': '-',
    'â\x80¦': '...',
    'Ã©': 'e',
    'Ã¨': 'e',
    'Ã¡': 'a',
    'Ã ': 'a',
    'Ã³': 'o',
    'Ã±': 'n',
    'Ã¼': 'u',
}


def fix_mojibake(text):
    if not isinstance(text, str):
        return text
    for bad, good in _MOJIBAKE.items():
        text = text.replace(bad, good)
    return text


@lru_cache(maxsize=1)
def load_abbreviations():
    """config/abbreviations.yaml flattened into {abbreviation: expansion}."""
    with open(CONFIG_DIR / "abbreviations.yaml") as f:
        groups = yaml.safe_load(f)
    flat = {}
    for abbrevs in groups.values():
        flat.update(abbrevs)
    return flat


def expand_abbreviations(text):
    """Expand whole-token abbreviations once.

    A token matches when its core (outer punctuation stripped) equals a key,
    ignoring case, so "Na'cho" is left alone. Keys in AMBIGUOUS_ABBREV_KEYS
    also require the token to be uppercase. Callers must not expand twice.
    """
    if not isinstance(text, str) or not text.strip():
        return text

    abbreviations = load_abbreviations()
    out = []
    for token in text.split():
        core = token.strip(string.punctuation)
        expansion = None
        if core:
            for abbrev, full in abbreviations.items():
                if core.lower() == abbrev.lower():
                    if core.lower() in AMBIGUOUS_ABBREV_KEYS and not core.isupper():
                        continue
                    expansion = full
                    break
        if expansion is None:
            out.append(token)
        else:
            lead = token[:len(token) - len(token.lstrip(string.punctuation))]
            trail = token[len(token.rstrip(string.punctuation)):]
            out.append(lead + expansion + trail)
    return ' '.join(out)


def normalize_text(text, keep_empty_as_placeholder=False, expand_abbrev=True):
    """Lowercase ASCII text with abbreviations expanded and punctuation reduced.

    Missing or blank input returns "" or, with keep_empty_as_placeholder,
    "[empty]".
    """
    empty = "[empty]" if keep_empty_as_placeholder else ""
    if pd.isna(text):
        return empty
    text = str(text)
    if text.lower() == 'nan':
        return empty

    text = fix_mojibake(text)
    if expand_abbrev:
        text = expand_abbreviations(text)

    # Map typographic dashes and quotes before the ASCII step would drop them.
    text = re.sub(r'[\u2010-\u2015\u2212]', '-', text)
    text = re.sub(r'[\u2018\u2019\u2032]', "'", text)
    text = re.sub(r'[\u201c\u201d\u2033]', '"', text)

    text = unicodedata.normalize('NFKD', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = text.lower().strip()

    # ";" separates variants in O*NET documents, so it cannot survive here.
    for ch in ',.;:?!"`':
        text = text.replace(ch, '')
    text = text.replace('\\ ', ' ')

    text = re.sub(r'\s*\(\s*', ' (', text)
    text = re.sub(r'\s*\)\s*', ') ', text)
    text = re.sub(r'\s*/\s*', '/', text)
    text = re.sub(r'\s+', ' ', text)

    # Rejoin spelled-out initials: "u s" (from "U. S.") becomes "us".
    text = re.sub(r'\b([a-z]) ([a-z])(?:\b| ([a-z])(?:\b| ([a-z])\b)?)',
                  lambda m: ''.join(filter(None, m.groups())),
                  text)

    text = text.strip()
    return text or empty


def pluralize_to_singular(text):
    """Singularize the last word of a phrase, never producing a non-word.

    The candidate from inflect is accepted only if the word does not end in a
    protected suffix, the result has at least 3 characters, and pluralizing it
    again gives back the original word.
    """
    if not isinstance(text, str) or not text.strip():
        return text

    words = text.split()
    last = words[-1]
    lw = last.lower()
    if lw.endswith(_PROTECTED_SINGULAR_SUFFIXES):
        return text

    candidate = _INFLECT.singular_noun(last)
    if not candidate or candidate.lower() == lw or len(candidate) < 3:
        return text
    try:
        back = _INFLECT.plural_noun(candidate)
    except Exception:
        back = None
    if not back or back.lower() != lw:
        return text

    words[-1] = candidate.capitalize() if last[:1].isupper() else candidate
    return ' '.join(words)
