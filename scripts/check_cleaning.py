#!/usr/bin/env python3
"""QC: check that answers are shown verbatim and cleaning added no words.

A) Verbatim answers. The job title, industry and duty columns of the cleaned
   survey must equal the survey answer, trimmed of surrounding whitespace.
   Where stage 03 or 04 files exist, their columns of the same names must
   equal the survey answer from the participant's row in the cleaned survey,
   or, for worksheets built from a review frame, the row the frame names.
B) Cleaned job titles. Every token must appear in the raw answer (normalized
   without abbreviation expansion) or come from expanding an abbreviation
   the respondent wrote as a whole token, in uppercase for the ambiguous keys.
C) O*NET documents. Where a title was singularized, the new last word must
   be a real English word (wordfreq) or inflect's singular of the original.

Run after both stage 01 scripts, and again after stage 03 or 04 to cover
their files. Exits with status 1 if anything is found.
"""

import argparse
import re
import sys

import inflect
import pandas as pd
from wordfreq import zipf_frequency

from onet.config import load_settings, require
from onet.survey import EMPTY, TEXT_FIELDS, read_clean, read_sav, verbatim
from onet.text_normalization import AMBIGUOUS_ABBREV_KEYS, load_abbreviations, normalize_text

WORD = re.compile(r"[a-z0-9']+")
RAW_WORD = re.compile(r"[A-Za-z0-9']+")
MIN_ZIPF = 1.0  # wordfreq frequency treated as a real word


def tokens(text):
    return [t.strip("'") for t in WORD.findall(str(text).lower()) if t.strip("'")]


def expansion_tokens(raw, abbreviations):
    """Tokens explained by abbreviations written as whole tokens in `raw`."""
    allowed = set()
    for token in RAW_WORD.findall(str(raw)):
        core = token.strip("'")
        for key, expansion in abbreviations.items():
            if not core or core.lower() != key.lower():
                continue
            if key.lower() in AMBIGUOUS_ABBREV_KEYS and not core.isupper():
                continue
            allowed.update(tokens(expansion))
            allowed.add(core.lower())
            break
    return allowed


def check_verbatim(raw, columns, shown):
    """Return {label: [(participant_id, field), ...]} of answers not shown verbatim.

    `shown` maps a label to (table, {participant_id: survey row}).
    """
    findings = {}
    for label, (table, source_rows) in shown.items():
        found = []
        for role, field in TEXT_FIELDS.items():
            source = raw[columns[role]].to_numpy()
            found += [(pid, field) for pid, value in zip(table.participant_id, table[field])
                      if pid not in source_rows or value != verbatim(source[source_rows[pid]])]
        findings[label] = found
    return findings


def check_clean_titles(clean, raw, columns, abbreviations):
    """Return [(participant_id, added tokens), ...] for the cleaned job titles."""
    source = raw[columns["job_title"]].to_numpy()
    found = []
    for row in clean.itertuples():
        original, cleaned = source[row.source_row], row.job_title_clean
        if pd.isna(original) or cleaned in ("", EMPTY):
            continue
        allowed = (set(tokens(normalize_text(original, expand_abbrev=False)))
                   | expansion_tokens(original, abbreviations))
        added = [t for t in tokens(cleaned) if t not in allowed]
        if added:
            found.append((row.participant_id, added))
    return found


def check_singulars(doc_texts):
    """Return [(title, singular), ...] whose singular ends in a non-word."""
    engine = inflect.engine()
    found = []
    for doc in doc_texts:
        parts = [p.strip() for p in str(doc).split(";")]
        if len(parts) < 2:
            continue
        title, singular = parts[0].split(), parts[1].split()
        if not title or not singular or title[-1] == singular[-1]:
            continue
        last = singular[-1].strip("'").lower()
        if not last:
            continue
        canonical = engine.singular_noun(title[-1]) or ""
        if len(last) < 3 or not (zipf_frequency(last, "en") >= MIN_ZIPF
                                 or last == str(canonical).lower()):
            found.append((parts[0], parts[1]))
    return found


def main():
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    settings = load_settings()
    require(settings.survey_file, settings.survey_clean, settings.onet_docs)

    clean = read_clean(settings.survey_clean)
    raw = read_sav(settings.survey_file, usecols=list(settings.columns.values()))
    kept = dict(zip(clean.participant_id, clean.source_row))
    worksheet_rows = kept
    if settings.review_frame is not None:
        require(settings.review_frame)
        frame = pd.read_csv(settings.review_frame, dtype=str, keep_default_na=False)
        worksheet_rows = dict(zip(frame.survey_id, frame.source_row.astype(int)))
    shown = {"cleaned survey": (clean, kept)}
    for kind in ("exact_match.csv", "review_queue.csv",
                 "review_queue_RA1.csv", "review_queue_RA2.csv"):
        path = settings.output_file(kind)
        if path.exists():
            rows = worksheet_rows if kind.startswith("review_queue_RA") else kept
            shown[path.name] = (pd.read_csv(path, dtype=str, keep_default_na=False), rows)
    not_verbatim = check_verbatim(raw, settings.columns, shown)
    added = check_clean_titles(clean, raw, settings.columns, load_abbreviations())
    singulars = check_singulars(
        pd.read_csv(settings.onet_docs, dtype=str, keep_default_na=False).doc_text)

    total = 0
    print("A) Answers not shown verbatim")
    for label, found in not_verbatim.items():
        total += len(found)
        print(f"   {label:<40} {len(found)}")
        for pid, field in found[:10]:
            print(f"      participant {pid}: {field}")
    print("B) Words added to cleaned job titles")
    total += len(added)
    print(f"   job_title_clean {len(added)}")
    for pid, words in added[:10]:
        print(f"      participant {pid}: {words}")
    print("C) Non-word singulars in O*NET documents")
    total += len(singulars)
    print(f"   onet_docs       {len(singulars)}")
    for title, singular in singulars[:10]:
        print(f"      {title!r} -> {singular!r}")

    print(f"\n{total} findings")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
