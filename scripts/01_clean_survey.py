#!/usr/bin/env python3
"""Stage 01: clean the survey's job title, industry and duty answers.

Reads the survey named in config/pipeline.yaml and writes one row per
participant to <survey>_clean.csv in the intermediate directory. The answers
are kept verbatim; the job title also gets a cleaned copy (normalized,
abbreviations expanded) for matching. A participant with several rows keeps the first row
that has a job title (or the first row), with its industry and duties.
"""

import argparse

from onet.config import load_settings, require
from onet.survey import TEXT_FIELDS, read_sav, clean_responses, resolve_duplicates


def main():
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    settings = load_settings()
    require(settings.survey_file)

    raw = read_sav(settings.survey_file, usecols=list(settings.columns.values()))
    rows = clean_responses(raw, settings.columns)
    clean, n_dup, n_conflict = resolve_duplicates(rows)
    print(f"{settings.survey_file.name}: {len(rows)} rows, {len(clean)} participants")
    if n_dup:
        print(f"  {n_dup} participant IDs appear more than once; one row kept for each "
              f"({n_conflict} with differing answers)")

    for field in TEXT_FIELDS.values():
        print(f"  {field:<15} present for {(clean[field] != '').sum()}")

    settings.intermediate_dir.mkdir(parents=True, exist_ok=True)
    clean.to_csv(settings.survey_clean, index=False)
    print(f"Wrote {settings.survey_clean}")


if __name__ == "__main__":
    main()
