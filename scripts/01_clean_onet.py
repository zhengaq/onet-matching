#!/usr/bin/env python3
"""Stage 01: build one O*NET reference document per occupation.

Stacks the alternate and reported titles named in config/pipeline.yaml and
writes onet_docs.csv to the intermediate directory, with columns onet_code,
title and doc_text, where doc_text is "title; singular title; variant; ...".
Accepts the official O*NET Excel files or copies with the columns renamed to
onet_code, title and AlternateTitle.
"""

import argparse

import pandas as pd

from onet.config import load_settings, require
from onet.text_normalization import normalize_text, pluralize_to_singular

OFFICIAL_COLUMNS = {
    "O*NET-SOC Code": "onet_code",
    "Title": "title",
    "Alternate Title": "AlternateTitle",
    "Reported Job Title": "AlternateTitle",
}


def read_titles(path):
    df = pd.read_excel(path).rename(columns=OFFICIAL_COLUMNS)
    missing = {"onet_code", "title", "AlternateTitle"} - set(df.columns)
    if missing:
        raise SystemExit(f"{path}: missing columns {sorted(missing)}")
    print(f"{path.name}: {len(df):,} rows")
    return df[["onet_code", "title", "AlternateTitle"]]


def main():
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    settings = load_settings()
    require(settings.onet_alternate_titles, settings.onet_reported_titles)

    titles = pd.concat([read_titles(settings.onet_alternate_titles),
                        read_titles(settings.onet_reported_titles)], ignore_index=True)
    before = len(titles)
    titles = titles.drop_duplicates(subset=["onet_code", "AlternateTitle"])
    print(f"Removed {before - len(titles):,} duplicate (code, title) pairs")

    titles["title_normalized"] = titles.title.map(
        lambda x: normalize_text(x, keep_empty_as_placeholder=True))
    titles["alt_normalized"] = titles.AlternateTitle.map(
        lambda x: normalize_text(x, keep_empty_as_placeholder=True))

    docs = titles.groupby("onet_code", as_index=False).agg(
        title=("title_normalized", "first"),
        variants=("alt_normalized", lambda x: "; ".join(sorted(set(x)))),
    )
    docs["doc_text"] = (docs.title + "; " + docs.title.map(pluralize_to_singular)
                        + "; " + docs.variants)

    settings.intermediate_dir.mkdir(parents=True, exist_ok=True)
    docs[["onet_code", "title", "doc_text"]].to_csv(settings.onet_docs, index=False)
    print(f"Wrote {len(docs):,} occupations to {settings.onet_docs}")


if __name__ == "__main__":
    main()
