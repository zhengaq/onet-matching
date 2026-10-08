"""Read the survey release, keep the answers verbatim and clean the job title.

The job title, industry and duty answers are kept as the participant wrote
them, trimmed of surrounding whitespace, in the columns reviewers see. Only
the job title is matched, so only it gets a cleaned copy (job_title_clean).
"""

import pandas as pd
import pyreadstat

from onet.text_normalization import normalize_text

EMPTY = "[empty]"
# Survey role -> column holding the verbatim answer.
TEXT_FIELDS = {"job_title": "job_title_raw", "industry": "industry", "duties": "jobDescription"}
CLEAN_COLUMNS = ["participant_id", "source_row", *TEXT_FIELDS.values(), "job_title_clean"]


def read_sav(path, usecols=None):
    """Read an SPSS file, trying explicit encodings if detection fails."""
    error = None
    for encoding in (None, "latin1", "windows-1252", "utf-8"):
        try:
            df, _ = pyreadstat.read_sav(str(path), encoding=encoding,
                                        apply_value_formats=False, usecols=usecols)
            return df
        except Exception as e:
            error = e
    raise RuntimeError(f"Could not read {path}: {error}")


def _as_id_text(series):
    """Participant IDs as text, keeping leading zeros of string IDs."""
    if pd.api.types.is_numeric_dtype(series):
        if series.isna().any() or (series % 1 != 0).any():
            raise ValueError("Numeric participant IDs must be whole numbers")
        return series.astype("int64").astype(str)
    return series.astype(str).str.strip()


# --- Cleaning ---------------------------------------------------------------

def verbatim(value):
    """The answer as written, trimmed of surrounding whitespace; "" if missing."""
    return "" if pd.isna(value) else str(value).strip()


def clean_responses(raw, columns):
    """One row per survey row, in file order: verbatim answers and the cleaned title.

    `columns` maps id, job_title, industry and duties to survey variables.
    """
    missing = [v for v in columns.values() if v not in raw.columns]
    if missing:
        raise SystemExit(f"Survey is missing configured columns: {missing}")

    clean = pd.DataFrame({
        "participant_id": _as_id_text(raw[columns["id"]]).to_numpy(),
        "source_row": range(len(raw)),
    })
    if (clean.participant_id == "").any():
        raise SystemExit("Survey has rows without a participant ID")

    for role, field in TEXT_FIELDS.items():
        clean[field] = [verbatim(v) for v in raw[columns[role]]]
    clean["job_title_clean"] = [normalize_text(v, keep_empty_as_placeholder=True)
                                for v in raw[columns["job_title"]]]
    return clean[CLEAN_COLUMNS]


def resolve_duplicates(clean):
    """Keep one row per participant: the first with a job title, else the first.

    The kept row supplies the job title, industry and duties together.
    Returns (rows kept, number of duplicated IDs, number whose rows disagree).
    """
    keep, n_dup, n_conflict = [], 0, 0
    for _, rows in clean.groupby("participant_id", sort=False):
        if len(rows) > 1:
            n_dup += 1
            n_conflict += len(rows[list(TEXT_FIELDS.values())].drop_duplicates()) > 1
        titled = rows[rows.job_title_clean != EMPTY]
        keep.append((titled if len(titled) else rows).index[0])
    return clean.loc[sorted(keep)].reset_index(drop=True), n_dup, n_conflict


def read_clean(path):
    """Read the cleaned survey written by 01_clean_survey.py."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df["source_row"] = df.source_row.astype(int)
    return df
