import pandas as pd
import pytest

from onet.survey import EMPTY, clean_responses, resolve_duplicates

COLUMNS = {"id": "pid", "job_title": "q_title", "industry": "q_industry", "duties": "q_duties"}


def survey(rows):
    return pd.DataFrame(rows, columns=["pid", "q_title", "q_industry", "q_duties"])


def test_ids_keep_leading_zeros():
    clean = clean_responses(survey([["00123", "Nurse", "Hospital", "Patient care"]]), COLUMNS)
    assert clean.participant_id.tolist() == ["00123"]


def test_answers_kept_verbatim_and_only_the_title_cleaned():
    clean = clean_responses(survey([
        ["001", "  IT Manager, RN ", "IT (healthcare)", "HR, payroll."],
        ["002", None, "", "  "],
    ]), COLUMNS)
    assert clean.loc[0, ["job_title_raw", "industry", "jobDescription"]].tolist() == [
        "IT Manager, RN", "IT (healthcare)", "HR, payroll."]
    assert clean.loc[0, "job_title_clean"] == "information technology manager registered nurse"
    assert clean.loc[1, ["job_title_raw", "industry", "jobDescription"]].tolist() == ["", "", ""]
    assert clean.loc[1, "job_title_clean"] == EMPTY


def test_numeric_ids_become_whole_numbers():
    clean = clean_responses(survey([[123.0, "Teacher", "School", "Teaching"]]), COLUMNS)
    assert clean.participant_id.tolist() == ["123"]


def test_duplicate_keeps_first_titled_row_with_its_context():
    rows = clean_responses(survey([
        ["001", "", "", ""],
        ["001", "Welder", "Shipyard", "Welding hulls"],
        ["002", "Teacher", "School", "Teaching"],
        ["002", "Tutor", "Online", "Tutoring"],
        ["003", "Chef", "Restaurant", "Cooking"],
    ]), COLUMNS)
    clean, n_dup, n_conflict = resolve_duplicates(rows)
    assert (n_dup, n_conflict) == (2, 2)
    assert clean.participant_id.tolist() == ["001", "002", "003"]
    assert clean.source_row.tolist() == [1, 2, 4]
    first = clean.iloc[0]
    assert (first.job_title_raw, first.industry, first.jobDescription) == (
        "Welder", "Shipyard", "Welding hulls")
    assert first.job_title_clean == "welder"


def test_missing_configured_column_is_an_error():
    with pytest.raises(SystemExit):
        clean_responses(survey([["1", "a", "b", "c"]]).drop(columns="q_duties"), COLUMNS)
