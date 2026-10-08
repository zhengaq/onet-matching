import pandas as pd
import pytest

from onet.review_frame import build_from_frame, freeze
from onet.survey import clean_responses

COLUMNS = {"id": "pid", "job_title": "q_title", "industry": "q_industry", "duties": "q_duties"}
CODES = [f"cand{i}_code" for i in range(1, 6)]


def survey_rows():
    raw = pd.DataFrame([
        ["001", "IT Manager", "Tech, Inc.", "Run IT"],
        ["002", "Welder", "Shipyard", "Welding"],
        ["002", "Pipe welder", "Shipyard", "Pipes"],
        ["003", "", "", ""],
    ], columns=["pid", "q_title", "q_industry", "q_duties"])
    return clean_responses(raw, COLUMNS)


def worksheet():
    """A pre-verbatim worksheet: IDs without leading zeros, cleaned text shown."""
    rows = [
        ["1", "2", "pipe welder", "shipyard", "pipes", "NEEDS_REVIEW", "51-4121.00"],
        ["1", "1", "information technology manager", "tech inc", "run information technology",
         "NEEDS_REVIEW", "11-3021.00"],
        ["2", "3", "[empty]", "[empty]", "[empty]", "NA_INPUT", ""],
    ]
    sheet = pd.DataFrame(rows, columns=["batch_id", "participant_id", "job_title_raw", "industry",
                                        "jobDescription", "match_status", "cand1_code"])
    for c in CODES[1:]:
        sheet[c] = ""
    sheet["RA_choice_code"] = ""
    return sheet


def queue(**changes):
    q = pd.DataFrame({"participant_id": ["001", "002", "003"],
                      "match_status": ["NEEDS_REVIEW", "NEEDS_REVIEW", "NA_INPUT"],
                      "cand1_code": ["11-3021.00", "47-2111.00", ""]})
    for c in CODES[1:]:
        q[c] = ""
    for pid, status in changes.items():
        q.loc[q.participant_id == pid, "match_status"] = status
    return q


def test_freeze_pins_rows_ids_and_known_differences():
    frame = freeze(worksheet(), survey_rows(), queue())
    assert frame.survey_id.tolist() == ["002", "001", "003"]
    assert frame.source_row.tolist() == [2, 0, 3]  # 002 showed its second survey row
    assert frame.known_difference.tolist() == ["candidates", "", ""]


def test_worksheet_keeps_frame_and_shows_answers_verbatim():
    frame = freeze(worksheet(), survey_rows(), queue())
    sheet, resolved = build_from_frame(frame, survey_rows(), queue())
    assert resolved == []
    assert sheet.participant_id.tolist() == ["002", "001", "003"]
    assert sheet.batch_id.tolist() == ["1", "1", "2"]
    assert sheet.cand1_code.tolist() == ["51-4121.00", "11-3021.00", ""]
    assert sheet.loc[1, ["job_title_raw", "job_title_clean", "industry", "jobDescription"]].tolist() == [
        "IT Manager", "information technology manager", "Tech, Inc.", "Run IT"]
    assert sheet.loc[0, "job_title_raw"] == "Pipe welder"
    assert sheet.loc[2, ["job_title_raw", "job_title_clean"]].tolist() == ["", "[empty]"]
    assert list(sheet.columns[:4]) == ["batch_id", "participant_id", "job_title_raw", "job_title_clean"]
    assert "source_row" not in sheet.columns


def test_unrecorded_difference_stops_the_build():
    frame = freeze(worksheet(), survey_rows(), queue())
    with pytest.raises(SystemExit):
        build_from_frame(frame, survey_rows(), queue(**{"001": "NO_MATCH"}))


def test_worksheet_text_must_match_a_survey_row():
    sheet = worksheet()
    sheet.loc[1, "job_title_raw"] = "information technology director"
    with pytest.raises(SystemExit):
        freeze(sheet, survey_rows(), queue())
