"""Freeze the worksheet reviewers coded, and rebuild worksheets from it.

Once reviewers have coded a worksheet, its rows, batches, statuses, scores and
candidates must not move, even if the matcher changes. A frame is that
worksheet plus three columns written by freeze():

- survey_id: the participant ID as the survey holds it (older worksheets
  dropped leading zeros);
- source_row: the survey row whose answers the worksheet showed;
- known_difference: where the run current at freezing time already
  disagreed with the frame on status or candidates, and how.

build_from_frame() returns worksheets with the frame's rows and candidates
and the answers verbatim from source_row. It stops if the current run
disagrees with the frame anywhere not recorded in known_difference.
"""

import pandas as pd

from onet.survey import EMPTY, TEXT_FIELDS
from onet.text_normalization import normalize_text

FRAME_COLUMNS = ["survey_id", "source_row", "known_difference"]
CANDIDATE_CODES = [f"cand{i}_code" for i in range(1, 6)]


def _shown_text(rows):
    """The cleaned answers, as pre-verbatim worksheets displayed them."""
    shown = pd.DataFrame(index=rows.index)
    shown["job_title_raw"] = rows.job_title_clean
    for field in ("industry", "jobDescription"):
        shown[field] = [normalize_text(v, keep_empty_as_placeholder=True) for v in rows[field]]
    return shown


def differences(frame, queue):
    """{survey_id: description} where the queue's status or candidates differ."""
    current = queue.set_index("participant_id")
    found = {}
    for r in frame.itertuples():
        now = current.loc[r.survey_id]
        parts = []
        if r.match_status != now.match_status:
            parts.append(f"status {r.match_status} -> {now.match_status}")
        if [getattr(r, c) for c in CANDIDATE_CODES] != now[CANDIDATE_CODES].tolist():
            parts.append("candidates")
        if parts:
            found[r.survey_id] = "; ".join(parts)
    return found


def _check_same_participants(frame, queue):
    in_frame, in_queue = set(frame.survey_id), set(queue.participant_id)
    if in_frame != in_queue:
        raise SystemExit(
            f"Frame and review queue cover different participants: "
            f"{len(in_frame - in_queue)} only in the frame, {len(in_queue - in_frame)} "
            "only in the queue.")


def freeze(worksheet, rows, queue):
    """Return the frame for a coded worksheet.

    `rows` is clean_responses() of the survey (every row, before duplicates
    are resolved); `queue` is the current review queue.
    """
    by_short_id = {}
    for pid in rows.participant_id.unique():
        by_short_id.setdefault(pid.lstrip("0"), []).append(pid)
    shown = _shown_text(rows)

    survey_ids, source_rows, problems = [], [], []
    for r in worksheet.itertuples():
        ids = by_short_id.get(str(r.participant_id).lstrip("0"), [])
        if len(ids) != 1:
            problems.append(f"{r.participant_id}: {len(ids)} survey IDs match")
            survey_ids.append(None)
            source_rows.append(None)
            continue
        mine = shown[rows.participant_id == ids[0]]
        hits = mine[(mine[list(TEXT_FIELDS.values())]
                     == [getattr(r, f) for f in TEXT_FIELDS.values()]).all(axis=1)]
        verbatims = rows.loc[hits.index, list(TEXT_FIELDS.values())].drop_duplicates()
        if len(hits) == 0 or len(verbatims) > 1:
            problems.append(f"{r.participant_id}: {len(hits)} survey rows match the text shown, "
                            f"{len(verbatims)} distinct answers")
        survey_ids.append(ids[0])
        source_rows.append(int(rows.loc[hits.index[0], "source_row"]) if len(hits) else None)
    if problems:
        raise SystemExit("Cannot place every worksheet row in the survey:\n  "
                         + "\n  ".join(problems))

    frame = worksheet.copy()
    frame["survey_id"] = survey_ids
    frame["source_row"] = source_rows
    _check_same_participants(frame, queue)
    known = differences(frame, queue)
    frame["known_difference"] = frame.survey_id.map(known).fillna("")
    return frame


def build_from_frame(frame, rows, queue):
    """Worksheet rows from the frame, answers verbatim from each source_row.

    Returns (worksheet, survey IDs whose recorded difference has gone away).
    """
    _check_same_participants(frame, queue)
    found = differences(frame, queue)
    recorded = dict(zip(frame.survey_id, frame.known_difference))
    unexpected = {pid: d for pid, d in found.items() if recorded[pid] != d}
    if unexpected:
        raise SystemExit(
            "The current run disagrees with the review frame beyond what the frame "
            "records, so the worksheets would no longer match what reviewers coded:\n  "
            + "\n  ".join(f"{pid}: {d}" for pid, d in sorted(unexpected.items())))
    resolved = sorted(pid for pid, d in recorded.items() if d and pid not in found)

    source = rows.set_index("source_row")
    at = frame.source_row.astype(int)
    sheet = frame.drop(columns=FRAME_COLUMNS)
    sheet["participant_id"] = frame.survey_id.to_numpy()
    for field in TEXT_FIELDS.values():
        sheet[field] = source.loc[at, field].to_numpy()
    clean = source.loc[at, "job_title_clean"].to_numpy()
    sheet.insert(sheet.columns.get_loc("job_title_raw") + 1, "job_title_clean", clean)
    return sheet, resolved
