#!/usr/bin/env python3
"""Stage 04: write one worksheet per reviewer.

Without a review frame, each worksheet is the review queue with batch_id and
the reviewer columns RA_choice_code, RA_notes, flag_ambiguous and
flag_non_job added. With review.frame set in config/pipeline.yaml, the
worksheets keep the frame's rows, batches, statuses, scores and candidates,
and take the answers verbatim from the survey; the stage stops if the current
run disagrees with the frame beyond what the frame records. The two
reviewers code independently and then reconcile batch by batch.
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

from onet.config import load_settings, require
from onet.review_frame import build_from_frame
from onet.survey import clean_responses, read_sav

ONET_CODE = re.compile(r'^\d{2}-\d{4}\.\d{2}$')
REQUIRED = ["participant_id", "job_title_raw", "industry", "jobDescription",
            "match_status", "cand1_code", "cand1_title", "cand1_score"]
REVIEWER_COLUMNS = {"RA_choice_code": "", "RA_notes": "", "flag_ambiguous": 0, "flag_non_job": 0}


def validate(queue, label):
    missing = [c for c in REQUIRED if c not in queue.columns]
    if missing:
        raise SystemExit(f"{label} is missing columns: {missing}")
    if queue.participant_id.duplicated().any():
        raise SystemExit(f"{label} has duplicate participant IDs")

    codes = queue.filter(regex=r'^cand\d+_code$').stack()
    codes = codes[codes != ""]
    bad = codes[~codes.str.match(ONET_CODE)]
    if len(bad):
        raise SystemExit(f"{len(bad)} candidate codes are not O*NET-SOC codes, e.g. {bad.iloc[0]!r}")


def prepare(queue, batch_size):
    validate(queue, "Review queue")
    sheet = queue.copy()
    sheet.insert(0, "batch_id", sheet.index // batch_size + 1)
    for col, value in REVIEWER_COLUMNS.items():
        sheet[col] = value
    return sheet


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path,
                        help="review queue (default: <survey>_onet_review_queue.csv in the output directory)")
    parser.add_argument("--output-dir", type=Path,
                        help="where to write the copies (default: the output directory)")
    parser.add_argument("--batch-size", type=int, default=60,
                        help="rows per batch, without a review frame (default: 60)")
    args = parser.parse_args()
    if args.batch_size < 1:
        sys.exit("--batch-size must be at least 1")
    settings = load_settings()
    args.input = args.input or settings.output_file("review_queue.csv")
    args.output_dir = args.output_dir or settings.output_dir

    queue = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    if settings.review_frame is None:
        sheet = prepare(queue, args.batch_size)
    else:
        require(settings.review_frame, settings.survey_file)
        validate(queue, "Review queue")
        frame = pd.read_csv(settings.review_frame, dtype=str, keep_default_na=False)
        raw = read_sav(settings.survey_file, usecols=list(settings.columns.values()))
        sheet, resolved = build_from_frame(frame, clean_responses(raw, settings.columns), queue)
        validate(sheet, "Review frame")
        print(f"Built from review frame {settings.review_frame.name}; "
              f"{(frame.known_difference != '').sum()} rows differ from the current run "
              "as recorded in the frame")
        if resolved:
            print(f"  the current run now agrees with the frame for {len(resolved)} of them")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for reviewer in ("RA1", "RA2"):
        path = args.output_dir / f"{args.input.stem}_{reviewer}.csv"
        sheet.to_csv(path, index=False)
        print(f"Wrote {path}")
    print(f"{len(sheet)} rows in {sheet.batch_id.astype(int).max()} batches; "
          + ", ".join(f"{s} {n}" for s, n in sheet.match_status.value_counts().items()))


if __name__ == "__main__":
    main()
