#!/usr/bin/env python3
"""Freeze a worksheet reviewers have coded as the frame for stage 04.

Run once per reviewer round, after stage 03, on the worksheet the reviewers
received. Writes the worksheet with survey_id, source_row and
known_difference added (see src/onet/review_frame.py), then name the output
under review.frame in config/pipeline.yaml. Stops if a worksheet row cannot
be placed on exactly one survey answer.
"""

import argparse
from pathlib import Path

import pandas as pd

from onet.config import load_settings, require
from onet.review_frame import freeze
from onet.survey import clean_responses, read_sav


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("worksheet", type=Path, help="the worksheet the reviewers coded")
    parser.add_argument("output", type=Path, help="where to write the frame")
    args = parser.parse_args()
    settings = load_settings()
    queue_path = settings.output_file("review_queue.csv")
    require(args.worksheet, settings.survey_file, queue_path)
    if args.output.exists():
        raise SystemExit(f"{args.output} exists; a frame is written once.")

    worksheet = pd.read_csv(args.worksheet, dtype=str, keep_default_na=False)
    queue = pd.read_csv(queue_path, dtype=str, keep_default_na=False)
    raw = read_sav(settings.survey_file, usecols=list(settings.columns.values()))
    frame = freeze(worksheet, clean_responses(raw, settings.columns), queue)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    known = frame[frame.known_difference != ""]
    print(f"Wrote {args.output}: {len(frame)} rows, "
          f"{(frame.participant_id != frame.survey_id).sum()} IDs restored to survey form")
    print(f"{len(known)} rows where the current run already differs from the worksheet:")
    for r in known.itertuples():
        print(f"  {r.survey_id}: {r.known_difference}")


if __name__ == "__main__":
    main()
