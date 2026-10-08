# onet-matching

Codes free-text occupation answers from the CATSLife PhenX2 survey to O\*NET-SOC
occupations. Each participant describes their job title, industry and duties.
The pipeline matches the job title against O\*NET, accepts confident matches
automatically, and puts everything else in a review queue that shows reviewers
five candidate occupations next to the participant's industry and duties.

## Data

No participant data is stored in this repository. The pipeline takes two
inputs:

- a survey release with each participant's free-text job title, industry and
  duties;
- the O\*NET 30.1 alternate and reported titles, in Excel format, from
  <https://www.onetcenter.org/database.html>.

Their locations and the survey variables are set in a local configuration
file (see Setup).

## Setup

Requires [uv](https://docs.astral.sh/uv/). A GPU is optional; the embedding step
runs on CPU, more slowly.

```
uv sync
cp config/pipeline.example.yaml config/pipeline.yaml   # then fill it in
uv run pytest                                          # synthetic tests, no data needed
```

The `config/pipeline.yaml` shall name the directory holding the
inputs and outputs, the input files, the survey variables, and where the
pipeline writes cleaned inputs, results, the model cache and logs.
`ONET_DATA_ROOT`, if set, overrides its `data_root`.

The first run of stage 02 downloads the embedding model `BAAI/bge-large-en-v1.5`
(about 1.3 GB) into the cache directory.

## Running

From the repository root, in order:

| Command | Writes |
|---|---|
| `uv run python scripts/01_clean_onet.py` | `onet_docs.csv`, the O\*NET reference documents |
| `uv run python scripts/01_clean_survey.py` | `<survey>_clean.csv`, one row per participant |
| `uv run python scripts/check_cleaning.py` | nothing; exits 1 if an answer is not shown verbatim or cleaning added words |
| `uv run python scripts/02_embed.py` | `onet_embeddings.npz`, `<survey>_job_embeddings.npz` |
| `uv run python scripts/03_match.py` | `<survey>_onet_exact_match.csv`, `_review_queue.csv`, `_report.md` |
| `uv run python scripts/04_prepare_review_files.py` | `<survey>_onet_review_queue_RA1.csv`, `_RA2.csv` |

`<survey>` is `survey.name` in the configuration.

Stage 03 stops if the embeddings no longer match the cleaned inputs; re-run
stage 02 after any change to stage 01. Run `check_cleaning.py` again after
stage 04 to confirm the reviewer files show the answers verbatim.

### Reviewer rounds

Once reviewers have coded a worksheet, its rows and candidates must not move.
Freeze it once, after stage 03:

```
uv run python scripts/freeze_review_frame.py <coded worksheet> <frame file>
```

and name the frame file under `review.frame` in `config/pipeline.yaml`. The frame
is the worksheet plus, for each row, the survey row its text came from and
any difference from the run current at freezing time. While a frame is set,
stage 04 builds the worksheets from it: same rows, batches, statuses, scores
and candidates, with the answers verbatim. It stops if the current run
disagrees with the frame anywhere the frame does not record, so a matching
change cannot silently move what reviewers coded. Remove `review.frame` to
build worksheets for a new round from the current run.

## Outputs

- `*_exact_match.csv`: auto-accepted participants with the chosen O\*NET code,
  score, margin, job title, industry and duties. Section 5 of the report lists these by
  occupation for spot checks.
- `*_review_queue.csv`: one row per remaining participant, `NEEDS_REVIEW` first,
  then `NO_MATCH`, then `NA_INPUT` (no job title). Columns: `participant_id`,
  `job_title_raw`, `job_title_clean` (the cleaned title that was matched),
  `industry`, `jobDescription` (duties),
  `best_score_final`, `margin`, `match_status`, and `cand1`..`cand5`
  code, title and score.
- `*_review_queue_RA1.csv`, `_RA2.csv`: identical worksheets for two
  independent reviewers, with `batch_id` and the columns they fill in:
  `RA_choice_code`, `RA_notes`, `flag_ambiguous`, `flag_non_job`. Built from
  the review frame when one is set, so they can differ from the review queue
  where the frame records a difference.
- `*_report.md`: input coverage, status counts, score distributions, evidence
  types and the settings used.

## Method

**Answers.** The job title, industry and duty answers reach every output as
the participant wrote them, trimmed of surrounding whitespace: `job_title_raw`,
`industry` and `jobDescription` are never cleaned, so a reviewer reads the
participant's own words. Only the job title is matched, and its cleaned form
is a separate column, `job_title_clean`.

**Cleaning.** For matching, job titles are lowercased and converted to ASCII,
punctuation is reduced, and abbreviations in
[`config/abbreviations.yaml`](config/abbreviations.yaml) are expanded as whole
tokens; two-letter keys that are also words or state
codes (`IT`, `MA`, `PA`) expand only when written in uppercase. A participant ID
that appears on several survey rows keeps the first row with a job title, and
that row supplies the industry and duties. `check_cleaning.py` confirms that
the answer columns equal the source, in the cleaned survey and in every
review file present, and that cleaning introduced no word absent from the
source.

**Reference.** One document per O\*NET occupation: its title, the singular
form of the title, and every alternate and reported title.

**Matching.** Titles and documents are embedded with `BAAI/bge-large-en-v1.5`.
For each title the 20 nearest documents by cosine similarity are rescored:

- +0.15 if the title equals one of the document's titles exactly;
- +0.10 if both contain an equipment keyword (`ct`, `mri`, `laser`, ...),
  tested as substrings; otherwise +0.05 if both share a word found in fewer
  than 5% of documents;
- -0.10 if the occupation title is generic ("manager", "specialist", ...)
  and the cosine is below 0.75.

The five best become the candidates. The rank-1 score and its margin over
rank 2 set the status: `AUTO_ACCEPT` at score >= 0.80 with margin >= 0.15, or
an exact title with margin >= 0.10; `NO_MATCH` below 0.50; `NEEDS_REVIEW`
otherwise. The settings are in
`MatchConfig` in [`src/onet/matching.py`](src/onet/matching.py).

## Layout

```
config/    the configuration template and the abbreviation table
scripts/   the pipeline stages, the cleaning check and the frame freezer
src/onet/  shared code: configuration, cleaning, embedding, matching, review frames
tests/     unit tests on synthetic data
```
