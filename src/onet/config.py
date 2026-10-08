"""Repository paths and the local pipeline configuration.

Participant data is kept outside the repository, in a data root. Its
location, the input files, the survey variables and the directories the
pipeline writes to are set in config/pipeline.yaml, which is not tracked;
config/pipeline.example.yaml is the template. ONET_DATA_ROOT, if set,
overrides the configured data root. Nothing here touches the filesystem at
import time.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
PIPELINE_CONFIG = CONFIG_DIR / "pipeline.yaml"
PIPELINE_TEMPLATE = CONFIG_DIR / "pipeline.example.yaml"
DATA_ROOT_ENV = "ONET_DATA_ROOT"

PATH_KEYS = ("intermediate", "outputs", "cache", "logs")


@dataclass(frozen=True)
class Settings:
    data_root: Path
    survey_name: str
    survey_file: Path
    columns: dict  # role -> survey variable: id, job_title, industry, duties
    onet_alternate_titles: Path
    onet_reported_titles: Path
    intermediate_dir: Path  # cleaned inputs
    output_dir: Path  # results and reviewer worksheets
    cache_dir: Path  # embedding model and embeddings
    log_dir: Path  # embedding logs
    review_frame: Path | None = None  # frozen worksheet stage 04 builds from

    @property
    def onet_docs(self):
        return self.intermediate_dir / "onet_docs.csv"

    @property
    def survey_clean(self):
        return self.intermediate_dir / f"{self.survey_name}_clean.csv"

    @property
    def onet_embeddings(self):
        return self.cache_dir / "onet_embeddings.npz"

    @property
    def survey_embeddings(self):
        return self.cache_dir / f"{self.survey_name}_job_embeddings.npz"

    def output_file(self, kind):
        """<output dir>/<survey>_onet_<kind>, e.g. kind="review_queue.csv"."""
        return self.output_dir / f"{self.survey_name}_onet_{kind}"


def _data_root(cfg, config_path):
    value, source = os.environ.get(DATA_ROOT_ENV, "").strip(), DATA_ROOT_ENV
    if not value:
        value, source = str(cfg.get("data_root") or "").strip(), f"data_root in {config_path}"
    if not value:
        raise SystemExit(f"No data root: set data_root in {config_path} or {DATA_ROOT_ENV}.")
    root = Path(value).expanduser()
    if not root.is_absolute():
        root = (PROJECT_ROOT / root).resolve()
    if not root.is_dir():
        raise SystemExit(f"Data root {root} (from {source}) is not a directory.")
    return root


def load_settings(config_path=PIPELINE_CONFIG):
    config_path = Path(config_path)
    if not config_path.exists():
        raise SystemExit(f"{config_path} not found. Copy {PIPELINE_TEMPLATE.name} to "
                         f"{config_path.name} in {config_path.parent} and fill it in.")
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}
    missing = {"survey", "onet", "paths"} - cfg.keys()
    if missing:
        raise SystemExit(f"{config_path}: missing sections {sorted(missing)}")
    root = _data_root(cfg, config_path)
    survey, onet, paths = cfg["survey"], cfg["onet"], cfg["paths"]
    columns = dict(survey["columns"])
    missing = {"id", "job_title", "industry", "duties"} - columns.keys()
    if missing:
        raise SystemExit(f"{config_path}: survey.columns is missing {sorted(missing)}")
    missing = set(PATH_KEYS) - paths.keys()
    if missing:
        raise SystemExit(f"{config_path}: paths is missing {sorted(missing)}")
    frame = (cfg.get("review") or {}).get("frame")
    return Settings(
        data_root=root,
        survey_name=survey["name"],
        survey_file=root / survey["file"],
        columns=columns,
        onet_alternate_titles=root / onet["alternate_titles"],
        onet_reported_titles=root / onet["reported_titles"],
        intermediate_dir=root / paths["intermediate"],
        output_dir=root / paths["outputs"],
        cache_dir=root / paths["cache"],
        log_dir=root / paths["logs"],
        review_frame=root / frame if frame else None,
    )


def require(*paths):
    """Exit with a clear message if an expected input is absent."""
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        raise SystemExit("Missing input:\n  " + "\n  ".join(missing))
