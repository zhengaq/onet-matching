import pytest
import yaml

from onet.config import DATA_ROOT_ENV, PIPELINE_TEMPLATE, load_settings


def write_config(path, root, **changes):
    cfg = yaml.safe_load(PIPELINE_TEMPLATE.read_text())
    cfg["data_root"] = str(root)
    cfg.update(changes)
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_template_paths_resolve_under_the_data_root(tmp_path, monkeypatch):
    monkeypatch.delenv(DATA_ROOT_ENV, raising=False)
    settings = load_settings(write_config(tmp_path / "pipeline.yaml", tmp_path))
    assert settings.data_root == tmp_path
    assert settings.output_dir == tmp_path / "path/for/results/and/worksheets"
    assert settings.output_file("report.md").name == "survey_onet_report.md"
    assert settings.survey_clean.parent == tmp_path / "path/for/cleaned/inputs"
    assert settings.review_frame is None


def test_environment_overrides_data_root(tmp_path, monkeypatch):
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.setenv(DATA_ROOT_ENV, str(other))
    settings = load_settings(write_config(tmp_path / "pipeline.yaml", tmp_path))
    assert settings.data_root == other


def test_missing_config_and_sections_are_errors(tmp_path, monkeypatch):
    monkeypatch.delenv(DATA_ROOT_ENV, raising=False)
    with pytest.raises(SystemExit):
        load_settings(tmp_path / "absent.yaml")
    path = write_config(tmp_path / "pipeline.yaml", tmp_path)
    cfg = yaml.safe_load(path.read_text())
    del cfg["paths"]
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(SystemExit):
        load_settings(path)
