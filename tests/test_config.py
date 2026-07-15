import pytest

from ocrbench.config import Config

ENV_VARS = ("OCRBENCH_DOCAI_PROJECT_ID", "OCRBENCH_DOCAI_REGION", "OCRBENCH_DOCAI_PROCESSOR_ID")


@pytest.fixture(autouse=True)
def _clean_docai_env(monkeypatch):
    # Make sure no real environment leaks into these tests either way.
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def _cfg(docai_section):
    return Config({"docai": docai_section})


def test_falls_back_to_config_yaml_when_env_unset():
    cfg = _cfg({"project_id": "YOUR_GCP_PROJECT_ID", "processor_id": "YOUR_PROCESSOR_ID", "region": "us"})
    settings = cfg.docai_settings()
    assert settings == {
        "project_id": "YOUR_GCP_PROJECT_ID",
        "region": "us",
        "processor_id": "YOUR_PROCESSOR_ID",
    }


def test_env_vars_override_config_yaml(monkeypatch):
    monkeypatch.setenv("OCRBENCH_DOCAI_PROJECT_ID", "ocr-benchmark-502416")
    monkeypatch.setenv("OCRBENCH_DOCAI_REGION", "US")
    monkeypatch.setenv("OCRBENCH_DOCAI_PROCESSOR_ID", "234ea23afbe0364a")

    cfg = _cfg({"project_id": "YOUR_GCP_PROJECT_ID", "processor_id": "YOUR_PROCESSOR_ID", "region": "us"})
    settings = cfg.docai_settings()

    assert settings["project_id"] == "ocr-benchmark-502416"
    assert settings["processor_id"] == "234ea23afbe0364a"
    # region is lowercased regardless of how it was supplied
    assert settings["region"] == "us"


def test_partial_env_override_mixes_with_config_fallback(monkeypatch):
    # Only the project id is overridden; region/processor_id fall back to config.yaml.
    monkeypatch.setenv("OCRBENCH_DOCAI_PROJECT_ID", "ocr-benchmark-502416")

    cfg = _cfg({"project_id": "YOUR_GCP_PROJECT_ID", "processor_id": "234ea23afbe0364a", "region": "eu"})
    settings = cfg.docai_settings()

    assert settings["project_id"] == "ocr-benchmark-502416"
    assert settings["processor_id"] == "234ea23afbe0364a"
    assert settings["region"] == "eu"


def test_empty_env_var_treated_as_unset(monkeypatch):
    monkeypatch.setenv("OCRBENCH_DOCAI_PROJECT_ID", "")
    cfg = _cfg({"project_id": "YOUR_GCP_PROJECT_ID", "processor_id": "x", "region": "us"})
    assert cfg.docai_settings()["project_id"] == "YOUR_GCP_PROJECT_ID"


def test_missing_docai_section_yields_empty_strings():
    cfg = Config({})
    settings = cfg.docai_settings()
    assert settings == {"project_id": "", "region": "", "processor_id": ""}
