from pathlib import Path

import pytest

from src.config.profiles import resolve_config_path, validate_config_environment


def test_default_config_profile_is_local(monkeypatch):
    monkeypatch.delenv("MODEL_CONFIG_PROFILE", raising=False)
    monkeypatch.delenv("MIDAS_CONFIG_PATH", raising=False)

    assert resolve_config_path().name == "config.yaml"


def test_hosted_openai_profile_resolves(monkeypatch):
    monkeypatch.setenv("MODEL_CONFIG_PROFILE", "hosted-openai")
    monkeypatch.delenv("MIDAS_CONFIG_PATH", raising=False)

    assert resolve_config_path().name == "config.hosted-openai.yaml"


def test_explicit_config_path_overrides_profile(monkeypatch, tmp_path):
    config_path = tmp_path / "custom.yaml"
    monkeypatch.setenv("MIDAS_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("MODEL_CONFIG_PROFILE", "hosted-openai")

    assert resolve_config_path() == config_path


def test_hosted_openai_requires_openai_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is required"):
        validate_config_environment(Path("src/config/config.hosted-openai.yaml"))


def test_local_profile_does_not_require_openai_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    validate_config_environment(Path("src/config/config.yaml"))
