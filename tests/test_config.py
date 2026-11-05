"""Tests for src.utils.config module."""
import pytest
from pathlib import Path

from src.utils.config import load_config, get_project_root, resolve_path


class TestLoadConfig:
    """Tests for configuration loading."""

    def test_load_default_config(self):
        """Default config should load successfully."""
        config = load_config()
        assert isinstance(config, dict)
        assert "project" in config
        assert config["project"]["name"] == "LegalSaarthi"

    def test_load_config_has_data_source(self):
        """Config should contain data_source section."""
        config = load_config()
        assert "data_source" in config
        assert "base_url" in config["data_source"]
        assert "indian-supreme-court-judgments" in config["data_source"]["base_url"]

    def test_load_config_has_scope(self):
        """Config should contain scope with year range."""
        config = load_config()
        assert "scope" in config
        assert config["scope"]["year_start"] == 2015
        assert config["scope"]["year_end"] == 2025

    def test_load_config_has_stages(self):
        """Config should contain acquisition stages."""
        config = load_config()
        assert "stages" in config
        assert config["stages"]["smoke"]["target"] == 10
        assert config["stages"]["pilot"]["target"] == 50
        assert config["stages"]["main"]["target"] == 1000

    def test_load_config_missing_file(self):
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")

    def test_load_config_has_acquisition(self):
        """Config should have acquisition settings."""
        config = load_config()
        assert "acquisition" in config
        assert config["acquisition"]["rate_limit_rps"] > 0
        assert config["acquisition"]["max_retries"] > 0


class TestGetProjectRoot:
    """Tests for project root detection."""

    def test_project_root_exists(self):
        root = get_project_root()
        assert root.exists()

    def test_project_root_has_configs(self):
        root = get_project_root()
        assert (root / "configs" / "default.yaml").exists()


class TestResolvePath:
    """Tests for path resolution."""

    def test_resolve_raw_pdfs(self):
        config = load_config()
        path = resolve_path(config, "paths.raw_pdfs")
        assert path.is_absolute()
        assert str(path).endswith("data/raw") or str(path).endswith("data\\raw")
