"""Tests for src.ingestion.manifest module."""
import csv
from pathlib import Path

import pandas as pd
import pytest

from src.ingestion.manifest import (
    create_manifest_from_metadata,
    save_manifest,
    load_manifest,
    save_download_failures,
    compute_acquisition_stats,
    MANIFEST_COLUMNS,
)
from src.ingestion.source_resolver import ResolutionResult
from src.ingestion.pdf_downloader import DownloadResult


class TestCreateManifest:
    """Tests for manifest creation."""

    def test_create_from_resolutions(self):
        """Should create manifest from resolution results."""
        metadata_df = pd.DataFrame({"year_partition": [2020]})
        resolutions = [
            ResolutionResult(
                case_id="2020_4_552_564",
                filename="2020_4_552_564_EN.pdf",
                url="https://example.com/test.pdf",
                method="index_verified",
                status="resolved",
                year=2020,
            )
        ]
        manifest = create_manifest_from_metadata(metadata_df, resolutions)
        assert len(manifest) == 1
        assert manifest.iloc[0]["case_id"] == "2020_4_552_564"
        assert manifest.iloc[0]["license_note"] == "CC-BY-4.0"

    def test_manifest_has_required_columns(self):
        """Manifest should contain all required columns."""
        metadata_df = pd.DataFrame({"year_partition": [2020]})
        resolutions = [
            ResolutionResult(
                case_id="test",
                filename="test.pdf",
                status="resolved",
                year=2020,
            )
        ]
        manifest = create_manifest_from_metadata(metadata_df, resolutions)
        for col in MANIFEST_COLUMNS:
            assert col in manifest.columns, f"Missing column: {col}"


class TestSaveLoadManifest:
    """Tests for manifest persistence."""

    def test_save_and_load(self, tmp_path):
        """Should round-trip through CSV."""
        df = pd.DataFrame({
            col: ["test_value"] for col in MANIFEST_COLUMNS
        })
        path = tmp_path / "manifest.csv"
        save_manifest(df, path)

        loaded = load_manifest(path)
        assert len(loaded) == 1
        assert loaded.iloc[0]["case_id"] == "test_value"

    def test_load_nonexistent(self, tmp_path):
        """Should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_manifest(tmp_path / "nonexistent.csv")


class TestDownloadFailures:
    """Tests for failure logging."""

    def test_save_failures(self, tmp_path):
        """Should save only failed downloads."""
        downloads = [
            DownloadResult(case_id="ok", status="success"),
            DownloadResult(
                case_id="bad",
                status="failed",
                error="HTTP 404",
                filename="bad.pdf",
            ),
        ]
        path = tmp_path / "failures.csv"
        save_download_failures(downloads, path)

        with open(path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["case_id"] == "bad"
        assert "404" in rows[0]["error"]


class TestAcquisitionStats:
    """Tests for statistics computation."""

    def test_compute_stats(self):
        downloads = [
            DownloadResult(status="success", file_size=1000),
            DownloadResult(status="success", file_size=2000),
            DownloadResult(status="failed"),
            DownloadResult(status="skipped"),
        ]
        stats = compute_acquisition_stats(downloads)
        assert stats["total"] == 4
        assert stats["success"] == 2
        assert stats["failed"] == 1
        assert stats["skipped"] == 1
        assert stats["success_rate"] == 0.5
        assert stats["total_bytes"] == 3000
