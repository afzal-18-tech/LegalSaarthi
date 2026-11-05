"""Tests for src.ingestion.pdf_downloader module."""
import tempfile
from pathlib import Path

import pytest

from src.ingestion.pdf_downloader import validate_pdf, DownloadResult


class TestValidatePdf:
    """Tests for PDF validation."""

    def test_valid_pdf(self, tmp_path):
        """Should accept a file with valid PDF signature and size."""
        pdf = tmp_path / "test.pdf"
        # Create a minimal fake PDF (valid signature, > 1KB)
        content = b"%PDF-1.4 " + b"x" * 2000
        pdf.write_bytes(content)

        is_valid, err = validate_pdf(pdf)
        assert is_valid is True
        assert err == ""

    def test_invalid_signature(self, tmp_path):
        """Should reject a file without PDF signature."""
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"This is not a PDF" + b"x" * 2000)

        is_valid, err = validate_pdf(pdf)
        assert is_valid is False
        assert "Invalid PDF signature" in err

    def test_file_too_small(self, tmp_path):
        """Should reject a file smaller than min_size."""
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-1.4 tiny")

        is_valid, err = validate_pdf(pdf, min_size=1024)
        assert is_valid is False
        assert "too small" in err

    def test_nonexistent_file(self, tmp_path):
        """Should reject a nonexistent file."""
        pdf = tmp_path / "does_not_exist.pdf"

        is_valid, err = validate_pdf(pdf)
        assert is_valid is False
        assert "does not exist" in err


class TestDownloadResult:
    """Tests for DownloadResult data class."""

    def test_to_dict(self):
        result = DownloadResult(
            case_id="test_case",
            filename="test.pdf",
            status="success",
            file_size=12345,
            sha256="abc123",
        )
        d = result.to_dict()
        assert d["case_id"] == "test_case"
        assert d["status"] == "success"
        assert d["file_size"] == 12345

    def test_default_values(self):
        result = DownloadResult()
        assert result.status == "pending"
        assert result.attempts == 0
        assert result.file_size == 0
