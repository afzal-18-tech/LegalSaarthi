"""Tests for src.ingestion.source_resolver module."""
import pytest

from src.ingestion.source_resolver import (
    derive_pdf_filename,
    _derive_from_path,
    SourceResolver,
    ResolutionResult,
)


class TestDerivePdfFilename:
    """Tests for PDF filename derivation from metadata fields."""

    def test_derive_from_structured_fields(self):
        """Should derive filename from year/volume/start/end fields."""
        row = {
            "year": 2020,
            "volume": 4,
            "start_page": 552,
            "end_page": 564,
        }
        result = derive_pdf_filename(row)
        assert result == "2020_4_552_564_EN.pdf"

    def test_derive_from_alternative_column_names(self):
        """Should handle alternative column names."""
        row = {
            "Year": 2019,
            "vol": 3,
            "page_from": 100,
            "page_to": 150,
        }
        result = derive_pdf_filename(row)
        assert result == "2019_3_100_150_EN.pdf"

    def test_derive_from_path_field(self):
        """Should derive from 'path' field when structured fields missing."""
        row = {
            "year": 2020,
            "path": "2020_4_552_564.json",
        }
        result = derive_pdf_filename(row)
        assert result == "2020_4_552_564_EN.pdf"

    def test_derive_from_filename_field(self):
        """Should derive from 'filename' field."""
        row = {
            "year": 2020,
            "filename": "2020_4_552_564_EN.pdf",
        }
        result = derive_pdf_filename(row)
        assert result == "2020_4_552_564_EN.pdf"

    def test_derive_missing_fields_returns_none(self):
        """Should return None when required fields are missing."""
        row = {"year": 2020}
        result = derive_pdf_filename(row)
        assert result is None

    def test_derive_float_values(self):
        """Should handle float values from parquet (e.g., 2020.0)."""
        row = {
            "year": 2020.0,
            "volume": 4.0,
            "start_page": 552.0,
            "end_page": 564.0,
        }
        result = derive_pdf_filename(row)
        assert result == "2020_4_552_564_EN.pdf"


class TestDeriveFromPath:
    """Tests for path-based filename derivation."""

    def test_full_s3_path(self):
        result = _derive_from_path(
            "data/pdf/year=2020/english/2020_4_552_564_EN.pdf"
        )
        assert result == "2020_4_552_564_EN.pdf"

    def test_json_filename(self):
        result = _derive_from_path("2020_4_552_564.json")
        assert result == "2020_4_552_564_EN.pdf"

    def test_already_pdf(self):
        result = _derive_from_path("2020_4_552_564_EN.pdf")
        assert result == "2020_4_552_564_EN.pdf"

    def test_unrecognized_returns_none(self):
        result = _derive_from_path("random_document.txt")
        assert result is None


class TestResolutionResult:
    """Tests for ResolutionResult data class."""

    def test_to_dict(self):
        r = ResolutionResult(
            case_id="2020_4_552_564",
            filename="2020_4_552_564_EN.pdf",
            url="https://example.com/test.pdf",
            method="index_verified",
            status="resolved",
            year=2020,
        )
        d = r.to_dict()
        assert d["case_id"] == "2020_4_552_564"
        assert d["status"] == "resolved"
        assert d["year"] == 2020
