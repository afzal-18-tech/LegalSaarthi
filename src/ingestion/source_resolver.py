"""
LegalSaarthi source resolver.

Maps metadata records to verified, downloadable PDF URLs by:
1. Deriving S3 object keys from metadata fields
2. Verifying against year index (english.index.json)
3. Constructing HTTPS download URLs
"""
from pathlib import Path
from typing import Any

import pandas as pd

from src.ingestion.metadata_client import (
    download_index_for_year,
    get_all_pdf_filenames_from_index,
)
from src.utils.logging_setup import get_logger

logger = get_logger("ingestion.resolver")

S3_BASE_URL = "https://indian-supreme-court-judgments.s3.ap-south-1.amazonaws.com"


class ResolutionResult:
    """Result of attempting to resolve a metadata record to a PDF URL."""

    __slots__ = (
        "case_id",
        "filename",
        "url",
        "method",
        "status",
        "error",
        "year",
    )

    def __init__(
        self,
        case_id: str,
        filename: str = "",
        url: str = "",
        method: str = "",
        status: str = "pending",
        error: str = "",
        year: int = 0,
    ):
        self.case_id = case_id
        self.filename = filename
        self.url = url
        self.method = method
        self.status = status
        self.error = error
        self.year = year

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "filename": self.filename,
            "url": self.url,
            "method": self.method,
            "status": self.status,
            "error": self.error,
            "year": self.year,
        }


def derive_pdf_filename(row: dict[str, Any] | pd.Series) -> str | None:
    """Derive the expected PDF filename from a metadata record.

    The naming convention in the S3 bucket is:
        {year}_{volume}_{start_page}_{end_page}_EN.pdf

    The metadata parquet may use various column names. This function
    tries common variants.

    Args:
        row: A dict or Series with metadata fields.

    Returns:
        Derived filename string, or None if fields are missing.
    """
    # Try to extract year
    year = None
    for col in ["year", "year_partition", "Year"]:
        if col in row and row[col] is not None:
            year = row[col]
            break

    # PRIORITY: Check 'path' field first — the actual parquet schema
    # uses a 'path' column like '2020_4_552_564' which directly encodes
    # the PDF filename pattern.
    for col in ["path", "filename", "file_name", "pdf_path", "s3_key"]:
        if col in row and row[col] is not None:
            result = _derive_from_path(str(row[col]), year)
            if result is not None:
                return result

    # Fallback: try to extract volume, start_page, end_page
    # The parquet may use 'volume', 'vol', 'scr_volume', etc.
    volume = None
    for col in ["volume", "vol", "scr_volume", "Volume"]:
        if col in row and row[col] is not None:
            volume = row[col]
            break

    start_page = None
    for col in ["start_page", "page_from", "startPage", "StartPage", "page_start"]:
        if col in row and row[col] is not None:
            start_page = row[col]
            break

    end_page = None
    for col in ["end_page", "page_to", "endPage", "EndPage", "page_end"]:
        if col in row and row[col] is not None:
            end_page = row[col]
            break

    if year is None or volume is None or start_page is None or end_page is None:
        return None

    # Convert to int where needed, handle possible float from parquet
    try:
        year = int(year)
        volume = int(volume)
        start_page = int(start_page)
        end_page = int(end_page)
    except (ValueError, TypeError):
        return None

    return f"{year}_{volume}_{start_page}_{end_page}_EN.pdf"


def _derive_from_path(path_value: str, year: int | None = None) -> str | None:
    """Try to extract or reconstruct PDF filename from a path field.

    Handles cases like:
    - 'data/pdf/year=2020/english/2020_4_552_564_EN.pdf'
    - '2020_4_552_564_EN.pdf'
    - '2020_4_552_564.json' -> '2020_4_552_564_EN.pdf'

    Args:
        path_value: Raw path/filename from metadata.
        year: Optional year for validation.

    Returns:
        Derived PDF filename, or None.
    """
    import re

    # Extract basename
    basename = path_value.rsplit("/", 1)[-1] if "/" in path_value else path_value

    # Already a valid PDF filename?
    if re.match(r"\d{4}_\d+_\d+_\d+_EN\.pdf$", basename):
        return basename

    # JSON metadata filename -> PDF
    m = re.match(r"(\d{4}_\d+_\d+_\d+)\.json$", basename)
    if m:
        return f"{m.group(1)}_EN.pdf"

    # Loose pattern: digits separated by underscores
    m = re.match(r"(\d{4})_(\d+)_(\d+)_(\d+)", basename)
    if m:
        return f"{m.group(0)}_EN.pdf"

    return None


class SourceResolver:
    """Resolves metadata records to downloadable S3 PDF URLs.

    Resolution strategy:
    1. Derive expected filename from metadata fields
    2. Verify filename exists in year's english.index.json
    3. Construct HTTPS URL
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.base_url = config.get("data_source", {}).get("base_url", S3_BASE_URL)
        self._index_cache: dict[int, set[str]] = {}

    def _get_year_index(self, year: int) -> set[str]:
        """Get the set of available PDF filenames for a year (cached)."""
        if year not in self._index_cache:
            try:
                index_data = download_index_for_year(year, self.config)
                self._index_cache[year] = get_all_pdf_filenames_from_index(index_data)
                logger.info(
                    f"Loaded index for year {year}: "
                    f"{len(self._index_cache[year])} files"
                )
            except Exception as e:
                logger.warning(f"Could not load index for year {year}: {e}")
                self._index_cache[year] = set()
        return self._index_cache[year]

    def resolve(self, row: dict[str, Any] | pd.Series) -> ResolutionResult:
        """Resolve a single metadata record to a download URL.

        Args:
            row: Metadata record (dict or Series).

        Returns:
            ResolutionResult with status 'resolved' or 'failed'.
        """
        # Build a case_id from the metadata
        case_id = self._extract_case_id(row)

        # Step 1: Derive expected filename
        filename = derive_pdf_filename(row)
        if filename is None:
            return ResolutionResult(
                case_id=case_id,
                status="failed",
                error="Could not derive PDF filename from metadata fields",
                method="derive",
            )

        # Step 2: Determine year
        year = self._extract_year(row)
        if year is None:
            return ResolutionResult(
                case_id=case_id,
                filename=filename,
                status="failed",
                error="Could not determine year",
                method="derive",
            )

        # Step 3: Verify against index
        index_files = self._get_year_index(year)
        method = "index_verified"
        if not index_files:
            # Index unavailable: the filename is only derived, not verified.
            method = "derived_unverified"
        elif filename not in index_files:
            # Try without _EN suffix or with variations
            alt = filename.replace("_EN.pdf", ".pdf")
            if alt in index_files:
                filename = alt
                method = "index_verified_alt"
            else:
                logger.debug(
                    f"Filename {filename} not in index for year {year}, "
                    f"proceeding with derived URL (unverified)"
                )
                method = "derived_unverified"

        # Step 4: Construct URL
        pdf_path_template = self.config.get("data_source", {}).get("paths", {}).get(
            "pdf_english", "data/pdf/year={year}/english/{filename}"
        )
        s3_key = pdf_path_template.format(year=year, filename=filename)
        url = f"{self.base_url}/{s3_key}"

        return ResolutionResult(
            case_id=case_id,
            filename=filename,
            url=url,
            method=method,
            status="resolved",
            year=year,
        )

    def resolve_batch(
        self, df: pd.DataFrame, limit: int | None = None
    ) -> list[ResolutionResult]:
        """Resolve a batch of metadata records.

        Args:
            df: DataFrame of metadata records.
            limit: Optional max number of records to resolve.

        Returns:
            List of ResolutionResult objects.
        """
        results = []
        rows = df.head(limit) if limit else df
        total = len(rows)

        for idx, (_, row) in enumerate(rows.iterrows()):
            result = self.resolve(row)
            results.append(result)
            if (idx + 1) % 100 == 0:
                resolved = sum(1 for r in results if r.status == "resolved")
                logger.info(
                    f"Resolved {idx + 1}/{total}: {resolved} success, "
                    f"{idx + 1 - resolved} failed"
                )

        resolved = sum(1 for r in results if r.status == "resolved")
        failed = sum(1 for r in results if r.status == "failed")
        logger.info(
            f"Resolution complete: {resolved} resolved, {failed} failed "
            f"out of {len(results)} total"
        )
        return results

    @staticmethod
    def _extract_case_id(row: dict[str, Any] | pd.Series) -> str:
        """Extract or derive a case_id from metadata fields."""
        for col in ["case_id", "caseId", "id", "ID"]:
            if col in row and row[col] is not None:
                return str(row[col])

        # Derive from filename pattern
        filename = derive_pdf_filename(row)
        if filename:
            return filename.replace("_EN.pdf", "").replace(".pdf", "")

        # Fallback: use case_title or first available field
        for col in ["case_title", "title", "case_name"]:
            if col in row and row[col] is not None:
                return str(row[col])[:100]

        return "unknown"

    @staticmethod
    def _extract_year(row: dict[str, Any] | pd.Series) -> int | None:
        """Extract year from metadata fields."""
        for col in ["year", "year_partition", "Year"]:
            if col in row and row[col] is not None:
                try:
                    return int(row[col])
                except (ValueError, TypeError):
                    continue
        return None
