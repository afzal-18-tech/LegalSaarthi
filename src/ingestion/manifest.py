"""
LegalSaarthi corpus manifest manager.

Creates, reads, updates, and validates the development manifest CSV
that tracks all metadata records and their acquisition status.
"""
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.ingestion.pdf_downloader import DownloadResult
from src.ingestion.source_resolver import ResolutionResult
from src.utils.logging_setup import get_logger

logger = get_logger("ingestion.manifest")

# Required manifest columns per RESEARCH_SPEC.md
MANIFEST_COLUMNS = [
    "case_id",
    "case_title",
    "petitioner",
    "respondent",
    "court",
    "decision_date",
    "year",
    "language",
    "source_url",
    "source_path",
    "local_path",
    "sha256",
    "acquisition_method",
    "acquisition_status",
    "accessed_at",
    "license_note",
    "filename",
    "file_size",
]


def create_manifest_from_metadata(
    metadata_df: pd.DataFrame,
    resolutions: list[ResolutionResult],
    downloads: list[DownloadResult] | None = None,
) -> pd.DataFrame:
    """Create a development manifest from metadata + resolution + download results.

    Args:
        metadata_df: Raw parquet metadata DataFrame.
        resolutions: List of ResolutionResult from source resolver.
        downloads: Optional list of DownloadResult from PDF downloader.

    Returns:
        DataFrame with MANIFEST_COLUMNS.
    """
    records = []

    # Index downloads by case_id for lookup
    download_map: dict[str, DownloadResult] = {}
    if downloads:
        for dl in downloads:
            download_map[dl.case_id] = dl

    for res in resolutions:
        dl = download_map.get(res.case_id)

        record = {
            "case_id": res.case_id,
            "case_title": "",
            "petitioner": "",
            "respondent": "",
            "court": "Supreme Court of India",
            "decision_date": "",
            "year": res.year,
            "language": "english",
            "source_url": res.url,
            "source_path": f"data/pdf/year={res.year}/english/{res.filename}",
            "local_path": dl.local_path if dl else "",
            "sha256": dl.sha256 if dl else "",
            "acquisition_method": res.method,
            "acquisition_status": dl.status if dl else res.status,
            "accessed_at": dl.accessed_at if dl else "",
            "license_note": "CC-BY-4.0",
            "filename": res.filename,
            "file_size": dl.file_size if dl else 0,
        }

        # Try to fill in case_title and other fields from metadata
        # Match by case_id pattern (year_vol_start_end)
        records.append(record)

    manifest_df = pd.DataFrame(records, columns=MANIFEST_COLUMNS)
    logger.info(f"Created manifest with {len(manifest_df)} records")
    return manifest_df


def enrich_manifest_from_metadata(
    manifest_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
) -> pd.DataFrame:
    """Enrich manifest with additional metadata from parquet (titles, dates, etc.).

    This performs a best-effort join on filename patterns.

    Args:
        manifest_df: Current manifest.
        metadata_df: Raw parquet metadata.

    Returns:
        Enriched manifest DataFrame.
    """
    # This will be refined once we see the actual parquet schema
    # For now, return as-is
    return manifest_df


def save_manifest(
    manifest_df: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Save manifest to CSV.

    Args:
        manifest_df: Manifest DataFrame.
        output_path: Path for CSV file.

    Returns:
        Path to saved file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(output_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
    logger.info(f"Saved manifest: {len(manifest_df)} records -> {output_path}")
    return output_path


def load_manifest(manifest_path: Path) -> pd.DataFrame:
    """Load manifest from CSV.

    Args:
        manifest_path: Path to manifest CSV.

    Returns:
        Manifest DataFrame.

    Raises:
        FileNotFoundError: If file doesn't exist.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)
    logger.info(f"Loaded manifest: {len(df)} records from {manifest_path}")
    return df


def save_download_failures(
    downloads: list[DownloadResult],
    output_path: Path,
) -> Path:
    """Save download failures to CSV.

    Args:
        downloads: List of all DownloadResult objects.
        output_path: Path for failures CSV.

    Returns:
        Path to saved file.
    """
    failures = [d for d in downloads if d.status == "failed"]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "case_id", "filename", "url", "error", "method",
        "attempts", "accessed_at",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
        writer.writeheader()
        for fail in failures:
            writer.writerow({
                "case_id": fail.case_id,
                "filename": fail.filename,
                "url": fail.url,
                "error": fail.error,
                "method": fail.method,
                "attempts": fail.attempts,
                "accessed_at": fail.accessed_at,
            })

    logger.info(f"Saved {len(failures)} download failures -> {output_path}")
    return output_path


def compute_acquisition_stats(
    downloads: list[DownloadResult],
) -> dict[str, Any]:
    """Compute acquisition statistics.

    Args:
        downloads: List of DownloadResult objects.

    Returns:
        Dictionary of stats.
    """
    total = len(downloads)
    success = sum(1 for d in downloads if d.status in ("success", "already_exists"))
    failed = sum(1 for d in downloads if d.status == "failed")
    skipped = sum(1 for d in downloads if d.status == "skipped")
    total_bytes = sum(d.file_size for d in downloads if d.file_size)

    return {
        "total": total,
        "success": success,
        "failed": failed,
        "skipped": skipped,
        "success_rate": success / total if total > 0 else 0.0,
        "total_bytes": total_bytes,
        "total_mb": round(total_bytes / (1024 * 1024), 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
