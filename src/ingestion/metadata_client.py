"""
LegalSaarthi metadata client.

Downloads and parses parquet metadata from the AWS S3 public bucket
for Indian Supreme Court judgments.
"""
import io
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
import requests

from src.utils.config import get_project_root
from src.utils.logging_setup import get_logger

logger = get_logger("ingestion.metadata")

# S3 base URL for the public bucket
S3_BASE_URL = "https://indian-supreme-court-judgments.s3.ap-south-1.amazonaws.com"


def download_parquet_for_year(
    year: int,
    config: dict[str, Any],
    cache_dir: Path | None = None,
) -> Path:
    """Download the parquet metadata file for a given year.

    Args:
        year: Year to download metadata for (e.g. 2020).
        config: Configuration dictionary.
        cache_dir: Directory to cache parquet files. Defaults to data/metadata/.

    Returns:
        Path to the downloaded parquet file.

    Raises:
        requests.HTTPError: If download fails.
    """
    base_url = config.get("data_source", {}).get("base_url", S3_BASE_URL)
    template = config.get("data_source", {}).get("paths", {}).get(
        "metadata_parquet", "metadata/parquet/year={year}/metadata.parquet"
    )
    url = f"{base_url}/{template.format(year=year)}"

    if cache_dir is None:
        cache_dir = get_project_root() / "data" / "metadata" / "parquet"
    cache_dir.mkdir(parents=True, exist_ok=True)

    local_path = cache_dir / f"year={year}" / "metadata.parquet"
    local_path.parent.mkdir(parents=True, exist_ok=True)

    if local_path.exists():
        logger.info(f"Using cached parquet for year {year}: {local_path}")
        return local_path

    logger.info(f"Downloading metadata parquet for year {year} from {url}")
    timeout = config.get("acquisition", {}).get("timeout_seconds", 60)
    user_agent = config.get("acquisition", {}).get(
        "user_agent", "LegalSaarthi-Research/0.1"
    )

    resp = requests.get(url, timeout=timeout, headers={"User-Agent": user_agent})
    resp.raise_for_status()

    with open(local_path, "wb") as f:
        f.write(resp.content)

    logger.info(
        f"Downloaded parquet for year {year}: {len(resp.content)} bytes -> {local_path}"
    )
    return local_path


def load_parquet_metadata(parquet_path: Path) -> pd.DataFrame:
    """Load a parquet metadata file into a DataFrame.

    Args:
        parquet_path: Path to parquet file.

    Returns:
        DataFrame with metadata columns.
    """
    table = pq.read_table(parquet_path)
    df = table.to_pandas()
    logger.info(f"Loaded {len(df)} records from {parquet_path}")
    return df


def download_index_for_year(
    year: int,
    config: dict[str, Any],
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    """Download the english.index.json for a given year.

    This index lists all PDF filenames available in the TAR archive
    and can be used to verify that a PDF exists before downloading.

    Args:
        year: Year to download index for.
        config: Configuration dictionary.
        cache_dir: Cache directory.

    Returns:
        Parsed JSON index dict.

    Raises:
        requests.HTTPError: If download fails.
    """
    import json

    base_url = config.get("data_source", {}).get("base_url", S3_BASE_URL)
    template = config.get("data_source", {}).get("paths", {}).get(
        "index_english", "data/tar/year={year}/english/english.index.json"
    )
    url = f"{base_url}/{template.format(year=year)}"

    if cache_dir is None:
        cache_dir = get_project_root() / "data" / "metadata" / "indices"
    cache_dir.mkdir(parents=True, exist_ok=True)

    local_path = cache_dir / f"year={year}" / "english.index.json"
    local_path.parent.mkdir(parents=True, exist_ok=True)

    if local_path.exists():
        logger.info(f"Using cached index for year {year}: {local_path}")
        with open(local_path, "r", encoding="utf-8") as f:
            return json.load(f)

    logger.info(f"Downloading index for year {year} from {url}")
    timeout = config.get("acquisition", {}).get("timeout_seconds", 60)
    user_agent = config.get("acquisition", {}).get(
        "user_agent", "LegalSaarthi-Research/0.1"
    )

    resp = requests.get(url, timeout=timeout, headers={"User-Agent": user_agent})
    resp.raise_for_status()

    with open(local_path, "wb") as f:
        f.write(resp.content)

    data = resp.json()
    logger.info(
        f"Downloaded index for year {year}: {data.get('file_count', '?')} files"
    )
    return data


def get_all_pdf_filenames_from_index(index_data: dict[str, Any]) -> set[str]:
    """Extract all PDF filenames from an english.index.json.

    Args:
        index_data: Parsed index JSON.

    Returns:
        Set of PDF filenames.
    """
    filenames = set()
    for part in index_data.get("parts", []):
        for f in part.get("files", []):
            filenames.add(f)
    return filenames


def acquire_metadata_for_years(
    year_start: int,
    year_end: int,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Download and combine parquet metadata for a range of years.

    Args:
        year_start: First year (inclusive).
        year_end: Last year (inclusive).
        config: Configuration dictionary.

    Returns:
        Combined DataFrame with all metadata records.
    """
    frames = []
    for year in range(year_start, year_end + 1):
        try:
            parquet_path = download_parquet_for_year(year, config)
            df = load_parquet_metadata(parquet_path)
            df["year_partition"] = year
            frames.append(df)
            logger.info(f"Year {year}: {len(df)} records")
        except requests.HTTPError as e:
            logger.warning(f"Failed to download metadata for year {year}: {e}")
        except Exception as e:
            logger.error(f"Error processing year {year}: {e}")

    if not frames:
        logger.error("No metadata acquired for any year")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    logger.info(f"Total metadata records: {len(combined)} across {len(frames)} years")
    return combined
