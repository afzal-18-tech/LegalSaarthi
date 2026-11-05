#!/usr/bin/env python3
"""
LegalSaarthi — Metadata Acquisition Script

Downloads parquet metadata from AWS S3 for the configured year range,
normalizes fields, and generates the development manifest.

Usage:
    python scripts/acquire_metadata.py [--config configs/default.yaml]
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.metadata_client import acquire_metadata_for_years
from src.ingestion.manifest import save_manifest
from src.utils.config import load_config
from src.utils.logging_setup import setup_logging, get_logger


def main():
    parser = argparse.ArgumentParser(
        description="Download and process metadata for Indian SC judgments"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "default.yaml"),
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--year-start",
        type=int,
        default=None,
        help="Override start year",
    )
    parser.add_argument(
        "--year-end",
        type=int,
        default=None,
        help="Override end year",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config)
    logger = get_logger("scripts.acquire_metadata")

    year_start = args.year_start or config["scope"]["year_start"]
    year_end = args.year_end or config["scope"]["year_end"]

    logger.info(f"Acquiring metadata for years {year_start}–{year_end}")

    # Download and combine parquet metadata
    metadata_df = acquire_metadata_for_years(year_start, year_end, config)

    if metadata_df.empty:
        logger.error("No metadata acquired. Exiting.")
        sys.exit(1)

    # Log schema info
    logger.info(f"Metadata columns: {list(metadata_df.columns)}")
    logger.info(f"Metadata dtypes:\n{metadata_df.dtypes}")
    logger.info(f"Total records: {len(metadata_df)}")
    for year in range(year_start, year_end + 1):
        year_count = len(metadata_df[metadata_df["year_partition"] == year])
        if year_count > 0:
            logger.info(f"  Year {year}: {year_count} records")

    # Save raw metadata
    metadata_dir = PROJECT_ROOT / "data" / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    raw_path = metadata_dir / "raw_metadata.csv"
    metadata_df.to_csv(raw_path, index=False)
    logger.info(f"Saved raw metadata: {raw_path}")

    # Print sample
    logger.info(f"Sample record:\n{metadata_df.iloc[0].to_dict()}")

    print(f"\n{'='*60}")
    print(f"Metadata acquisition complete")
    print(f"Years: {year_start}–{year_end}")
    print(f"Total records: {len(metadata_df)}")
    print(f"Columns: {list(metadata_df.columns)}")
    print(f"Saved to: {raw_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
