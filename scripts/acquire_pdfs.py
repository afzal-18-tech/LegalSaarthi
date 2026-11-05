#!/usr/bin/env python3
"""
LegalSaarthi — PDF Acquisition Script

Resolves metadata to S3 URLs and downloads PDFs in staged gates.

Usage:
    python scripts/acquire_pdfs.py --stage smoke   # 10 documents
    python scripts/acquire_pdfs.py --stage pilot   # 50 documents
    python scripts/acquire_pdfs.py --stage prototype  # 200 documents
    python scripts/acquire_pdfs.py --stage main    # 1000 documents
    python scripts/acquire_pdfs.py --limit 1       # single document test
"""
import argparse
import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.metadata_client import acquire_metadata_for_years
from src.ingestion.source_resolver import SourceResolver
from src.ingestion.pdf_downloader import PDFDownloader
from src.ingestion.manifest import (
    create_manifest_from_metadata,
    save_manifest,
    save_download_failures,
    compute_acquisition_stats,
)
from src.utils.config import load_config
from src.utils.logging_setup import setup_logging, get_logger


def main():
    parser = argparse.ArgumentParser(
        description="Download Indian SC judgment PDFs with staged gates"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "default.yaml"),
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["smoke", "pilot", "prototype", "main"],
        default=None,
        help="Acquisition stage (smoke=10, pilot=50, prototype=200, main=1000)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Override: exact number of documents to download",
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
    parser.add_argument(
        "--skip-metadata",
        action="store_true",
        help="Skip metadata download (use cached)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config)
    logger = get_logger("scripts.acquire_pdfs")

    # Determine target count
    if args.limit is not None:
        target = args.limit
        stage_name = f"custom_{target}"
    elif args.stage:
        target = config["stages"][args.stage]["target"]
        stage_name = args.stage
    else:
        target = 10  # default to smoke
        stage_name = "smoke"

    min_success_rate = config["stages"].get(
        stage_name, config["stages"]["smoke"]
    ).get("min_success_rate", 0.90)

    year_start = args.year_start or config["scope"]["year_start"]
    year_end = args.year_end or config["scope"]["year_end"]

    logger.info(f"Stage: {stage_name} | Target: {target} documents")
    logger.info(f"Year range: {year_start}–{year_end}")

    # Step 1: Acquire metadata
    logger.info("Step 1: Acquiring metadata...")
    metadata_df = acquire_metadata_for_years(year_start, year_end, config)
    if metadata_df.empty:
        logger.error("No metadata acquired. Exiting.")
        sys.exit(1)
    logger.info(f"Metadata: {len(metadata_df)} records")

    # Step 2: Resolve to download URLs
    logger.info("Step 2: Resolving source URLs...")
    resolver = SourceResolver(config)
    resolutions = resolver.resolve_batch(metadata_df, limit=target)

    resolved_count = sum(1 for r in resolutions if r.status == "resolved")
    failed_count = sum(1 for r in resolutions if r.status == "failed")
    logger.info(f"Resolution: {resolved_count} resolved, {failed_count} failed")

    if resolved_count == 0:
        logger.error("No resolutions succeeded. Cannot download. Exiting.")
        sys.exit(1)

    # Step 3: Download PDFs
    logger.info("Step 3: Downloading PDFs...")
    output_dir = PROJECT_ROOT / "data" / "raw"
    downloader = PDFDownloader(config)
    downloads = downloader.download_batch(resolutions, output_dir, limit=target)

    # Step 4: Compute stats
    stats = compute_acquisition_stats(downloads)
    logger.info(f"Acquisition stats: {json.dumps(stats, indent=2)}")

    # Step 5: Save manifest
    manifest_df = create_manifest_from_metadata(metadata_df, resolutions, downloads)
    manifest_path = PROJECT_ROOT / "data" / "metadata" / f"manifest_{stage_name}.csv"
    save_manifest(manifest_df, manifest_path)

    # Step 6: Save failures
    failures_path = PROJECT_ROOT / "data" / "metadata" / f"download_failures_{stage_name}.csv"
    save_download_failures(downloads, failures_path)

    # Step 7: Save stats
    stats_path = PROJECT_ROOT / "data" / "metadata" / f"acquisition_stats_{stage_name}.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    # Gate check
    gate_pass = stats["success_rate"] >= min_success_rate

    print(f"\n{'='*60}")
    print(f"ACQUISITION REPORT — Stage: {stage_name}")
    print(f"{'='*60}")
    print(f"Target:       {target}")
    print(f"Resolved:     {resolved_count}")
    print(f"Downloaded:   {stats['success']}")
    print(f"Failed:       {stats['failed']}")
    print(f"Skipped:      {stats['skipped']}")
    print(f"Success rate: {stats['success_rate']:.1%}")
    print(f"Total size:   {stats['total_mb']} MB")
    print(f"Min required: {min_success_rate:.0%}")
    print(f"{'='*60}")
    print(f"GATE: {'PASS ✓' if gate_pass else 'FAIL ✗'}")
    print(f"{'='*60}")
    print(f"Manifest:  {manifest_path}")
    print(f"Failures:  {failures_path}")
    print(f"Stats:     {stats_path}")
    print(f"{'='*60}")

    sys.exit(0 if gate_pass else 1)


if __name__ == "__main__":
    main()
