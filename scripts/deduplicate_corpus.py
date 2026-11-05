#!/usr/bin/env python3
"""
LegalSaarthi — Corpus Deduplication Script (Phase 5)

Removes duplicate records from the corpus manifest based on case_id and SHA-256.
Generates a clean_manifest.csv.

Usage:
    python scripts/deduplicate_corpus.py --manifest data/metadata/manifest_prototype.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.manifest import load_manifest, save_manifest
from src.utils.config import load_config
from src.utils.logging_setup import setup_logging, get_logger


def deduplicate_manifest(manifest_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Remove duplicate case_ids and duplicate content (SHA-256).
    """
    logger = get_logger("scripts.deduplicate_corpus")
    
    initial_count = len(manifest_df)
    
    # 1. Deduplicate by case_id
    # We keep the first occurrence.
    df_dedup_id = manifest_df.drop_duplicates(subset=["case_id"], keep="first")
    id_dupes_removed = initial_count - len(df_dedup_id)
    
    # 2. Deduplicate by SHA-256
    # Only consider records that have a sha256 (successfully downloaded)
    # Empty sha256 should not be dropped as duplicates of each other
    mask_has_sha = df_dedup_id["sha256"].notna() & (df_dedup_id["sha256"] != "")
    
    df_with_sha = df_dedup_id[mask_has_sha]
    df_without_sha = df_dedup_id[~mask_has_sha]
    
    df_sha_dedup = df_with_sha.drop_duplicates(subset=["sha256"], keep="first")
    sha_dupes_removed = len(df_with_sha) - len(df_sha_dedup)
    
    # Recombine
    clean_df = pd.concat([df_sha_dedup, df_without_sha], ignore_index=True)
    
    final_count = len(clean_df)
    
    stats = {
        "initial_records": initial_count,
        "final_records": final_count,
        "removed_duplicate_case_ids": id_dupes_removed,
        "removed_duplicate_content": sha_dupes_removed,
        "total_removed": initial_count - final_count
    }
    
    logger.info(f"Deduplication stats: {stats}")
    return clean_df, stats


def main():
    parser = argparse.ArgumentParser(description="Deduplicate corpus manifest")
    parser.add_argument("--manifest", type=str, default=None, help="Path to manifest CSV")
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config)
    logger = get_logger("scripts.deduplicate_corpus")

    if args.manifest:
        manifest_path = Path(args.manifest)
    else:
        # Auto-detect latest manifest
        meta_dir = PROJECT_ROOT / "data" / "metadata"
        manifests = sorted(meta_dir.glob("manifest_*.csv"))
        if not manifests:
            logger.error("No manifest found.")
            sys.exit(1)
        # Exclude clean and split manifests from auto-detect
        valid_manifests = [m for m in manifests if "clean" not in m.name and "split" not in m.name]
        if not valid_manifests:
             logger.error("No valid manifest found.")
             sys.exit(1)
        manifest_path = valid_manifests[-1]

    logger.info(f"Deduplicating manifest: {manifest_path}")
    manifest_df = load_manifest(manifest_path)

    clean_df, stats = deduplicate_manifest(manifest_df)
    
    clean_manifest_path = PROJECT_ROOT / "data" / "metadata" / "clean_manifest.csv"
    save_manifest(clean_df, clean_manifest_path)
    
    print(f"\n{'='*60}")
    print(f"DEDUPLICATION REPORT")
    print(f"{'='*60}")
    print(f"Initial records: {stats['initial_records']}")
    print(f"Final records:   {stats['final_records']}")
    print(f"Removed ID dupes:{stats['removed_duplicate_case_ids']}")
    print(f"Removed content dupes: {stats['removed_duplicate_content']}")
    print(f"{'='*60}")
    print(f"Clean manifest:  {clean_manifest_path}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
