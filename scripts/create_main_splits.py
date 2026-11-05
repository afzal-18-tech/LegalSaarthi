#!/usr/bin/env python3
"""
LegalSaarthi — Main Corpus Incremental Split Script (Phase 19).

Deduplicates manifest_main.csv and assigns splits while strictly freezing
the original 198 prototype split assignments (dev/val/test). New cases are
stratified by year with dev=0.70, val=0.15, test=0.15.

Usage:
    python scripts/create_main_splits.py
"""
import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.create_splits import create_stratified_split
from scripts.deduplicate_corpus import deduplicate_manifest
from src.ingestion.manifest import load_manifest, save_manifest
from src.utils.config import load_config
from src.utils.logging_setup import get_logger, setup_logging

logger = get_logger("scripts.create_main_splits")


def create_incremental_splits(
    clean_manifest: pd.DataFrame,
    existing_split_df: pd.DataFrame,
    dev_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    stratify_col: str = "year",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Assigns splits incrementally:
    - Retains exact split assignments for all cases in existing_split_df.
    - Performs stratified split on new cases only.
    """
    frozen_map = dict(zip(existing_split_df["case_id"], existing_split_df["split"]))
    
    # Partition clean manifest into frozen core vs new cases
    is_frozen = clean_manifest["case_id"].isin(frozen_map)
    frozen_part = clean_manifest[is_frozen].copy()
    new_part = clean_manifest[~is_frozen].copy()
    
    frozen_part["split"] = frozen_part["case_id"].map(frozen_map)
    
    logger.info(f"Preserving {len(frozen_part)} frozen core split assignments.")
    logger.info(f"Splitting {len(new_part)} new cases across years (seed={seed}).")
    
    if len(new_part) > 0:
        new_split_part = create_stratified_split(
            new_part,
            dev_ratio=dev_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
            stratify_col=stratify_col,
        )
    else:
        new_split_part = new_part.assign(split="")
        
    combined = pd.concat([frozen_part, new_split_part], ignore_index=True)
    
    stats = {
        "total_records": len(combined),
        "frozen_core_count": len(frozen_part),
        "new_cases_count": len(new_split_part),
        "split_counts": combined["split"].value_counts().to_dict(),
        "frozen_split_counts": frozen_part["split"].value_counts().to_dict(),
        "new_split_counts": new_split_part["split"].value_counts().to_dict(),
    }
    return combined, stats


def main():
    parser = argparse.ArgumentParser(description="Create incremental splits for Phase 19 main corpus")
    parser.add_argument("--manifest", type=str, default="data/metadata/manifest_main.csv")
    parser.add_argument("--frozen-splits", type=str, default="data/metadata/split_manifest.csv")
    parser.add_argument("--dev", type=float, default=0.70)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config)

    manifest_path = PROJECT_ROOT / args.manifest
    frozen_splits_path = PROJECT_ROOT / args.frozen_splits
    
    logger.info(f"Loading main manifest: {manifest_path}")
    raw_manifest = load_manifest(manifest_path)
    
    # 1. Deduplicate
    logger.info(f"Deduplicating main corpus manifest ({len(raw_manifest)} records)...")
    clean_df, dedup_stats = deduplicate_manifest(raw_manifest)
    
    clean_path = PROJECT_ROOT / "data" / "metadata" / "clean_manifest_main.csv"
    save_manifest(clean_df, clean_path)
    logger.info(f"Clean main manifest saved to {clean_path} ({len(clean_df)} records)")
    
    # 2. Backup prototype split manifest if not already backed up
    proto_backup = PROJECT_ROOT / "data" / "metadata" / "split_manifest_prototype.csv"
    if not proto_backup.exists() and frozen_splits_path.exists():
        shutil.copy2(frozen_splits_path, proto_backup)
        logger.info(f"Backed up prototype split manifest to {proto_backup}")
        
    frozen_df = load_manifest(frozen_splits_path)
    
    # 3. Create incremental splits
    split_df, split_stats = create_incremental_splits(
        clean_manifest=clean_df,
        existing_split_df=frozen_df,
        dev_ratio=args.dev,
        val_ratio=args.val,
        test_ratio=args.test,
        seed=args.seed,
    )
    
    # 4. Save results
    main_split_path = PROJECT_ROOT / "data" / "metadata" / "split_manifest_main.csv"
    save_manifest(split_df, main_split_path)
    
    # Also update split_manifest.csv to point to the main corpus
    save_manifest(split_df, frozen_splits_path)
    
    stats_output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "deduplication": dedup_stats,
        "splits": split_stats,
        "by_year": {},
    }
    if "year" in split_df.columns:
        for year_val in sorted(split_df["year"].unique()):
            yg = split_df[split_df["year"] == year_val]
            stats_output["by_year"][str(year_val)] = yg["split"].value_counts().to_dict()
            
    stats_path = PROJECT_ROOT / "data" / "metadata" / "split_stats_main.json"
    stats_path.write_text(json.dumps(stats_output, indent=2), encoding="utf-8")
    
    print("\n" + "=" * 60)
    print("MAIN CORPUS DEDUPLICATION & SPLIT REPORT (Phase 19)")
    print("=" * 60)
    print(f"Raw records:          {dedup_stats['initial_records']}")
    print(f"Clean records:        {dedup_stats['final_records']}")
    print(f"Removed duplicates:   {dedup_stats['total_removed']}")
    print(f"Frozen core cases:    {split_stats['frozen_core_count']} (100% preserved)")
    print(f"New scaled cases:     {split_stats['new_cases_count']}")
    print("\nOverall Split Distribution:")
    for s_name, cnt in sorted(split_stats['split_counts'].items()):
        pct = cnt / len(split_df) * 100
        print(f"  {s_name:5s}: {cnt:4d} ({pct:.1f}%)")
    print("=" * 60)
    print(f"Clean manifest: {clean_path}")
    print(f"Split manifest: {main_split_path}")
    print(f"Split stats:    {stats_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
