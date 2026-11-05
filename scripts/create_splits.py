#!/usr/bin/env python3
"""
LegalSaarthi — Data Split Script (Phase 5)

Assigns validated corpus records to development, validation, and test splits.
Uses stratified random split by year to ensure temporal representativeness.

Usage:
    python scripts/create_splits.py --manifest data/metadata/manifest_prototype.csv
    python scripts/create_splits.py --stage prototype --dev 0.7 --val 0.15 --test 0.15
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.manifest import load_manifest, save_manifest
from src.utils.config import load_config
from src.utils.logging_setup import setup_logging, get_logger


def create_stratified_split(
    manifest_df: pd.DataFrame,
    dev_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    stratify_col: str = "year",
) -> pd.DataFrame:
    """Create stratified dev/val/test split.

    Args:
        manifest_df: DataFrame with valid corpus records.
        dev_ratio: Fraction for development set.
        val_ratio: Fraction for validation set.
        test_ratio: Fraction for test set.
        seed: Random seed for reproducibility.
        stratify_col: Column to stratify by.

    Returns:
        DataFrame with added 'split' column.
    """
    logger = get_logger("scripts.create_splits")

    assert abs(dev_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        f"Ratios must sum to 1.0, got {dev_ratio + val_ratio + test_ratio}"

    # Filter to successfully acquired records only
    acquired = manifest_df[
        manifest_df["acquisition_status"].isin(["success", "already_exists"])
    ].copy()

    if len(acquired) == 0:
        logger.error("No acquired records to split")
        manifest_df["split"] = ""
        return manifest_df

    logger.info(f"Splitting {len(acquired)} acquired records (seed={seed})")

    # Shuffle within each stratum
    acquired = acquired.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    if stratify_col in acquired.columns:
        # Stratified split by year
        splits = []
        for year_val, group in acquired.groupby(stratify_col):
            n = len(group)
            n_dev = max(1, int(n * dev_ratio))
            n_val = max(1, int(n * val_ratio)) if n > 2 else 0
            n_test = n - n_dev - n_val

            group = group.copy()
            group["split"] = ""
            group.iloc[:n_dev, group.columns.get_loc("split")] = "dev"
            group.iloc[n_dev:n_dev + n_val, group.columns.get_loc("split")] = "val"
            group.iloc[n_dev + n_val:, group.columns.get_loc("split")] = "test"
            splits.append(group)

        result = pd.concat(splits, ignore_index=True)
    else:
        # Simple random split
        n = len(acquired)
        n_dev = int(n * dev_ratio)
        n_val = int(n * val_ratio)

        acquired["split"] = ""
        acquired.iloc[:n_dev, acquired.columns.get_loc("split")] = "dev"
        acquired.iloc[n_dev:n_dev + n_val, acquired.columns.get_loc("split")] = "val"
        acquired.iloc[n_dev + n_val:, acquired.columns.get_loc("split")] = "test"
        result = acquired

    # Stats
    split_counts = result["split"].value_counts().to_dict()
    logger.info(f"Split distribution: {split_counts}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Create dev/val/test splits")
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--stage", type=str, default=None)
    parser.add_argument("--dev", type=float, default=0.70)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "default.yaml"),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config)
    logger = get_logger("scripts.create_splits")

    if args.manifest:
        manifest_path = Path(args.manifest)
    elif args.stage:
        manifest_path = PROJECT_ROOT / "data" / "metadata" / f"manifest_{args.stage}.csv"
    else:
        meta_dir = PROJECT_ROOT / "data" / "metadata"
        manifests = sorted(meta_dir.glob("manifest_*.csv"))
        if not manifests:
            logger.error("No manifest found.")
            sys.exit(1)
        manifest_path = manifests[-1]

    logger.info(f"Creating splits from: {manifest_path}")
    manifest_df = load_manifest(manifest_path)

    split_df = create_stratified_split(
        manifest_df,
        dev_ratio=args.dev,
        val_ratio=args.val,
        test_ratio=args.test,
        seed=args.seed,
    )

    # Save split manifest
    split_path = PROJECT_ROOT / "data" / "metadata" / "split_manifest.csv"
    save_manifest(split_df, split_path)

    # Save split stats
    split_counts = split_df["split"].value_counts().to_dict()
    stats = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(manifest_path),
        "seed": args.seed,
        "ratios": {"dev": args.dev, "val": args.val, "test": args.test},
        "counts": split_counts,
        "total": len(split_df),
        "by_year": {},
    }

    if "year" in split_df.columns:
        for year_val in sorted(split_df["year"].unique()):
            year_group = split_df[split_df["year"] == year_val]
            stats["by_year"][str(year_val)] = year_group["split"].value_counts().to_dict()

    stats_path = PROJECT_ROOT / "data" / "metadata" / "split_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"\n{'='*60}")
    print(f"DATA SPLIT REPORT")
    print(f"{'='*60}")
    print(f"Total records: {len(split_df)}")
    for split_name, count in sorted(split_counts.items()):
        pct = count / len(split_df) * 100
        print(f"  {split_name:5s}: {count:4d} ({pct:.1f}%)")
    print(f"Seed: {args.seed}")
    print(f"{'='*60}")
    print(f"Split manifest: {split_path}")
    print(f"Split stats:    {stats_path}")
    print(f"{'='*60}")
    print(f"\n⚠️  Final test set is NOT YET LOCKED.")
    print(f"    Lock only before final tuning begins.")


if __name__ == "__main__":
    main()
