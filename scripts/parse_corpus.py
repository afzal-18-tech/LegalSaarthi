#!/usr/bin/env python3
"""
LegalSaarthi — Corpus Parsing Script (Phase 6)

Iterates over the validated corpus manifest, parses the PDF files using PyMuPDF,
and saves the canonical text output as JSON files.

Usage:
    python scripts/parse_corpus.py --manifest data/metadata/split_manifest.csv
    python scripts/parse_corpus.py --limit 10
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.manifest import load_manifest
from src.preprocessing.pdf_parser import PDFParser
from src.utils.config import load_config
from src.utils.logging_setup import setup_logging, get_logger


def _parse_single_pdf(
    args: Tuple[str, str, str, Path]
) -> Tuple[str, str, Dict[str, Any], Optional[str]]:
    """Worker function for parallel parsing."""
    case_id, local_path_str, year, output_dir = args
    pdf_path = PROJECT_ROOT / local_path_str if not Path(local_path_str).is_absolute() else Path(local_path_str)
    
    output_file = output_dir / f"year={year}" / f"{case_id}.json"
    
    if output_file.exists():
        # Already parsed
        with open(output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return case_id, data.get("parse_status", "success"), data, None

    parser = PDFParser(remove_header_footer=True)
    try:
        parsed_data = parser.parse_pdf(pdf_path, case_id)
        
        # Save to disk
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(parsed_data, f, ensure_ascii=False, indent=2)
            
        return case_id, parsed_data["parse_status"], parsed_data, None
    except Exception as e:
        return case_id, "failed_error", {}, str(e)


def main():
    parser = argparse.ArgumentParser(description="Parse corpus PDFs to JSON text.")
    parser.add_argument("--manifest", type=str, default=None, help="Path to manifest CSV")
    parser.add_argument("--limit", type=int, default=None, help="Process max N documents")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config)
    logger = get_logger("scripts.parse_corpus")

    if args.manifest:
        manifest_path = Path(args.manifest)
    else:
        # Default to split_manifest if exists, else clean_manifest
        manifest_path = PROJECT_ROOT / "data" / "metadata" / "split_manifest.csv"
        if not manifest_path.exists():
            manifest_path = PROJECT_ROOT / "data" / "metadata" / "clean_manifest.csv"
        if not manifest_path.exists():
            logger.error("No valid manifest found.")
            sys.exit(1)

    logger.info(f"Parsing corpus from: {manifest_path}")
    manifest_df = load_manifest(manifest_path)

    # Filter to successfully acquired files
    mask_acquired = manifest_df["acquisition_status"].isin(["success", "already_exists"])
    df_acquired = manifest_df[mask_acquired]
    
    if args.limit:
        df_acquired = df_acquired.head(args.limit)
        
    logger.info(f"Targeting {len(df_acquired)} PDFs for parsing.")

    output_dir = PROJECT_ROOT / "data" / "processed" / "parsed"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare arguments for multiprocessing
    tasks = []
    for _, row in df_acquired.iterrows():
        tasks.append((
            row["case_id"],
            row["local_path"],
            row.get("year", "unknown"),
            output_dir
        ))

    stats = {
        "total": len(tasks),
        "success": 0,
        "failed_possible_scan": 0,
        "failed_error": 0,
        "errors": []
    }

    logger.info(f"Starting parsing with {args.workers} workers...")
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_parse_single_pdf, task): task for task in tasks}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Parsing PDFs"):
            case_id, status, data, error = future.result()
            
            stats[status] = stats.get(status, 0) + 1
            if error:
                stats["errors"].append({"case_id": case_id, "error": error})

    # Save parsing stats
    stats_path = PROJECT_ROOT / "data" / "metadata" / "parsing_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    success_rate = stats["success"] / stats["total"] if stats["total"] > 0 else 0

    print(f"\n{'='*60}")
    print(f"PARSING REPORT")
    print(f"{'='*60}")
    print(f"Total processed: {stats['total']}")
    print(f"Success:         {stats['success']}")
    print(f"Failed (scan):   {stats['failed_possible_scan']} (Requires OCR)")
    print(f"Failed (error):  {stats['failed_error']}")
    print(f"Success rate:    {success_rate:.1%}")
    print(f"{'='*60}")
    print(f"Parsed JSONs:    {output_dir}")
    print(f"Parsing stats:   {stats_path}")
    print(f"{'='*60}")
    
    if success_rate >= 0.95:
         print(f"GATE: PASS ✓")
         sys.exit(0)
    else:
         print(f"GATE: FAIL ✗ (Success rate < 95%, review required)")
         sys.exit(1)


if __name__ == "__main__":
    main()
