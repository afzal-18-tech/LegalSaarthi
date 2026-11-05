#!/usr/bin/env python3
"""
LegalSaarthi — Legal Structure Detection Script (Phase 7)

Reads parsed JSON documents, runs the rule-based StructureDetector on each,
and saves enriched JSON with section labels per paragraph.

Usage:
    python scripts/detect_structure.py
    python scripts/detect_structure.py --limit 10
    python scripts/detect_structure.py --manifest data/metadata/split_manifest.csv
"""
import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.structure_detector import SectionLabel, StructureDetector
from src.ingestion.manifest import load_manifest
from src.utils.config import load_config
from src.utils.logging_setup import setup_logging, get_logger


def _load_parsed_doc(parsed_dir: Path, case_id: str, year: str) -> Dict[str, Any]:
    """Load a parsed JSON document by case_id and year."""
    parsed_path = parsed_dir / f"year={year}" / f"{case_id}.json"
    if not parsed_path.exists():
        raise FileNotFoundError(f"Parsed file not found: {parsed_path}")
    with open(parsed_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_structured_doc(output_dir: Path, doc: Dict[str, Any], year: str) -> Path:
    """Save a structured JSON document."""
    out_path = output_dir / f"year={year}" / f"{doc['case_id']}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    return out_path


def _generate_report(
    stats: Dict[str, Any],
    output_path: Path,
) -> None:
    """Generate a markdown report from structure detection stats."""
    total = stats["total"]
    success = stats["success"]
    failed = stats["failed"]
    label_totals = stats["label_totals"]
    per_doc_distinct = stats["per_doc_distinct_labels"]
    docs_all_unknown = stats["docs_all_unknown"]

    total_paras = sum(label_totals.values())

    lines = [
        "# Structure Detection Report (Phase 7)",
        "",
        f"**Date**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"**Detector version**: {StructureDetector.STRUCTURE_VERSION}",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Documents processed | {total} |",
        f"| Successful | {success} |",
        f"| Failed | {failed} |",
        f"| Total paragraphs labelled | {total_paras} |",
        f"| Documents with all UNKNOWN | {docs_all_unknown} |",
        "",
        "## Label Distribution",
        "",
        "| Label | Count | Percentage |",
        "|---|---|---|",
    ]
    for label in SectionLabel:
        count = label_totals.get(label.value, 0)
        pct = (count / total_paras * 100) if total_paras > 0 else 0
        lines.append(f"| {label.value} | {count} | {pct:.1f}% |")

    lines += [
        "",
        "## Per-Document Distinct Labels",
        "",
        "| Distinct Labels | Documents |",
        "|---|---|",
    ]
    for k in sorted(per_doc_distinct.keys()):
        lines.append(f"| {k} | {per_doc_distinct[k]} |")

    lines += [
        "",
        "## Gate Check",
        "",
    ]
    # Gate: every document has at least 3 distinct labels
    min_distinct = min(per_doc_distinct.keys()) if per_doc_distinct else 0
    docs_below_3 = sum(v for k, v in per_doc_distinct.items() if k < 3)
    if docs_below_3 == 0 and success > 0:
        lines.append("> ✅ **PASS** — All documents have ≥3 distinct section labels.")
    else:
        lines.append(f"> ⚠️ **WARNING** — {docs_below_3} documents have <3 distinct labels.")

    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Detect legal structure in parsed documents.")
    parser.add_argument("--manifest", type=str, default=None, help="Path to manifest CSV")
    parser.add_argument("--limit", type=int, default=None, help="Process max N documents")
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config)
    logger = get_logger("scripts.detect_structure")

    # Resolve manifest
    if args.manifest:
        manifest_path = Path(args.manifest)
    else:
        manifest_path = PROJECT_ROOT / "data" / "metadata" / "split_manifest.csv"
        if not manifest_path.exists():
            manifest_path = PROJECT_ROOT / "data" / "metadata" / "clean_manifest.csv"
        if not manifest_path.exists():
            logger.error("No valid manifest found.")
            sys.exit(1)

    logger.info(f"Structure detection from manifest: {manifest_path}")
    manifest_df = load_manifest(manifest_path)

    # Filter to successfully acquired files
    mask = manifest_df["acquisition_status"].isin(["success", "already_exists"])
    df = manifest_df[mask]

    if args.limit:
        df = df.head(args.limit)

    logger.info(f"Targeting {len(df)} documents for structure detection.")

    # Directories
    parsed_dir = PROJECT_ROOT / "data" / "processed" / "parsed"
    structure_config = config.get("structure_detection", {})
    output_dir = PROJECT_ROOT / structure_config.get("output_dir", "data/processed/structured")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize detector
    detector = StructureDetector(
        header_page_threshold=structure_config.get("header_page_threshold", 2),
        decision_tail_ratio=structure_config.get("decision_tail_ratio", 0.15),
    )

    # Aggregate stats
    label_totals: Counter = Counter()
    per_doc_distinct: Counter = Counter()
    docs_all_unknown = 0
    success_count = 0
    fail_count = 0
    errors: List[Dict[str, str]] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Detecting structure"):
        case_id = row["case_id"]
        year = str(row.get("year", "unknown"))

        try:
            parsed_doc = _load_parsed_doc(parsed_dir, case_id, year)
            structured_doc = detector.detect_structure(parsed_doc)
            _save_structured_doc(output_dir, structured_doc, year)

            # Accumulate stats
            summary = structured_doc.get("sections_summary", {})
            for lbl, count in summary.items():
                label_totals[lbl] += count

            distinct = sum(1 for v in summary.values() if v > 0)
            per_doc_distinct[distinct] += 1

            if distinct == 0 or (distinct == 1 and summary.get("UNKNOWN", 0) > 0):
                docs_all_unknown += 1

            success_count += 1

        except Exception as e:
            logger.error(f"Failed to process {case_id}: {e}")
            errors.append({"case_id": case_id, "error": str(e)})
            fail_count += 1

    # Save stats JSON
    stats = {
        "total": success_count + fail_count,
        "success": success_count,
        "failed": fail_count,
        "label_totals": dict(label_totals),
        "per_doc_distinct_labels": {int(k): v for k, v in sorted(per_doc_distinct.items())},
        "docs_all_unknown": docs_all_unknown,
        "errors": errors,
        "detector_version": StructureDetector.STRUCTURE_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    stats_path = PROJECT_ROOT / "data" / "metadata" / "structure_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    # Generate report
    report_path = PROJECT_ROOT / "reports" / "structure_detection_report.md"
    _generate_report(stats, report_path)

    # Print summary
    total_paras = sum(label_totals.values())
    print(f"\n{'='*60}")
    print(f"STRUCTURE DETECTION REPORT")
    print(f"{'='*60}")
    print(f"Documents processed: {stats['total']}")
    print(f"Success:             {success_count}")
    print(f"Failed:              {fail_count}")
    print(f"Total paragraphs:    {total_paras}")
    print(f"Docs all UNKNOWN:    {docs_all_unknown}")
    print(f"{'='*60}")
    print(f"Label distribution:")
    for lbl in SectionLabel:
        count = label_totals.get(lbl.value, 0)
        pct = (count / total_paras * 100) if total_paras > 0 else 0
        print(f"  {lbl.value:15s} {count:6d}  ({pct:5.1f}%)")
    print(f"{'='*60}")
    print(f"Per-doc distinct labels:")
    for k in sorted(per_doc_distinct.keys()):
        print(f"  {k} labels: {per_doc_distinct[k]} docs")
    print(f"{'='*60}")
    print(f"Structured JSONs: {output_dir}")
    print(f"Stats:            {stats_path}")
    print(f"Report:           {report_path}")
    print(f"{'='*60}")

    # Gate check
    docs_below_3 = sum(v for k, v in per_doc_distinct.items() if k < 3)
    if docs_below_3 == 0 and success_count > 0:
        print(f"GATE: PASS ✓ (all documents have ≥3 distinct labels)")
        sys.exit(0)
    else:
        print(f"GATE: WARNING — {docs_below_3} documents have <3 distinct labels")
        sys.exit(0)  # Not a hard fail, this is a heuristic baseline


if __name__ == "__main__":
    main()
