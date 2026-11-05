#!/usr/bin/env python3
"""
LegalSaarthi — Corpus Validation Script (Phase 5)

Validates downloaded PDFs, detects duplicates, checks metadata quality,
and generates a data quality report.

Usage:
    python scripts/validate_corpus.py [--manifest data/metadata/manifest_pilot.csv]
    python scripts/validate_corpus.py --stage prototype
"""
import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.pdf_downloader import validate_pdf
from src.ingestion.manifest import load_manifest, save_manifest
from src.utils.config import load_config
from src.utils.hashing import sha256_file
from src.utils.logging_setup import setup_logging, get_logger


def validate_corpus(manifest_path: Path, config: dict) -> dict:
    """Run full corpus validation.

    Checks:
    1. PDF existence and integrity
    2. Duplicate case_ids
    3. Duplicate file content (SHA-256)
    4. Missing metadata fields
    5. Year/date consistency
    6. File size distribution

    Returns:
        Validation report dict.
    """
    logger = get_logger("scripts.validate_corpus")
    manifest_df = load_manifest(manifest_path)
    project_root = PROJECT_ROOT

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(manifest_path),
        "total_records": len(manifest_df),
        "checks": {},
        "issues": [],
        "summary": {},
    }

    # ── Check 1: PDF existence and integrity ──
    logger.info("Check 1: PDF existence and integrity...")
    valid_count = 0
    missing_count = 0
    invalid_count = 0
    total_size = 0
    sizes = []
    sha_map = {}  # sha256 -> list of case_ids (for dedup)

    for _, row in manifest_df.iterrows():
        local_path = row.get("local_path", "")
        case_id = row.get("case_id", "unknown")
        status = row.get("acquisition_status", "")

        if status not in ("success", "already_exists") or not local_path:
            missing_count += 1
            continue

        path = Path(local_path)
        if not path.is_absolute():
            path = project_root / path

        if not path.exists():
            missing_count += 1
            report["issues"].append({
                "type": "missing_file",
                "case_id": case_id,
                "path": str(path),
            })
            continue

        is_valid, err = validate_pdf(path)
        if not is_valid:
            invalid_count += 1
            report["issues"].append({
                "type": "invalid_pdf",
                "case_id": case_id,
                "path": str(path),
                "error": err,
            })
            continue

        valid_count += 1
        fsize = path.stat().st_size
        total_size += fsize
        sizes.append(fsize)

        # Check SHA for content duplicates
        sha = row.get("sha256", "")
        if not sha:
            sha = sha256_file(path)
        if sha in sha_map:
            sha_map[sha].append(case_id)
        else:
            sha_map[sha] = [case_id]

    report["checks"]["pdf_integrity"] = {
        "valid": valid_count,
        "missing": missing_count,
        "invalid": invalid_count,
        "total_size_mb": round(total_size / (1024 * 1024), 2),
    }

    if sizes:
        report["checks"]["file_sizes"] = {
            "min_kb": round(min(sizes) / 1024, 1),
            "max_kb": round(max(sizes) / 1024, 1),
            "mean_kb": round(sum(sizes) / len(sizes) / 1024, 1),
            "median_kb": round(sorted(sizes)[len(sizes) // 2] / 1024, 1),
        }

    # ── Check 2: Duplicate case_ids ──
    logger.info("Check 2: Duplicate case_ids...")
    case_ids = manifest_df["case_id"].tolist()
    case_id_counts = Counter(case_ids)
    duplicates = {k: v for k, v in case_id_counts.items() if v > 1}

    report["checks"]["duplicate_case_ids"] = {
        "unique": len(case_id_counts),
        "duplicates": len(duplicates),
        "duplicate_ids": list(duplicates.keys())[:20],  # cap for readability
    }

    if duplicates:
        for dup_id, count in duplicates.items():
            report["issues"].append({
                "type": "duplicate_case_id",
                "case_id": dup_id,
                "count": count,
            })

    # ── Check 3: Duplicate file content ──
    logger.info("Check 3: Duplicate file content (SHA-256)...")
    content_dupes = {sha: ids for sha, ids in sha_map.items() if len(ids) > 1}

    report["checks"]["duplicate_content"] = {
        "unique_files": len(sha_map),
        "content_duplicates": len(content_dupes),
        "duplicate_groups": [
            {"sha256": sha[:16], "case_ids": ids}
            for sha, ids in list(content_dupes.items())[:10]
        ],
    }

    if content_dupes:
        for sha, ids in content_dupes.items():
            report["issues"].append({
                "type": "duplicate_content",
                "sha256": sha[:16],
                "case_ids": ids,
            })

    # ── Check 4: Missing metadata fields ──
    logger.info("Check 4: Missing metadata fields...")
    required_fields = ["case_id", "year", "decision_date", "source_url", "sha256"]
    field_coverage = {}
    for field in required_fields:
        if field in manifest_df.columns:
            non_null = manifest_df[field].notna().sum()
            non_empty = manifest_df[field].astype(str).str.strip().ne("").sum()
            field_coverage[field] = {
                "present": int(min(non_null, non_empty)),
                "missing": int(len(manifest_df) - min(non_null, non_empty)),
            }
        else:
            field_coverage[field] = {"present": 0, "missing": len(manifest_df)}

    report["checks"]["metadata_completeness"] = field_coverage

    # ── Check 5: Year distribution ──
    logger.info("Check 5: Year distribution...")
    if "year" in manifest_df.columns:
        year_dist = manifest_df["year"].value_counts().sort_index().to_dict()
        report["checks"]["year_distribution"] = {
            str(k): int(v) for k, v in year_dist.items()
        }

    # ── Summary ──
    total_issues = len(report["issues"])
    report["summary"] = {
        "total_records": len(manifest_df),
        "valid_pdfs": valid_count,
        "validation_rate": round(valid_count / max(len(manifest_df), 1), 4),
        "unique_case_ids": len(case_id_counts),
        "duplicate_case_ids": len(duplicates),
        "content_duplicates": len(content_dupes),
        "total_issues": total_issues,
        "quality_gate": "PASS" if (
            valid_count / max(len(manifest_df), 1) >= 0.95
            and len(duplicates) == 0
        ) else "REVIEW_NEEDED",
    }

    logger.info(f"Validation complete: {valid_count} valid, {total_issues} issues")
    return report


def save_quality_report(report: dict, output_dir: Path) -> tuple[Path, Path]:
    """Save quality report in JSON and markdown formats."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON report
    json_path = output_dir / "corpus_quality_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    # Markdown report
    md_path = output_dir / "corpus_quality_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Corpus Quality Report\n\n")
        f.write(f"**Generated**: {report['timestamp']}\n")
        f.write(f"**Manifest**: `{report['manifest_path']}`\n\n")

        s = report["summary"]
        f.write("## Summary\n\n")
        f.write(f"| Metric | Value |\n|--------|-------|\n")
        f.write(f"| Total records | {s['total_records']} |\n")
        f.write(f"| Valid PDFs | {s['valid_pdfs']} |\n")
        f.write(f"| Validation rate | {s['validation_rate']:.1%} |\n")
        f.write(f"| Unique case IDs | {s['unique_case_ids']} |\n")
        f.write(f"| Duplicate case IDs | {s['duplicate_case_ids']} |\n")
        f.write(f"| Content duplicates | {s['content_duplicates']} |\n")
        f.write(f"| Total issues | {s['total_issues']} |\n")
        f.write(f"| **Quality gate** | **{s['quality_gate']}** |\n\n")

        if "pdf_integrity" in report["checks"]:
            pi = report["checks"]["pdf_integrity"]
            f.write("## PDF Integrity\n\n")
            f.write(f"- Valid: {pi['valid']}\n")
            f.write(f"- Missing: {pi['missing']}\n")
            f.write(f"- Invalid: {pi['invalid']}\n")
            f.write(f"- Total size: {pi['total_size_mb']} MB\n\n")

        if "file_sizes" in report["checks"]:
            fs = report["checks"]["file_sizes"]
            f.write("## File Size Distribution\n\n")
            f.write(f"- Min: {fs['min_kb']} KB\n")
            f.write(f"- Max: {fs['max_kb']} KB\n")
            f.write(f"- Mean: {fs['mean_kb']} KB\n")
            f.write(f"- Median: {fs['median_kb']} KB\n\n")

        if "year_distribution" in report["checks"]:
            yd = report["checks"]["year_distribution"]
            f.write("## Year Distribution\n\n")
            f.write("| Year | Count |\n|------|-------|\n")
            for year, count in sorted(yd.items()):
                f.write(f"| {year} | {count} |\n")
            f.write("\n")

        if report["issues"]:
            f.write("## Issues\n\n")
            for issue in report["issues"][:50]:
                f.write(f"- **{issue['type']}**: case_id=`{issue.get('case_id', 'N/A')}`")
                if "error" in issue:
                    f.write(f" — {issue['error']}")
                f.write("\n")

    return json_path, md_path


def main():
    parser = argparse.ArgumentParser(description="Validate corpus quality")
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Path to manifest CSV",
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["smoke", "pilot", "prototype", "main"],
        default=None,
        help="Use manifest for this stage",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "default.yaml"),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config)
    logger = get_logger("scripts.validate_corpus")

    if args.manifest:
        manifest_path = Path(args.manifest)
    elif args.stage:
        manifest_path = PROJECT_ROOT / "data" / "metadata" / f"manifest_{args.stage}.csv"
    else:
        # Auto-detect latest manifest
        meta_dir = PROJECT_ROOT / "data" / "metadata"
        manifests = sorted(meta_dir.glob("manifest_*.csv"))
        if not manifests:
            logger.error("No manifest found. Run acquire_pdfs.py first.")
            sys.exit(1)
        manifest_path = manifests[-1]

    logger.info(f"Validating corpus from: {manifest_path}")

    report = validate_corpus(manifest_path, config)
    json_path, md_path = save_quality_report(
        report, PROJECT_ROOT / "reports"
    )

    s = report["summary"]
    print(f"\n{'='*60}")
    print(f"CORPUS QUALITY REPORT")
    print(f"{'='*60}")
    print(f"Records:      {s['total_records']}")
    print(f"Valid PDFs:    {s['valid_pdfs']}")
    print(f"Valid rate:    {s['validation_rate']:.1%}")
    print(f"Unique IDs:   {s['unique_case_ids']}")
    print(f"Dup case IDs: {s['duplicate_case_ids']}")
    print(f"Content dupes:{s['content_duplicates']}")
    print(f"Issues:       {s['total_issues']}")
    print(f"{'='*60}")
    print(f"QUALITY GATE: {s['quality_gate']}")
    print(f"{'='*60}")
    print(f"JSON report:  {json_path}")
    print(f"MD report:    {md_path}")
    print(f"{'='*60}")

    sys.exit(0 if s["quality_gate"] == "PASS" else 1)


if __name__ == "__main__":
    main()
