#!/usr/bin/env python3
"""
LegalSaarthi — Main corpus acquisition (Phase 19).

Grows the corpus from the frozen 200-case prototype to 1,000 cases with staged
gates. The selection is created once, saved, and then reused (immutable):

    python -m scripts.acquire_main_corpus --select-only
    python -m scripts.acquire_main_corpus --upto 10     # gate 1
    python -m scripts.acquire_main_corpus --upto 50     # gate 2
    python -m scripts.acquire_main_corpus --upto 200    # gate 3
    python -m scripts.acquire_main_corpus --upto 800    # full (=1000 total)

Each run is idempotent (already downloaded valid PDFs are re-verified, not
re-fetched). The gate requires success_rate >= stages.main.min_success_rate AND
reports how many resolutions were index-verified vs unverified.
"""
import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.corpus_selection import select_new_cases
from src.ingestion.manifest import (
    compute_acquisition_stats, create_manifest_from_metadata,
    load_manifest, save_download_failures, save_manifest,
)
from src.ingestion.metadata_client import acquire_metadata_for_years
from src.ingestion.pdf_downloader import PDFDownloader
from src.ingestion.source_resolver import SourceResolver
from src.utils.config import load_config
from src.utils.logging_setup import get_logger, setup_logging

META = PROJECT_ROOT / "data" / "metadata"
FROZEN_MANIFEST = META / "manifest_prototype.csv"
SELECTION = META / "main_selection.csv"
KEEP_COLS = ["case_id", "title", "path", "year", "year_partition", "decision_date",
             "filename", "stage_order", "selection_seed"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    ap.add_argument("--n-total", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--upto", type=int, default=None, help="Acquire first N of the new-case list")
    ap.add_argument("--select-only", action="store_true")
    args = ap.parse_args()

    config = load_config(args.config)
    setup_logging(config)
    log = get_logger("scripts.acquire_main_corpus")
    sys.stdout.reconfigure(encoding="utf-8")

    frozen = load_manifest(FROZEN_MANIFEST)
    n_new = args.n_total - len(frozen)

    # ---- Selection (immutable once written) -----------------------------
    if SELECTION.exists():
        sel = pd.read_csv(SELECTION)
        log.info(f"Reusing frozen selection: {SELECTION} ({len(sel)} new cases)")
    else:
        meta = acquire_metadata_for_years(
            config["scope"]["year_start"], config["scope"]["year_end"], config)
        sel = select_new_cases(meta, frozen["filename"], frozen["case_id"], n_new, seed=args.seed)
        sel_out = sel[[c for c in KEEP_COLS if c in sel.columns]]
        sel_out.to_csv(SELECTION, index=False)
        log.info(f"Wrote selection: {SELECTION} ({len(sel_out)} new cases, seed={args.seed})")
        sel = sel_out

    print("Selection by year:", sel["year_partition"].value_counts().sort_index().to_dict())
    if args.select_only or args.upto is None:
        return

    # ---- Staged acquisition --------------------------------------------
    batch = sel[sel["stage_order"] <= args.upto].copy()
    log.info(f"Acquiring first {len(batch)} new cases (stage_order <= {args.upto})")

    resolver = SourceResolver(config)
    resolutions = resolver.resolve_batch(batch)
    downloader = PDFDownloader(config)
    downloads = downloader.download_batch(resolutions, PROJECT_ROOT / "data" / "raw")

    stats = compute_acquisition_stats(downloads)
    # Denominator = ALL selected cases (resolution failures count as failures).
    n_ok = stats["success"]
    stats["selected"] = len(batch)
    stats["success_rate_of_selected"] = n_ok / len(batch) if len(batch) else 0.0
    stats["resolution_methods"] = dict(Counter(r.method for r in resolutions))
    stats["resolution_status"] = dict(Counter(r.status for r in resolutions))
    stats["unverified_resolutions"] = sum(
        1 for r in resolutions if r.status == "resolved" and r.method == "derived_unverified")
    by_year = {}
    dmap = {d.case_id: d for d in downloads}
    for r in resolutions:
        d = dmap.get(r.case_id)
        ok = d is not None and d.status in ("success", "already_exists")
        y = str(r.year or "unknown")
        by_year.setdefault(y, {"selected": 0, "ok": 0})
        by_year[y]["selected"] += 1
        by_year[y]["ok"] += int(ok)
    stats["by_year"] = by_year
    stats["upto"] = args.upto
    stats["seed"] = args.seed

    tag = f"main_new_upto{args.upto}"
    manifest_new = create_manifest_from_metadata(batch, resolutions, downloads)
    save_manifest(manifest_new, META / f"manifest_{tag}.csv")
    save_download_failures(downloads, META / f"download_failures_{tag}.csv")
    (META / f"acquisition_stats_{tag}.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    # Failures from resolution (never silently dropped)
    res_fail = [r.to_dict() for r in resolutions if r.status != "resolved"]
    if res_fail:
        pd.DataFrame(res_fail).to_csv(META / f"resolution_failures_{tag}.csv", index=False)

    # Combined manifest = frozen core + new (all rows kept)
    combined = pd.concat([frozen, manifest_new], ignore_index=True)
    save_manifest(combined, META / "manifest_main.csv")

    min_rate = config["stages"]["main"]["min_success_rate"]
    gate = stats["success_rate_of_selected"] >= min_rate
    print("\n" + "=" * 60)
    print(f"PHASE 19 ACQUISITION — first {args.upto} new cases")
    print("=" * 60)
    print(f"Selected:            {len(batch)}")
    print(f"Downloaded/valid:    {n_ok}  ({stats['success_rate_of_selected']:.1%})")
    print(f"Failed / skipped:    {stats['failed']} / {stats['skipped']}")
    print(f"Resolution methods:  {stats['resolution_methods']}")
    print(f"Unverified resolves: {stats['unverified_resolutions']}")
    print(f"Total size:          {stats['total_mb']} MB")
    print(f"By year:             {by_year}")
    print(f"GATE (>= {min_rate:.0%}): {'PASS' if gate else 'FAIL'}")
    print("=" * 60)
    sys.exit(0 if gate else 1)


if __name__ == "__main__":
    main()
