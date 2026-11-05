"""Phase 19 acquisition integrity audit (read-only)."""
import json
from pathlib import Path

import pandas as pd

from src.ingestion.pdf_downloader import validate_pdf

META = Path("data/metadata")
m = pd.read_csv(META / "manifest_main.csv")
core = pd.read_csv(META / "manifest_prototype.csv")
split = pd.read_csv(META / "split_manifest.csv")
sel = pd.read_csv(META / "main_selection.csv")

ok = m["acquisition_status"].isin(["success", "already_exists"])
def safe_validate(filepath: Path) -> tuple[bool, str]:
    if not filepath.exists():
        return False, "missing"
    try:
        return validate_pdf(filepath)
    except OSError as e:
        import time
        time.sleep(0.5)
        try:
            return validate_pdf(filepath)
        except Exception as e2:
            return False, f"OSError: {e2}"
    except Exception as e:
        return False, str(e)

files_ok = []
validation_errors = []
for idx, r in m[ok].iterrows():
    p = Path(str(r["local_path"]))
    v, err = safe_validate(p)
    files_ok.append(v)
    if not v:
        validation_errors.append({"index": idx, "path": str(p), "error": err})

new = m.iloc[len(core):]
dup_hash = m[ok & m["sha256"].astype(str).str.len().gt(0)]["sha256"].duplicated(keep=False)
out = {
    "total_rows": len(m),
    "frozen_core_rows": len(core),
    "new_rows": len(new),
    "acquired_ok": int(ok.sum()),
    "files_valid_on_disk": int(sum(files_ok)),
    "unique_case_id": int(m["case_id"].nunique()),
    "unique_filename": int(m["filename"].nunique()),
    "unique_sha256": int(m.loc[ok, "sha256"].nunique()),
    "content_duplicate_rows": int(dup_hash.sum()),
    "core_prefix_unchanged": bool(
        list(m.iloc[: len(core)]["case_id"]) == list(core["case_id"])
        and list(m.iloc[: len(core)]["sha256"].astype(str)) == list(core["sha256"].astype(str))),
    "new_overlaps_core_case_id": int(len(set(new["case_id"]) & set(core["case_id"]))),
    "new_overlaps_core_sha256": int(len(set(new["sha256"]) & set(core["sha256"]))),
    "selection_rows": len(sel),
    "all_selected_acquired": bool(set(sel["case_id"]) <= set(new.loc[new["acquisition_status"].isin(["success", "already_exists"]), "case_id"])),
    "existing_split_counts_still": split["split"].value_counts().to_dict(),
    "year_counts_all": m["year"].value_counts().sort_index().to_dict(),
    "total_mb": round(m.loc[ok, "file_size"].sum() / 1048576, 1),
    "validation_errors": validation_errors,
}
Path("data/metadata/acquisition_audit_main.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
print(json.dumps(out, indent=2, default=str))
