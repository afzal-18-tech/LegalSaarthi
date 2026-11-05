"""Unit tests for Phase 19 incremental splitting."""
import pandas as pd
import pytest

from scripts.create_main_splits import create_incremental_splits


def test_incremental_splits_preserves_frozen_core():
    # 4 existing cases
    existing_df = pd.DataFrame([
        {"case_id": "C1", "year": 2015, "split": "dev"},
        {"case_id": "C2", "year": 2015, "split": "val"},
        {"case_id": "C3", "year": 2015, "split": "test"},
        {"case_id": "C4", "year": 2015, "split": "dev"},
    ])
    
    # Clean manifest with 4 existing + 10 new cases
    new_rows = [{"case_id": f"NEW_{i}", "year": 2016 + (i % 3), "acquisition_status": "success"} for i in range(10)]
    clean_manifest = pd.concat([
        existing_df.assign(acquisition_status="already_exists").drop(columns=["split"]),
        pd.DataFrame(new_rows)
    ], ignore_index=True)
    
    result_df, stats = create_incremental_splits(
        clean_manifest=clean_manifest,
        existing_split_df=existing_df,
        seed=42,
    )
    
    assert len(result_df) == 14
    # Check that C1..C4 kept their exact splits
    for cid, expected_split in [("C1", "dev"), ("C2", "val"), ("C3", "test"), ("C4", "dev")]:
        actual = result_df.loc[result_df["case_id"] == cid, "split"].iloc[0]
        assert actual == expected_split, f"Case {cid} expected {expected_split}, got {actual}"
    
    # Check that new cases have dev/val/test assigned
    new_splits = result_df[result_df["case_id"].str.startswith("NEW_")]["split"]
    assert set(new_splits.unique()) <= {"dev", "val", "test"}
    assert not (new_splits == "").any()
    assert stats["frozen_core_count"] == 4
    assert stats["new_cases_count"] == 10
