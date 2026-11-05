"""Unit tests for Phase 19 corpus selection (no network)."""
import pandas as pd
import pytest

from src.ingestion.corpus_selection import largest_remainder, select_new_cases


def make_meta():
    rows = []
    for y, n in [(2015, 40), (2016, 30), (2017, 20), (2018, 10)]:
        for i in range(n):
            rows.append({"case_id": f"{y} INSC {i}", "path": f"{y}_1_{i}_{i+1}",
                         "year": y, "year_partition": y})
    return pd.DataFrame(rows)


def test_largest_remainder_sums_and_caps():
    a = largest_remainder({2015: 40, 2016: 30, 2017: 20, 2018: 10}, 25)
    assert sum(a.values()) == 25
    assert all(a[k] <= c for k, c in {2015: 40, 2016: 30, 2017: 20, 2018: 10}.items())
    assert a[2015] >= a[2018]


def test_largest_remainder_too_many_raises():
    with pytest.raises(ValueError):
        largest_remainder({2015: 3}, 5)


def test_frozen_core_excluded_and_count_exact():
    meta = make_meta()
    frozen_fn = ["2015_1_0_1_EN.pdf", "2015_1_1_2_EN.pdf"]
    frozen_ids = ["2015 INSC 0", "2015 INSC 1"]
    sel = select_new_cases(meta, frozen_fn, frozen_ids, 30, seed=1)
    assert len(sel) == 30
    assert not set(sel["filename"]) & set(frozen_fn)
    assert not set(sel["case_id"]) & set(frozen_ids)
    assert sel["filename"].is_unique


def test_deterministic_and_seed_sensitive():
    meta = make_meta()
    a = select_new_cases(meta, [], [], 30, seed=7)
    b = select_new_cases(meta, [], [], 30, seed=7)
    c = select_new_cases(meta, [], [], 30, seed=8)
    assert list(a["case_id"]) == list(b["case_id"])
    assert list(a["case_id"]) != list(c["case_id"])


def test_stage_order_is_contiguous_and_prefix_spans_years():
    sel = select_new_cases(make_meta(), [], [], 40, seed=3)
    assert list(sel["stage_order"]) == list(range(1, 41))
    assert sel[sel["stage_order"] <= 20]["year_partition"].nunique() >= 3


def test_underivable_filename_rows_stay_eligible():
    meta = make_meta()
    meta.loc[0, "path"] = "garbage"
    meta.loc[0, "year"] = None  # filename not derivable
    sel = select_new_cases(meta, [], [], 100, seed=1)
    assert len(sel) == 100  # nothing silently dropped
