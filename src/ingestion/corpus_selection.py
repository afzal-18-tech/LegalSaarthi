"""
LegalSaarthi corpus selection for scaling (Phase 19).

Selects the additional judgments needed to grow the corpus from the frozen
200-case prototype to the 1,000-case main corpus.

Design (AGENTS.md compliance):
  * The existing prototype cases are a FROZEN CORE: never re-selected, never
    re-split (Phases 9-18 benchmark/test results depend on them).
  * New cases are drawn from the metadata of ALL configured years with
    proportional-to-year quotas (largest remainder), seeded and reproducible.
  * Selection happens BEFORE resolution/download and is NOT filtered by
    whether a PDF turns out to be downloadable, so failures stay visible
    instead of being silently replaced by easier cases.
  * The selected list is ordered by a seeded shuffle so any prefix
    (10 / 50 / 200 documents) is itself roughly year-representative; this is
    what the staged acquisition gates operate on.
"""
from typing import Dict, Iterable, List

import pandas as pd

from src.ingestion.source_resolver import derive_pdf_filename


def largest_remainder(counts: Dict[int, int], total: int) -> Dict[int, int]:
    """Allocate `total` seats across keys proportionally to `counts`.

    Never allocates more than counts[k] to a key (given total <= sum(counts)).
    """
    pool = sum(counts.values())
    if total > pool:
        raise ValueError(f"Cannot select {total} from pool of {pool}")
    if total == 0 or pool == 0:
        return {k: 0 for k in counts}
    shares = {k: total * c / pool for k, c in counts.items()}
    alloc = {k: int(s) for k, s in shares.items()}
    remaining = total - sum(alloc.values())
    order = sorted(counts, key=lambda k: (shares[k] - alloc[k], counts[k], -k), reverse=True)
    for k in order:
        if remaining <= 0:
            break
        if alloc[k] < counts[k]:
            alloc[k] += 1
            remaining -= 1
    return alloc


def annotate_filenames(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["filename"] = [derive_pdf_filename(r) for _, r in out.iterrows()]
    return out


def select_new_cases(
    metadata_df: pd.DataFrame,
    frozen_filenames: Iterable[str],
    frozen_case_ids: Iterable[str],
    n_new: int,
    seed: int = 42,
    year_col: str = "year_partition",
) -> pd.DataFrame:
    """Select `n_new` cases not in the frozen core.

    Returns a DataFrame with the metadata rows plus:
      filename, stage_order (1..n_new, seeded shuffle order), selection_seed.
    Rows whose filename cannot be derived are KEPT eligible (they surface as
    resolution failures downstream rather than being silently skipped).
    """
    df = annotate_filenames(metadata_df)
    frozen_fn = set(frozen_filenames)
    frozen_ids = {str(c) for c in frozen_case_ids}

    id_col = "case_id" if "case_id" in df.columns else None
    mask = ~df["filename"].isin(frozen_fn)
    if id_col:
        mask &= ~df[id_col].astype(str).isin(frozen_ids)
    pool = df[mask].copy()

    # Drop duplicate physical documents inside the pool (same derived filename).
    has_fn = pool["filename"].notna()
    dup = pool[has_fn].duplicated(subset=["filename"], keep="first")
    pool = pd.concat([pool[~has_fn], pool[has_fn][~dup]], ignore_index=True)

    counts = {int(y): int(c) for y, c in pool[year_col].value_counts().items()}
    quota = largest_remainder(counts, n_new)

    picks: List[pd.DataFrame] = []
    for y in sorted(quota):
        if quota[y] == 0:
            continue
        g = pool[pool[year_col] == y].sample(n=quota[y], random_state=seed + int(y))
        picks.append(g)
    sel = pd.concat(picks, ignore_index=True)
    sel = sel.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    sel["stage_order"] = range(1, len(sel) + 1)
    sel["selection_seed"] = seed
    return sel
