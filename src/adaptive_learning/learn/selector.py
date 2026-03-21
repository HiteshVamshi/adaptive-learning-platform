from __future__ import annotations

import pandas as pd

# Max resources per concept by mastery band
_LIMITS = {
    "needs_support": {"video": 3, "slides": 2, "article": 2},
    "developing": {"video": 2, "slides": 2, "article": 1},
    "mastered": {"video": 1, "slides": 1, "article": 0},
}


def _band_for_row(row: pd.Series) -> str:
    band = str(row.get("mastery_band") or "needs_support")
    if band not in _LIMITS:
        return "needs_support"
    return band


def select_resources_for_concept(
    *,
    concept_id: str,
    chapter_id: str,
    live_snapshot: pd.DataFrame,
    resources_df: pd.DataFrame,
) -> pd.DataFrame:
    """Return filtered, ordered resources for one concept using mastery-based caps."""
    if resources_df.empty:
        return resources_df

    snap = live_snapshot[live_snapshot["concept_id"].astype(str) == str(concept_id)]
    if snap.empty:
        band = "needs_support"
    else:
        band = _band_for_row(snap.iloc[0])

    limits = _LIMITS[band]
    cid = str(concept_id)
    chid = str(chapter_id)

    direct = resources_df[resources_df["concept_id"].astype(str) == cid].copy()
    chapter_rows = resources_df[
        (resources_df["concept_id"].astype(str) == "")
        & (resources_df["chapter_id"].astype(str) == chid)
    ].copy()
    if direct.empty:
        pool = chapter_rows
    else:
        pool = pd.concat([direct, chapter_rows], ignore_index=True).drop_duplicates(subset=["resource_id"])

    if pool.empty:
        return pool

    pool = pool.sort_values(by=["order_index", "resource_id"], ignore_index=True)
    picked: list[pd.DataFrame] = []
    for rtype, cap in limits.items():
        if cap <= 0:
            continue
        sub = pool[pool["resource_type"].astype(str) == rtype].head(cap)
        if not sub.empty:
            picked.append(sub)

    if not picked:
        return pool.head(0)

    out = pd.concat(picked, ignore_index=True).drop_duplicates(subset=["resource_id"])
    return out.sort_values(by=["order_index", "resource_id"], ignore_index=True)


def _snapshot_with_chapter_id(live_snapshot: pd.DataFrame, concepts: pd.DataFrame | None) -> pd.DataFrame:
    snap = live_snapshot.copy()
    if concepts is None or concepts.empty:
        return snap
    if "chapter_id" in snap.columns and snap["chapter_id"].notna().any():
        return snap
    cmap = concepts[concepts["node_type"] == "concept"][["concept_id", "chapter_id"]].drop_duplicates(
        subset=["concept_id"]
    )
    return snap.merge(cmap, on="concept_id", how="left")


def select_learn_plan(
    *,
    live_snapshot: pd.DataFrame,
    resources_df: pd.DataFrame,
    concepts: pd.DataFrame | None = None,
    max_concepts: int = 8,
) -> list[tuple[dict, pd.DataFrame]]:
    """
    Order concepts by weakest mastery first; attach selected resources per concept.

    Returns list of (concept_row_dict, resources_df).
    """
    if live_snapshot.empty or resources_df.empty:
        return []

    enriched = _snapshot_with_chapter_id(live_snapshot, concepts)
    concepts_sorted = enriched.sort_values(
        by=["graph_adjusted_mastery", "attempts_count"],
        ascending=[True, True],
        ignore_index=True,
    )
    seen: set[str] = set()
    plan: list[tuple[dict, pd.DataFrame]] = []

    for _, row in concepts_sorted.iterrows():
        cid = str(row["concept_id"])
        if cid in seen:
            continue
        seen.add(cid)
        chid = str(row.get("chapter_id", "") or "")
        if not chid or chid == "nan":
            chap_rows = resources_df[resources_df["concept_id"].astype(str) == cid]
            if not chap_rows.empty:
                chid = str(chap_rows.iloc[0]["chapter_id"])
            else:
                continue

        res = select_resources_for_concept(
            concept_id=cid,
            chapter_id=chid,
            live_snapshot=live_snapshot,
            resources_df=resources_df,
        )
        if res.empty:
            continue
        plan.append((row.to_dict(), res))
        if len(plan) >= max_concepts:
            break

    return plan


def format_learn_summary(plan: list[tuple[dict, pd.DataFrame]], max_items: int = 12) -> str:
    """Plain-text summary for agent / tutor responses."""
    lines: list[str] = []
    n = 0
    for concept_row, res_df in plan:
        name = str(concept_row.get("concept_name", concept_row.get("concept_id", "")))
        lines.append(f"- {name} ({concept_row.get('mastery_band', '')}):")
        for _, r in res_df.iterrows():
            if n >= max_items:
                lines.append("  …")
                return "\n".join(lines)
            lines.append(f"  • [{r['resource_type']}] {r['title']}: {r['url']}")
            n += 1
    return "\n".join(lines) if lines else "No curated learn links matched your concepts yet."
