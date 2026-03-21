from __future__ import annotations

import pandas as pd


def build_spaced_repetition_queue(
    *,
    user_attempts: pd.DataFrame,
    live_snapshot: pd.DataFrame,
    now: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Rank concepts for review: weaker mastery and longer gaps since last attempt score higher.

    Returns columns: concept_id, concept_name, chapter_name, mastery_band, graph_adjusted_mastery,
    days_since_last_attempt, priority, reason, suggested_action.
    """
    if live_snapshot.empty:
        return pd.DataFrame()

    def _naive_utc(ts: pd.Timestamp) -> pd.Timestamp:
        t = pd.Timestamp(ts)
        if t.tzinfo is not None:
            return t.tz_convert("UTC").tz_localize(None)
        return t

    now = _naive_utc(now or pd.Timestamp.utcnow())
    last_by_concept: dict[str, pd.Timestamp] = {}
    if not user_attempts.empty and "concept_id" in user_attempts.columns:
        ua = user_attempts.copy()
        ts_raw = pd.to_datetime(ua["timestamp"], errors="coerce", utc=True)
        ua["ts"] = ts_raw.dt.tz_convert("UTC").dt.tz_localize(None)
        grouped = ua.groupby("concept_id")["ts"].max()
        for cid, ts in grouped.items():
            if pd.notna(ts):
                last_by_concept[str(cid)] = ts

    rows: list[dict] = []
    for _, row in live_snapshot.iterrows():
        cid = str(row["concept_id"])
        band = str(row.get("mastery_band", "needs_support"))
        mastery = float(row.get("graph_adjusted_mastery", 0.0))
        last_ts = last_by_concept.get(cid)
        if last_ts is not None:
            days = max(0, (now - last_ts).days)
        else:
            days = 999

        # Priority: low mastery + time decay; mastered concepts get lower base but still surface if stale
        if band == "mastered":
            base = (1.0 - mastery) * 3.0 + min(days, 45) * 0.08
            reason = "Spaced review" if days > 7 else "Consolidate mastery"
            action = "Quick test or one hard question"
        elif band == "developing":
            base = (1.0 - mastery) * 8.0 + min(days, 21) * 0.15
            reason = "Reinforce developing skill"
            action = "Practice + short video recap"
        else:
            base = (1.0 - mastery) * 12.0 + min(days, 14) * 0.12
            reason = "Priority support topic"
            action = "Learn module + guided practice"

        rows.append(
            {
                "concept_id": cid,
                "concept_name": row.get("concept_name", ""),
                "chapter_name": row.get("chapter_name", ""),
                "mastery_band": band,
                "graph_adjusted_mastery": round(mastery, 4),
                "days_since_last_attempt": int(min(days, 999)),
                "priority": round(base, 4),
                "reason": reason,
                "suggested_action": action,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(by=["priority", "graph_adjusted_mastery"], ascending=[False, True], ignore_index=True)


def top_review_concepts(queue: pd.DataFrame, top_k: int = 8) -> pd.DataFrame:
    if queue.empty:
        return queue
    return queue.head(top_k).reset_index(drop=True)
