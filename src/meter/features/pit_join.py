"""Point-in-time joins and an intentional leaky join for the correctness demo."""

from __future__ import annotations

import pandas as pd


def point_in_time_join(
    labels: pd.DataFrame,
    features: pd.DataFrame,
    *,
    entity_col: str = "zone_id",
    label_time_col: str = "prediction_ts",
    feature_time_col: str = "as_of_ts",
) -> pd.DataFrame:
    """
    Backward as-of join: for each label row at time T, attach the latest feature
    row with as_of_ts <= T for the same entity.
    """
    left = labels.copy()
    right = features.copy()
    left[label_time_col] = pd.to_datetime(left[label_time_col])
    right[feature_time_col] = pd.to_datetime(right[feature_time_col])

    # merge_asof with `by` requires the on-keys to be ordered; do it per entity
    # so PIT semantics stay obvious and pandas sort checks pass reliably.
    parts: list[pd.DataFrame] = []
    feature_cols = [c for c in right.columns if c != entity_col]
    entities = left[entity_col].drop_duplicates().tolist()
    right_by = {z: g for z, g in right.groupby(entity_col, sort=False)}

    for zone_id in entities:
        l = left.loc[left[entity_col] == zone_id].sort_values(label_time_col)
        r = right_by.get(zone_id)
        if r is None or r.empty:
            parts.append(l)
            continue
        r = r.sort_values(feature_time_col)
        joined = pd.merge_asof(
            l,
            r[feature_cols],
            left_on=label_time_col,
            right_on=feature_time_col,
            direction="backward",
        )
        parts.append(joined)

    return pd.concat(parts, ignore_index=True)


def leaky_join_use_future_hour(
    labels: pd.DataFrame,
    zone_hours: pd.DataFrame,
    *,
    entity_col: str = "zone_id",
    label_time_col: str = "prediction_ts",
) -> pd.DataFrame:
    """
    Incorrect join for demos: use trip_count from the *next* hour as a feature.

    That value is exactly the label in many cases — classic leakage.
    """
    zh = zone_hours.copy()
    zh["event_hour"] = pd.to_datetime(zh["event_hour"])
    future = zh.rename(
        columns={
            "event_hour": "future_hour",
            "trip_count": "leaked_trips_next_1h",
        }
    )[[entity_col, "future_hour", "leaked_trips_next_1h"]]

    out = labels.copy()
    out[label_time_col] = pd.to_datetime(out[label_time_col])
    out["future_hour"] = out[label_time_col] + pd.Timedelta(hours=1)
    return out.merge(future, on=[entity_col, "future_hour"], how="left")
