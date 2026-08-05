"""Build point-in-time correct training rows from zone-hour facts and features."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from meter.config import Settings, get_settings
from meter.features.pit_join import point_in_time_join
from meter.schemas import TRAINING_COLUMNS


def build_label_frame(
    zone_hours: pd.DataFrame,
    horizon_hours: int = 1,
    high_demand_threshold: int = 50,
) -> pd.DataFrame:
    """
    Label at prediction time H = trip_count in hour H + horizon.

    Implemented by shifting zone-hour trip_count backward so each row's label
    is the count from the next hour.
    """
    df = zone_hours.copy()
    df["event_hour"] = pd.to_datetime(df["event_hour"])
    df = df.sort_values(["zone_id", "event_hour"])

    parts: list[pd.DataFrame] = []
    for zone_id, g in df.groupby("zone_id", sort=False):
        g = g.sort_values("event_hour").copy()
        g["label_trips_next_1h"] = g["trip_count"].shift(-horizon_hours)
        g["prediction_ts"] = g["event_hour"]
        g["zone_id"] = zone_id
        parts.append(g)

    labels = pd.concat(parts, ignore_index=True)
    labels = labels.dropna(subset=["label_trips_next_1h"]).copy()
    labels["label_trips_next_1h"] = labels["label_trips_next_1h"].astype(int)
    labels["label_high_demand"] = (
        labels["label_trips_next_1h"] >= high_demand_threshold
    ).astype(int)
    return labels[["zone_id", "prediction_ts", "label_trips_next_1h", "label_high_demand"]]


def build_training_set(
    settings: Settings | None = None,
    zone_hours_path: Path | None = None,
    features_path: Path | None = None,
) -> Path:
    settings = settings or get_settings()
    settings.ensure_dirs()

    zh_path = zone_hours_path or (settings.path("offline_dir") / "zone_hours.parquet")
    feat_path = features_path or (settings.path("offline_dir") / "zone_features.parquet")

    zone_hours = pd.read_parquet(zh_path)
    features = pd.read_parquet(feat_path)

    threshold = int(settings.raw["training"]["high_demand_threshold_trips"])
    horizon = int(settings.raw["training"]["label_horizon_hours"])
    labels = build_label_frame(
        zone_hours,
        horizon_hours=horizon,
        high_demand_threshold=threshold,
    )
    training = point_in_time_join(labels, features)

    # Keep a stable column order; drop rows where features were missing.
    missing = [c for c in TRAINING_COLUMNS if c not in training.columns]
    if missing:
        raise ValueError(f"Training set missing columns: {missing}")
    training = training.dropna(subset=["as_of_ts"]).copy()
    training = training[TRAINING_COLUMNS]

    out = settings.path("training_dir") / "training_set.parquet"
    training.to_parquet(out, index=False)
    return out
