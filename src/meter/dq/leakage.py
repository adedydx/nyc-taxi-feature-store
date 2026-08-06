"""Leaky vs correct join demo for training-set correctness."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from meter.config import Settings, get_settings
from meter.features.pit_join import leaky_join_use_future_hour, point_in_time_join
from meter.training.dataset import build_label_frame


@dataclass
class LeakageDemoResult:
    n_rows: int
    leaky_label_corr: float
    correct_feature_label_corr: float
    leaky_exact_match_rate: float
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _corr(a: pd.Series, b: pd.Series) -> float:
    paired = pd.concat([a, b], axis=1).dropna()
    if len(paired) < 2:
        return float("nan")
    return float(paired.iloc[:, 0].corr(paired.iloc[:, 1]))


def run_leakage_demo(settings: Settings | None = None) -> LeakageDemoResult:
    """
    Compare an intentional leaky join against a correct PIT join.

    Leaky feature = next-hour trip count (same signal as the label).
    Correct feature = trips_1h known at prediction time.
    """
    settings = settings or get_settings()
    zone_hours_path = settings.path("offline_dir") / "zone_hours.parquet"
    features_path = settings.path("offline_dir") / "zone_features.parquet"
    if not zone_hours_path.exists() or not features_path.exists():
        raise FileNotFoundError(
            "Offline artifacts missing. Run `meter offline` before the leakage demo."
        )

    zone_hours = pd.read_parquet(zone_hours_path)
    features = pd.read_parquet(features_path)
    threshold = int(settings.raw["training"]["high_demand_threshold_trips"])
    horizon = int(settings.raw["training"]["label_horizon_hours"])
    labels = build_label_frame(
        zone_hours,
        horizon_hours=horizon,
        high_demand_threshold=threshold,
    )

    leaky = leaky_join_use_future_hour(labels, zone_hours)
    correct = point_in_time_join(labels, features)

    leaky_corr = _corr(leaky["leaked_trips_next_1h"], leaky["label_trips_next_1h"])
    correct_corr = _corr(correct["trips_1h"], correct["label_trips_next_1h"])

    matched = leaky.dropna(subset=["leaked_trips_next_1h"])
    exact_rate = (
        float((matched["leaked_trips_next_1h"] == matched["label_trips_next_1h"]).mean())
        if len(matched)
        else float("nan")
    )

    summary = (
        "Leaky join nearly reproduces the label (future peek). "
        "Correct PIT join only uses features with as_of_ts <= prediction_ts, "
        "so correlation with the label is much weaker — as it should be."
    )
    return LeakageDemoResult(
        n_rows=int(len(labels)),
        leaky_label_corr=leaky_corr,
        correct_feature_label_corr=correct_corr,
        leaky_exact_match_rate=exact_rate,
        summary=summary,
    )
