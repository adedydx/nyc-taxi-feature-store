"""Unit tests for PIT joins and offline pipeline shapes."""

from __future__ import annotations

import pandas as pd

from meter.features.aggregate import trips_to_zone_hours
from meter.features.compute import compute_zone_features
from meter.features.pit_join import leaky_join_use_future_hour, point_in_time_join
from meter.ingest import generate_sample_trips
from meter.training.dataset import build_label_frame


def test_point_in_time_join_does_not_use_future_features():
    labels = pd.DataFrame(
        {
            "zone_id": [1, 1],
            "prediction_ts": pd.to_datetime(["2024-01-01 10:00", "2024-01-01 11:00"]),
            "label_trips_next_1h": [5, 9],
        }
    )
    features = pd.DataFrame(
        {
            "zone_id": [1, 1, 1],
            "as_of_ts": pd.to_datetime(
                ["2024-01-01 09:00", "2024-01-01 10:00", "2024-01-01 11:00"]
            ),
            "trips_1h": [1.0, 2.0, 99.0],
        }
    )
    out = point_in_time_join(labels, features)
    row_10 = out.loc[out["prediction_ts"] == "2024-01-01 10:00"].iloc[0]
    assert row_10["trips_1h"] == 2.0
    assert row_10["trips_1h"] != 99.0


def test_leaky_join_correlates_with_label():
    trips = generate_sample_trips(n_rows=8_000, days=5, n_zones=10, seed=1)
    zh = trips_to_zone_hours(trips)
    labels = build_label_frame(zh, horizon_hours=1)
    leaky = leaky_join_use_future_hour(labels, zh)
    matched = leaky.dropna(subset=["leaked_trips_next_1h"])
    assert not matched.empty
    assert (matched["leaked_trips_next_1h"] == matched["label_trips_next_1h"]).all()


def test_compute_features_shapes():
    trips = generate_sample_trips(n_rows=3_000, days=3, n_zones=5, seed=0)
    zh = trips_to_zone_hours(trips)
    feats = compute_zone_features(zh)
    assert {"trips_1h", "trips_24h", "trips_7d", "as_of_ts", "zone_id"} <= set(
        feats.columns
    )
    assert feats["trips_24h"].min() >= 0


def test_training_labels_are_next_hour_counts():
    zh = pd.DataFrame(
        {
            "zone_id": [1, 1, 1],
            "event_hour": pd.to_datetime(
                ["2024-01-01 10:00", "2024-01-01 11:00", "2024-01-01 12:00"]
            ),
            "trip_count": [10, 20, 30],
            "total_fare": [1.0, 2.0, 3.0],
            "avg_fare": [1.0, 2.0, 3.0],
            "avg_distance": [1.0, 2.0, 3.0],
            "avg_passengers": [1.0, 1.0, 1.0],
        }
    )
    labels = build_label_frame(zh, horizon_hours=1, high_demand_threshold=25)
    by_ts = labels.set_index("prediction_ts")["label_trips_next_1h"]
    assert by_ts.loc["2024-01-01 10:00"] == 20
    assert by_ts.loc["2024-01-01 11:00"] == 30
    assert "2024-01-01 12:00" not in by_ts.index  # no next hour → dropped
