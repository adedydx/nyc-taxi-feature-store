"""DQ gates for ingest, features, freshness, and online/offline parity."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from meter.config import Settings, get_settings
from meter.online.store import LocalOnlineStore, latest_features_by_zone
from meter.schemas import FEATURE_COLUMNS, TRIP_COLUMNS


@dataclass
class CheckResult:
    name: str
    passed: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class DQReport:
    passed: bool
    checks: list[CheckResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "checks": [asdict(c) for c in self.checks],
        }

    def failed_names(self) -> list[str]:
        return [c.name for c in self.checks if not c.passed]


def check_ingest(trips: pd.DataFrame) -> CheckResult:
    if trips.empty:
        return CheckResult("ingest_non_empty", False, "Trip extract is empty")
    missing = [c for c in TRIP_COLUMNS if c not in trips.columns]
    if missing:
        return CheckResult(
            "ingest_schema",
            False,
            f"Missing required columns: {missing}",
            {"missing": missing},
        )
    try:
        pd.to_datetime(trips["pickup_datetime"])
    except Exception as exc:  # noqa: BLE001 - surface parse failures as DQ
        return CheckResult(
            "ingest_types",
            False,
            f"pickup_datetime is not parseable: {exc}",
        )
    return CheckResult(
        "ingest",
        True,
        f"Ingest ok ({len(trips)} rows)",
        {"rows": int(len(trips))},
    )


def check_feature_nulls(features: pd.DataFrame, max_null_rate: float) -> CheckResult:
    cols = [c for c in FEATURE_COLUMNS if c in features.columns]
    rates = {c: float(features[c].isna().mean()) for c in cols}
    bad = {c: r for c, r in rates.items() if r > max_null_rate}
    if bad:
        return CheckResult(
            "feature_null_rates",
            False,
            f"Null rate above {max_null_rate}: {bad}",
            {"null_rates": rates},
        )
    return CheckResult(
        "feature_null_rates",
        True,
        "Feature null rates within threshold",
        {"null_rates": rates},
    )


def check_feature_non_negative(features: pd.DataFrame) -> CheckResult:
    count_cols = ["trips_1h", "trips_24h", "trips_7d"]
    negatives = {
        c: int((features[c] < 0).sum())
        for c in count_cols
        if c in features.columns and (features[c] < 0).any()
    }
    if negatives:
        return CheckResult(
            "feature_non_negative",
            False,
            f"Negative trip counts found: {negatives}",
            {"negatives": negatives},
        )
    return CheckResult(
        "feature_non_negative",
        True,
        "Trip count features are non-negative",
    )


def check_volume_drop(
    zone_hours: pd.DataFrame,
    max_drop_pct: float,
    window_hours: int = 24,
) -> CheckResult:
    """
    Compare trip volume in the latest window vs the prior window (event-time).

    Uses the dataset timeline so historical sample data still gets a meaningful check.
    """
    df = zone_hours.copy()
    df["event_hour"] = pd.to_datetime(df["event_hour"])
    if df.empty:
        return CheckResult("volume_drop", False, "zone_hours is empty")

    max_hour = df["event_hour"].max()
    recent_start = max_hour - pd.Timedelta(hours=window_hours - 1)
    prior_start = recent_start - pd.Timedelta(hours=window_hours)
    prior_end = recent_start - pd.Timedelta(hours=1)

    recent = df.loc[df["event_hour"] >= recent_start, "trip_count"].sum()
    prior = df.loc[
        (df["event_hour"] >= prior_start) & (df["event_hour"] <= prior_end),
        "trip_count",
    ].sum()

    if prior <= 0:
        return CheckResult(
            "volume_drop",
            True,
            "Not enough prior volume for baseline; skipping fail",
            {"recent": int(recent), "prior": int(prior)},
        )

    drop_pct = float((prior - recent) / prior)
    details = {
        "recent": int(recent),
        "prior": int(prior),
        "drop_pct": drop_pct,
        "max_drop_pct": max_drop_pct,
    }
    if drop_pct > max_drop_pct:
        return CheckResult(
            "volume_drop",
            False,
            f"Volume drop {drop_pct:.1%} exceeds {max_drop_pct:.0%}",
            details,
        )
    return CheckResult(
        "volume_drop",
        True,
        f"Volume change ok (drop={drop_pct:.1%})",
        details,
    )


def check_materialization_freshness(
    features: pd.DataFrame,
    max_lag_hours: float,
    now: datetime | None = None,
) -> CheckResult:
    """Fail if feature materialization (computed_at) is stale vs wall clock."""
    if "computed_at" not in features.columns or features.empty:
        return CheckResult(
            "materialization_freshness",
            False,
            "computed_at missing or features empty",
        )
    now = now or datetime.now(timezone.utc).replace(tzinfo=None)
    computed = pd.to_datetime(features["computed_at"]).max()
    lag_hours = (now - computed.to_pydatetime()).total_seconds() / 3600.0
    details = {
        "max_computed_at": computed.isoformat(),
        "lag_hours": lag_hours,
        "max_lag_hours": max_lag_hours,
    }
    if lag_hours > max_lag_hours:
        return CheckResult(
            "materialization_freshness",
            False,
            f"Materialization lag {lag_hours:.2f}h exceeds {max_lag_hours}h",
            details,
        )
    return CheckResult(
        "materialization_freshness",
        True,
        f"Materialization fresh (lag={lag_hours:.2f}h)",
        details,
    )


def check_online_offline_parity(
    offline_latest: pd.DataFrame,
    online_store: LocalOnlineStore,
    sample_size: int,
    max_rel_error: float,
    feature_col: str = "trips_1h",
) -> CheckResult:
    if offline_latest.empty:
        return CheckResult("online_offline_parity", False, "No offline latest rows")

    sample = offline_latest.sample(
        n=min(sample_size, len(offline_latest)),
        random_state=42,
    )
    mismatches = 0
    compared = 0
    examples: list[dict[str, Any]] = []

    for row in sample.itertuples(index=False):
        zone_id = int(getattr(row, "zone_id"))
        offline_val = float(getattr(row, feature_col))
        online = online_store.get(zone_id)
        if online is None or feature_col not in online:
            mismatches += 1
            examples.append({"zone_id": zone_id, "reason": "missing_online"})
            continue
        online_val = float(online[feature_col])
        compared += 1
        denom = max(abs(offline_val), 1.0)
        rel_err = abs(online_val - offline_val) / denom
        if rel_err > max_rel_error:
            mismatches += 1
            examples.append(
                {
                    "zone_id": zone_id,
                    "offline": offline_val,
                    "online": online_val,
                    "rel_err": rel_err,
                }
            )

    details = {
        "sample_size": int(len(sample)),
        "compared": compared,
        "mismatches": mismatches,
        "feature_col": feature_col,
        "examples": examples[:5],
    }
    if mismatches:
        return CheckResult(
            "online_offline_parity",
            False,
            f"Parity failed for {mismatches}/{len(sample)} sampled zones",
            details,
        )
    return CheckResult(
        "online_offline_parity",
        True,
        f"Parity ok for {len(sample)} sampled zones",
        details,
    )


def run_dq_suite(settings: Settings | None = None) -> DQReport:
    settings = settings or get_settings()
    dq = settings.raw["dq"]
    freshness = settings.raw["freshness"]

    trips_path = settings.path("raw_dir") / "yellow_trips.parquet"
    zone_hours_path = settings.path("offline_dir") / "zone_hours.parquet"
    features_path = settings.path("offline_dir") / "zone_features.parquet"
    online_path = settings.path("online_dir") / "zone_features.json"

    checks: list[CheckResult] = []

    for label, path in [
        ("trips", trips_path),
        ("zone_hours", zone_hours_path),
        ("features", features_path),
        ("online", online_path),
    ]:
        if not path.exists():
            checks.append(
                CheckResult(
                    f"artifact_{label}",
                    False,
                    f"Missing artifact: {path}. Run meter offline && meter online first.",
                )
            )
    if any(not c.passed for c in checks):
        return DQReport(passed=False, checks=checks)

    trips = pd.read_parquet(trips_path)
    zone_hours = pd.read_parquet(zone_hours_path)
    features = pd.read_parquet(features_path)
    offline_latest = latest_features_by_zone(features)
    online_store = LocalOnlineStore(online_path)

    checks.extend(
        [
            check_ingest(trips),
            check_feature_nulls(features, float(dq["max_null_rate"])),
            check_feature_non_negative(features),
            check_volume_drop(zone_hours, float(dq["max_volume_drop_pct"])),
            check_materialization_freshness(
                features,
                float(freshness["alert_if_lag_hours"]),
            ),
            check_online_offline_parity(
                offline_latest,
                online_store,
                sample_size=int(dq["parity_sample_size"]),
                max_rel_error=float(dq["parity_max_rel_error"]),
            ),
        ]
    )
    passed = all(c.passed for c in checks)
    return DQReport(passed=passed, checks=checks)


def write_dq_report(report: DQReport, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(report.to_dict(), f, indent=2)
        f.write("\n")
    return path
