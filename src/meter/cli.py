"""CLI for the offline feature / training pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path


def _run_offline(sample_rows: int | None = None) -> dict[str, Path]:
    from meter.config import get_settings
    from meter.features.aggregate import materialize_zone_hours
    from meter.features.compute import materialize_features
    from meter.ingest import write_sample_trips
    from meter.training.dataset import build_training_set

    settings = get_settings()
    settings.ensure_dirs()

    trips = write_sample_trips(settings=settings, n_rows=sample_rows)
    zone_hours = materialize_zone_hours(trips_path=trips, settings=settings)
    features = materialize_features(zone_hours_path=zone_hours, settings=settings)
    training = build_training_set(settings=settings)

    return {
        "trips": trips,
        "zone_hours": zone_hours,
        "features": features,
        "training": training,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="meter",
        description="Meter — NYC taxi zone feature store",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    offline = sub.add_parser(
        "offline",
        help="Run sample ingest → zone-hour → features → PIT training set",
    )
    offline.add_argument(
        "--rows",
        type=int,
        default=None,
        help="Override sample trip row count (default from config)",
    )

    args = parser.parse_args(argv)

    if args.command == "offline":
        paths = _run_offline(sample_rows=args.rows)
        print("Offline path complete:")
        for name, path in paths.items():
            print(f"  {name}: {path}")
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
