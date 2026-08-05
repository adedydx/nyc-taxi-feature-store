# Meter — NYC Taxi Zone Feature Store

Offline-first feature store for next-hour taxi zone demand: point-in-time training sets, online lookup (next), freshness, and data-quality gates.

See [docs/architecture.md](docs/architecture.md) for the system design.

## Quick start (offline path)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
meter offline --rows 20000
pytest
```

Outputs (local):

- `data/raw/yellow_trips.parquet`
- `data/offline/zone_hours.parquet`
- `data/offline/zone_features.parquet`
- `data/training/training_set.parquet`
