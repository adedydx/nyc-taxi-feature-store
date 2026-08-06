# Leaky vs correct join

This demo shows why point-in-time (PIT) joins matter when building training rows.

## Setup

```bash
meter offline --rows 20000
meter leakage
```

## What “leaky” means

For prediction time `H`, the label is trips in hour `H+1`.

A **leaky** feature uses that same next-hour count as an input. The model effectively sees the answer while training, so offline metrics look great and production fails.

A **correct PIT** feature (e.g. `trips_1h`) only uses data with event time ≤ `H`.

## What the CLI reports

| Metric | Meaning |
|---|---|
| `leaky_label_corr` | Correlation of leaked next-hour count with the label (~1.0) |
| `leaky_exact_match_rate` | Fraction of rows where leaked value equals the label |
| `correct_feature_label_corr` | Correlation of `trips_1h` with the label (weaker, realistic) |

## Rule

Features for time `H` may only use information available at or before `H`.
