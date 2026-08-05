# Meter — NYC Taxi Zone Feature Store

## 1. Problem & scope

**Business question:** For each taxi zone, at hour `H`, will the next hour be high-demand?

**What this project is:** A mini feature store that turns raw taxi trips into trustworthy, time-correct zone features for:

1. **Training** — historical, point-in-time correct feature rows + labels
2. **Serving** — latest features per zone for low-latency lookup
3. **Operations** — freshness SLOs and data-quality gates so bad/stale/skewed data fails loudly

**What this project is not (v1):**

- A production ML model or hyperparameter tuning exercise
- A streaming feature platform (hourly batch is enough for next-hour demand)
- Full MLOps (MLflow / Kubeflow / model serving)

Working name for the library/package: **Meter**.

---

## 2. Design goals

| Goal | How we show it |
|---|---|
| Training–serving consistency | One feature definition feeds offline history and online latest |
| No future leakage | Point-in-time (PIT) join; demo leaky vs correct |
| Freshness | Hourly materialization + staleness metrics / alerts |
| Data quality for models | Gates on nulls, volume shocks, train↔online parity |
| Runnable local stack | Python, Airflow, AWS-shaped design (S3, Glue/PySpark, DynamoDB), local-first runtime |

---

## 3. Locked decisions

| Decision | Choice |
|---|---|
| Use case | Zone demand next hour (taxi/orders style) |
| Entity | `zone_id` (pickup location) |
| Cadence (v1) | Hourly batch |
| Runtime | Local-first; architecture documented as AWS prod |
| Feature tooling | Thin DIY store first (optional Feast/SageMaker later) |
| Data (v1) | Sample TLC-style trips first; real TLC Parquet after offline path works |
| Out of scope (v1) | Streaming, Kubeflow, full model serving |

---

## 4. System architecture

Two paths, **one feature definition** — that is how we avoid training–serving skew.

```text
                    ┌─────────────────────────────┐
                    │     Feature definitions     │
                    │  (same code / same version) │
                    └─────────────┬───────────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              ▼                                       ▼
   ┌─────────────────────┐               ┌─────────────────────┐
   │   Offline store     │               │   Online store      │
   │   (training)        │               │   (serving)         │
   │                     │               │                     │
   │  Historical features│               │  Latest by zone_id  │
   │  + PIT training set │               │  low-latency get()  │
   └──────────┬──────────┘               └──────────┬──────────┘
              │                                     │
              ▼                                     ▼
        Model training                        Model inference
```

| Store | Answers | Prod mental model | Local stand-in |
|---|---|---|---|
| **Offline** | What were features for zone Z at time T? | S3 Parquet / warehouse (Athena/Redshift) | `data/offline/*.parquet` |
| **Online** | What are features for zone Z *right now*? | DynamoDB | Local JSON / KV file under `data/online/` |

Orchestration (Airflow) runs materialization on a schedule, then DQ gates. If DQ fails, the run fails.

```text
Ingest trips
    → Aggregate zone × hour facts
    → Materialize features (as_of_ts)
    → Build PIT training set
    → Push latest features to online store
    → DQ / freshness / parity gates
```

---

## 5. Data model & contracts

### 5.1 Flow

```text
TripEvent
    → ZoneHourFact (zone_id + hour)
    → ZoneFeatures (as_of_ts + rolling windows)
    → Labels (trips in next hour)
    → TrainingRow (PIT join: features as_of ≤ prediction time + label)
```

### 5.2 Trip event (raw)

TLC-shaped fields (sample generator first; real TLC later):

| Column | Meaning |
|---|---|
| `pickup_datetime` | Event time |
| `pulocation_id` | Pickup zone → entity `zone_id` |
| `dolocation_id` | Dropoff zone |
| `trip_distance` | Miles |
| `fare_amount` / `total_amount` | Fare signals |
| `passenger_count` | Optional aggregate input |
| `payment_type` | Optional |

### 5.3 Zone-hour fact

Grain: one row per (`zone_id`, `event_hour`).

| Column | Meaning |
|---|---|
| `zone_id` | Entity key |
| `event_hour` | Hour bucket start |
| `trip_count` | Pickups in that hour |
| `total_fare` / `avg_fare` | Fare aggregates |
| `avg_distance` | Distance aggregate |
| `avg_passengers` | Passenger aggregate |

### 5.4 Features (v1)

Computed for each (`zone_id`, `as_of_ts = H`) using **only data with event time ≤ H**.

| Feature | Meaning |
|---|---|
| `trips_1h` | Pickups in the last completed hour ≤ H |
| `trips_24h` | Pickups in the last 24 hours ≤ H |
| `trips_7d` | Pickups in the last 7 days ≤ H |
| `avg_fare_24h` | Mean fare over last 24h ≤ H |
| `avg_distance_24h` | Mean distance over last 24h ≤ H |

Metadata on every feature row:

| Column | Meaning |
|---|---|
| `feature_version` | Definition version (e.g. `v1`) |
| `computed_at` | Processing time (when the job wrote the row) |

### 5.5 Labels

At prediction time `H`:

| Label | Definition |
|---|---|
| `label_trips_next_1h` | Trip count in hour `H+1` |
| `label_high_demand` | `label_trips_next_1h ≥ threshold` (config) |

Labels are for **training only**. Serving never sees them.

### 5.6 Training row

PIT join of features to labels:

- Entity match on `zone_id`
- Feature `as_of_ts ≤ prediction_ts` (backward as-of)
- Attach labels for horizon `H → H+1`

---

## 6. Point-in-time correctness & leakage

**Rule:** Features for prediction time `H` may only use information available at or before `H`.

| Allowed at `H=17:00` | Forbidden (leakage) |
|---|---|
| Trips with pickup ≤ 17:00 in rolling windows | Using hour 17:00–18:00 trips as a “feature” |
| Zone-hour facts with `event_hour ≤ 17:00` | Joining features where `as_of_ts > H` |

**Correctness demo:**

1. **Leaky join** — intentionally use next-hour trip count as a feature (correlates almost perfectly with the label)
2. **Correct PIT join** — `merge_asof`-style backward join
3. Document why leaky offline metrics lie and production fails

Unit tests must lock the correct behavior (e.g. at 10:00 you get the 10:00 feature row, never the 11:00 row).

---

## 7. Freshness

For next-hour demand, **hourly batch is the v1 design**.

| Concept | v1 choice |
|---|---|
| Refresh cadence | Hourly Airflow schedule |
| Feature TTL | ~1 hour expected freshness |
| Staleness | `now - last_successful_materialization` (and/or `now - max(as_of_ts)` online) |
| Alert | Fail / alert if lag exceeds SLO (target: ~2 hours for hourly features) |

Streaming (Kinesis / Kafka → near-real-time upserts) is a **phase-2** add-on for stricter freshness requirements.

---

## 8. Data quality gates

Treat DQ as product reliability for ML, not dashboard hygiene.

| Gate | Checks |
|---|---|
| **Ingest** | Required columns present; types parseable; empty extract fails |
| **Feature materialization** | Null rate caps; non-negative counts; volume drop vs recent baseline |
| **Pre-train / pre-serve** | Freshness/staleness SLO; sample parity: online latest ≈ offline latest per `zone_id` |

If any gate fails → pipeline fails (no “green DAG, silent bad features”).

Config-owned thresholds (examples): max null rate, max volume drop %, parity sample size, max relative error on parity.

---

## 9. Local vs AWS mapping

| Concern | Local (build/demo) | AWS (production design) |
|---|---|---|
| Raw / offline features | `data/raw`, `data/offline` Parquet | S3 + Glue/PySpark or Athena |
| Online store | Local KV / JSON | DynamoDB (`zone_id` PK) |
| Orchestration | Airflow DAG calling the same Python steps | MWAA / self-hosted Airflow |
| Metrics / alerts | Logs + DQ report | CloudWatch metrics/alarms |
| Config | `config/default.yaml` | Same shapes; env-specific overrides |

Implementation order: prove semantics locally, then optionally swap backends without changing feature definitions.

---

## 10. Repo shape (target)

```text
nyc-taxi-feature-store/
  docs/
    architecture.md          # this document
  src/meter/                 # library: ingest, features, PIT, online, dq, training
  dags/                      # Airflow DAG wiring the same steps
  config/                    # SLOs, thresholds, feature version
  tests/                     # PIT + leakage + happy-path tests
  data/                      # local artifacts (gitignored except tiny samples)
  README.md                  # diagram, how to run, project overview
```

---

## 11. Build phases

| Phase | Deliverable | Done when |
|---|---|---|
| **0 — Docs** | This architecture + contracts | Decisions locked; no ambiguity on entity/features/PIT |
| **1 — Offline path** | Sample ingest → zone-hour → features → PIT training set | CLI/tests produce training Parquet; PIT tests pass |
| **2 — Online path** | Push latest + `get(zone_id)` | Latest features readable from online store |
| **3 — DQ + leakage demo** | Gates + leaky-vs-correct CLI/docs | Failed DQ fails the run; leakage demo reproducible |
| **4 — Airflow** | Hourly DAG over the same steps | One DAG: materialize → train set → online → DQ |
| **5 — Docs polish** | README, diagrams, run instructions | Clone → run → understand in &lt;15 minutes |
| **Later** | Real TLC ingest | Same schemas; sample path remains for CI |
| **Optional** | Tiny baseline + MLflow logging | Thin consumer of the feature store |
| **Out of scope unless needed** | Kubeflow, full model serving cluster | Not part of v1 |

---

## 12. Non-goals

Do not expand v1 into:

- Every MLOps tool
- Multi-entity feature graphs
- Real-time streaming before batch correctness is solid
- Deep model research

Depth on **PIT correctness, skew prevention, freshness, and DQ** beats breadth.
