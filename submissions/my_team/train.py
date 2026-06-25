#!/usr/bin/env python3
"""Train the bike-demand model and save artifacts to weights.joblib.

Run from inside the submission folder:

    cd submissions/my_team
    python train.py

The script reads ../../dataset/train_set.csv, builds a full station-hour
demand grid (including zero-demand hours), engineers features, trains a
LightGBM model with MAE objective, and saves everything to weights.joblib.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor

from model import normalize_sid, build_features, FEATURE_COLS, WEATHER_COLS, STATION_META_COLS

DATA_ROOT = Path("../../dataset")
TRAIN_CSV = DATA_ROOT / "train_set.csv"
OUTPUT_WEIGHTS = "weights.joblib"

# Hours kept for evaluation (matches build_station_hour_eval_data.py default)
KEEP_HOURS = set(range(6, 23))
# Minimum active window per station so zero-demand context is always present
MIN_ACTIVE_HOURS = 72
CAL_COLS = ["weekend", "holiday", "working_day"]


# ── Grid construction ──────────────────────────────────────────────────────────

def _build_training_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw ride data into a station-hour demand grid with zeros.

    For each station we find its active window (first/last ride), expand it
    to at least MIN_ACTIVE_HOURS, filter to daytime hours, then fill demand=0
    for hours with no rides. This mirrors what the official evaluator does.
    """
    df = df.copy()
    df["hour_ts"] = pd.to_datetime(df["hour_ts"], errors="coerce")
    df["city"] = df["city"].astype(str)
    df["sid"] = normalize_sid(df["start_station_id"].astype(str))

    # Ride count per city-station-hour
    demand = (
        df.groupby(["city", "sid", "hour_ts"])
        .size()
        .reset_index(name="demand")
    )

    # Station metadata: one representative value per station
    meta = [c for c in STATION_META_COLS if c in df.columns]
    station_meta = df.groupby(["city", "sid"])[meta].first().reset_index()

    # Weather: mean per city-hour (same for all stations in a city at a given hour)
    wcols = [c for c in WEATHER_COLS if c in df.columns]
    city_hour_weather = df.groupby(["city", "hour_ts"])[wcols].mean().reset_index()

    # Calendar flags: first observed value per city-hour
    calcols = [c for c in CAL_COLS if c in df.columns]
    city_hour_cal = df.groupby(["city", "hour_ts"])[calcols].first().reset_index()

    # City-level time span
    city_bounds = (
        demand.groupby("city")["hour_ts"]
        .agg(c_min="min", c_max="max")
        .reset_index()
    )

    # Station-level activity windows
    sw = (
        demand.groupby(["city", "sid"])["hour_ts"]
        .agg(first="min", last="max")
        .reset_index()
        .merge(city_bounds, on="city", how="left")
    )

    parts = []
    for row in sw.itertuples(index=False):
        s, e = row.first, row.last
        if pd.isna(s) or pd.isna(e):
            continue

        win = int((e - s).total_seconds() / 3600) + 1
        if win < MIN_ACTIVE_HOURS:
            deficit = MIN_ACTIVE_HOURS - win
            s = max(row.c_min, s - pd.Timedelta(hours=deficit // 2))
            e = min(row.c_max, e + pd.Timedelta(hours=(deficit + 1) // 2))

        hours = [h for h in pd.date_range(s, e, freq="h") if h.hour in KEEP_HOURS]
        if hours:
            parts.append(pd.DataFrame({
                "city": row.city,
                "sid": row.sid,
                "hour_ts": hours,
            }))

    if not parts:
        raise RuntimeError("Could not build any training rows from the data.")

    grid = pd.concat(parts, ignore_index=True)
    grid = grid.drop_duplicates(["city", "sid", "hour_ts"]).reset_index(drop=True)

    grid = grid.merge(demand, on=["city", "sid", "hour_ts"], how="left")
    grid["demand"] = grid["demand"].fillna(0).astype(int)
    grid = grid.merge(station_meta, on=["city", "sid"], how="left")
    grid = grid.merge(city_hour_weather, on=["city", "hour_ts"], how="left")
    grid = grid.merge(city_hour_cal, on=["city", "hour_ts"], how="left")

    # Rename so build_features recognises the column
    grid = grid.rename(columns={"sid": "start_station_id"})
    return grid


# ── Historical statistics ──────────────────────────────────────────────────────

def _compute_lookup_artifacts(grid: pd.DataFrame, global_mean: float) -> dict:
    """Precompute mean demand at several granularities for use as model features.

    These serve two purposes:
    1. Strong predictive features (station baseline, hourly pattern).
    2. Fallback for unseen stations/cities at prediction time.
    """
    sid = normalize_sid(grid["start_station_id"].astype(str))
    sk = grid["city"].astype(str) + "_" + sid
    ts = pd.to_datetime(grid["hour_ts"])

    tmp = pd.DataFrame({
        "city": grid["city"].astype(str),
        "sk": sk,
        "hour": ts.dt.hour,
        "weekday": ts.dt.weekday,
        "demand": grid["demand"].astype(float),
    })

    city_stats = tmp.groupby("city")["demand"].mean().to_dict()

    city_hour_stats = {
        f"{c}_{h}": v
        for (c, h), v in tmp.groupby(["city", "hour"])["demand"].mean().items()
    }

    city_weekday_hour_stats = {
        f"{c}_{wd}_{h}": v
        for (c, wd, h), v in tmp.groupby(["city", "weekday", "hour"])["demand"].mean().items()
    }

    station_stats = tmp.groupby("sk")["demand"].mean().to_dict()

    station_hour_stats = {
        f"{s}_{h}": v
        for (s, h), v in tmp.groupby(["sk", "hour"])["demand"].mean().items()
    }

    station_weekday_hour_stats = {
        f"{s}_{wd}_{h}": v
        for (s, wd, h), v in tmp.groupby(["sk", "weekday", "hour"])["demand"].mean().items()
    }

    return {
        "global_mean": global_mean,
        "city_stats": city_stats,
        "city_hour_stats": city_hour_stats,
        "city_weekday_hour_stats": city_weekday_hour_stats,
        "station_stats": station_stats,
        "station_hour_stats": station_hour_stats,
        "station_weekday_hour_stats": station_weekday_hour_stats,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Loading {TRAIN_CSV} ...")
    df = pd.read_csv(TRAIN_CSV, low_memory=False)
    print(f"  {len(df):,} rides | cities: {sorted(df['city'].unique())}")

    print("Building station-hour grid (includes zero-demand hours) ...")
    grid = _build_training_grid(df)
    print(f"  {len(grid):,} station-hour rows | demand mean={grid['demand'].mean():.3f}")

    global_mean = float(grid["demand"].mean())
    cities = sorted(grid["city"].unique())
    city_encoder = {c: i for i, c in enumerate(cities)}

    print("Computing historical demand statistics ...")
    lookup = _compute_lookup_artifacts(grid, global_mean)
    lookup["city_encoder"] = city_encoder

    print("Building feature matrix ...")
    X = build_features(grid, lookup)
    y = grid["demand"].values.astype(float)
    print(f"  X: {X.shape} | y mean={y.mean():.3f} std={y.std():.3f}")

    # Time-based split — last 15 % of chronological data used for early stopping
    order = pd.to_datetime(grid["hour_ts"]).argsort().values
    split = int(len(order) * 0.85)
    tr_idx, va_idx = order[:split], order[split:]
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_va, y_va = X[va_idx], y[va_idx]
    print(f"  train={len(X_tr):,}  val={len(X_va):,}")

    # Poisson loss: designed for count data, log-link ensures positive predictions,
    # handles zero-inflated demand better than absolute_error (which collapses to 0
    # when most labels are 0 because it targets the median).
    print("Training HistGradientBoostingRegressor (Poisson, early stopping on 15% val) ...")
    model = HistGradientBoostingRegressor(
        loss="poisson",
        max_iter=3000,
        learning_rate=0.05,
        max_leaf_nodes=127,
        min_samples_leaf=20,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=50,
        l2_regularization=0.1,
        random_state=42,
        verbose=1,
    )
    model.fit(X, y)
    best_iter = model.n_iter_

    val_preds = np.maximum(0.0, model.predict(X_va))
    val_mae = float(np.abs(y_va - val_preds).mean())
    print(f"  Val MAE (held-out): {val_mae:.4f}  |  trees used: {best_iter}")

    print(f"Retraining on full data with {best_iter} trees (no early stopping) ...")
    final_model = HistGradientBoostingRegressor(
        loss="poisson",
        max_iter=best_iter,
        learning_rate=0.05,
        max_leaf_nodes=127,
        min_samples_leaf=20,
        early_stopping=False,
        l2_regularization=0.1,
        random_state=42,
        verbose=0,
    )
    final_model.fit(X, y)

    artifacts = {
        "model": final_model,
        "feature_cols": FEATURE_COLS,
        **lookup,
    }

    joblib.dump(artifacts, OUTPUT_WEIGHTS, compress=3)
    size_mb = Path(OUTPUT_WEIGHTS).stat().st_size / 1e6
    print(f"Saved {OUTPUT_WEIGHTS}  ({size_mb:.1f} MB)")
    print(f"Features used ({len(FEATURE_COLS)}): {FEATURE_COLS}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", default=None, help="Override default train CSV path")
    ap.add_argument("--output", default=None, help="Override default output weights path")
    known, _ = ap.parse_known_args()
    if known.train_csv:
        TRAIN_CSV = Path(known.train_csv)
    if known.output:
        OUTPUT_WEIGHTS = known.output
    main()
