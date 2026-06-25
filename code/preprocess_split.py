#!/usr/bin/env python3
"""Clean raw bike rides and create local development splits.

Default outputs follow the project workflow:

    dataset/split_train.csv       50%
    dataset/split_dev.csv         15%
    dataset/split_test.csv        15%
    dataset/split_holdout.csv     20%

For compatibility with the official local-evaluation instructions, the script
also writes:

    dataset/local_train_set.csv       same as split_train.csv
    dataset/local_validation_set.csv  same as split_dev.csv

The split is chronological within each city to better mimic forecasting.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path("dataset/train_set.csv")
DEFAULT_DATASET_DIR = Path("dataset")
REQUIRED_COLUMNS = ["city", "start_station_id"]
SPLIT_FRACTIONS = {
    "split_train.csv": 0.50,
    "split_dev.csv": 0.15,
    "split_test.csv": 0.15,
    "split_holdout.csv": 0.20,
}


def normalize_station_id(s: pd.Series) -> pd.Series:
    raw = s.astype("string").str.strip()
    numeric = pd.to_numeric(raw, errors="coerce")
    is_int_like = numeric.notna() & np.isfinite(numeric) & (numeric % 1 == 0)
    out = raw.copy()
    out.loc[is_int_like] = numeric.loc[is_int_like].astype("int64").astype("string")
    return out.fillna("__missing_station__")


def clean_raw_rides(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()

    if "hour_ts" in out.columns:
        out["hour_ts"] = pd.to_datetime(out["hour_ts"], errors="coerce").dt.floor("h")
    elif "started_at" in out.columns:
        out["hour_ts"] = pd.to_datetime(out["started_at"], errors="coerce").dt.floor("h")
    else:
        raise ValueError("Need hour_ts or started_at to define station-hour splits.")

    out["city"] = out["city"].astype("string").str.strip().fillna("__missing_city__")
    out["start_station_id"] = normalize_station_id(out["start_station_id"])

    before = len(out)
    out = out.dropna(subset=["hour_ts"])
    out = out.drop_duplicates().reset_index(drop=True)
    after = len(out)

    print(f"Cleaned rows: {before:,} -> {after:,}")
    return out


def chronological_city_split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    parts = {name: [] for name in SPLIT_FRACTIONS}

    for city, city_df in df.groupby("city", dropna=False):
        city_df = city_df.sort_values(["hour_ts", "start_station_id"]).reset_index(drop=True)
        n = len(city_df)

        n_train = int(round(n * SPLIT_FRACTIONS["split_train.csv"]))
        n_dev = int(round(n * SPLIT_FRACTIONS["split_dev.csv"]))
        n_test = int(round(n * SPLIT_FRACTIONS["split_test.csv"]))

        bounds = [
            n_train,
            n_train + n_dev,
            n_train + n_dev + n_test,
            n,
        ]

        city_parts = {
            "split_train.csv": city_df.iloc[:bounds[0]],
            "split_dev.csv": city_df.iloc[bounds[0]:bounds[1]],
            "split_test.csv": city_df.iloc[bounds[1]:bounds[2]],
            "split_holdout.csv": city_df.iloc[bounds[2]:bounds[3]],
        }

        print(
            f"{city}: "
            f"train={len(city_parts['split_train.csv']):,}, "
            f"dev={len(city_parts['split_dev.csv']):,}, "
            f"test={len(city_parts['split_test.csv']):,}, "
            f"holdout={len(city_parts['split_holdout.csv']):,}"
        )

        for name, part in city_parts.items():
            parts[name].append(part)

    return {
        name: pd.concat(frames, ignore_index=True).sort_values(["city", "hour_ts"]).reset_index(drop=True)
        for name, frames in parts.items()
    }


def write_splits(splits: dict[str, pd.DataFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename, split_df in splits.items():
        path = output_dir / filename
        split_df.to_csv(path, index=False)
        print(f"Wrote {path} ({len(split_df):,} rows)")

    splits["split_train.csv"].to_csv(output_dir / "local_train_set.csv", index=False)
    splits["split_dev.csv"].to_csv(output_dir / "local_validation_set.csv", index=False)
    print(f"Wrote {output_dir / 'local_train_set.csv'}")
    print(f"Wrote {output_dir / 'local_validation_set.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", default=DEFAULT_INPUT, type=Path)
    parser.add_argument("--output_dir", default=DEFAULT_DATASET_DIR, type=Path)
    args = parser.parse_args()

    raw = pd.read_csv(args.input_csv, low_memory=False)
    clean = clean_raw_rides(raw)
    splits = chronological_city_split(clean)
    write_splits(splits, args.output_dir)


if __name__ == "__main__":
    main()
