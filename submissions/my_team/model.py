import numpy as np
import pandas as pd

WEATHER_COLS = [
    "temperature_2m", "relative_humidity_2m", "apparent_temperature",
    "precipitation", "rain", "snowfall", "cloud_cover", "wind_speed_10m",
]

STATION_META_COLS = [
    "start_lat", "start_lng",
    "bike_lane_length_500m", "park_area_500m",
    "university_count_1000m", "office_poi_count_1000m",
    "retail_poi_count_1000m", "restaurant_cafe_count_500m",
    "transit_stop_count_500m", "distance_to_nearest_rail_station",
    "distance_to_city_center",
]

FEATURE_COLS = [
    # raw time
    "f_hour", "f_weekday", "f_month",
    # cyclical encodings so the model sees periodicity
    "f_hour_sin", "f_hour_cos",
    "f_weekday_sin", "f_weekday_cos",
    "f_month_sin", "f_month_cos",
    # peak-hour flags (commuter patterns)
    "f_is_morning_peak", "f_is_evening_peak",
    # calendar
    "f_weekend", "f_holiday", "f_working_day",
    # city identity
    "f_city",
] + [f"f_{c}" for c in WEATHER_COLS] + [f"f_{c}" for c in STATION_META_COLS] + [
    # historical demand statistics (precomputed from training grid)
    "f_station_mean", "f_station_hour_mean", "f_station_weekday_hour_mean",
    "f_city_mean", "f_city_hour_mean", "f_city_weekday_hour_mean",
]


def normalize_sid(s: pd.Series) -> pd.Series:
    """Normalize station IDs so 3074, 3074.0, '3074.0' all become '3074'.
    Non-numeric IDs like 'station_abc' are kept as-is."""
    raw = s.astype("string").str.strip()
    numeric = pd.to_numeric(raw, errors="coerce")
    is_int = numeric.notna() & np.isfinite(numeric) & (numeric % 1 == 0)
    out = raw.copy()
    out.loc[is_int] = numeric.loc[is_int].astype("int64").astype("string")
    return out.fillna("__missing__")


def build_features(df: pd.DataFrame, artifacts: dict) -> np.ndarray:
    """Build the feature matrix from a station-hour DataFrame.

    Works at both train time (called by train.py) and predict time
    (called by BikeDemandModel.predict). LightGBM handles NaN natively,
    so missing weather/metadata values are left as NaN.
    """
    n = len(df)
    out = {}

    # --- timestamp → time features ---
    if "hour_ts" in df.columns:
        ts = pd.to_datetime(df["hour_ts"], errors="coerce")
    elif "target_hour_start" in df.columns:
        ts = pd.to_datetime(df["target_hour_start"], errors="coerce")
    else:
        ts = pd.Series([pd.NaT] * n, index=df.index)

    hour = ts.dt.hour.fillna(12).astype(int)
    weekday = ts.dt.weekday.fillna(0).astype(int)
    month = ts.dt.month.fillna(1).astype(int)

    out["f_hour"] = hour.values.astype(float)
    out["f_weekday"] = weekday.values.astype(float)
    out["f_month"] = month.values.astype(float)
    out["f_hour_sin"] = np.sin(2 * np.pi * hour.values / 24)
    out["f_hour_cos"] = np.cos(2 * np.pi * hour.values / 24)
    out["f_weekday_sin"] = np.sin(2 * np.pi * weekday.values / 7)
    out["f_weekday_cos"] = np.cos(2 * np.pi * weekday.values / 7)
    out["f_month_sin"] = np.sin(2 * np.pi * month.values / 12)
    out["f_month_cos"] = np.cos(2 * np.pi * month.values / 12)
    out["f_is_morning_peak"] = hour.isin([7, 8, 9]).astype(float).values
    out["f_is_evening_peak"] = hour.isin([17, 18, 19]).astype(float).values

    # --- calendar flags ---
    is_weekend = (weekday >= 5).values
    if "weekend" in df.columns:
        raw_wknd = df["weekend"].values.astype(float)
        wknd = np.where(np.isnan(raw_wknd), is_weekend.astype(float), raw_wknd)
    else:
        wknd = is_weekend.astype(float)
    out["f_weekend"] = wknd

    if "holiday" in df.columns:
        raw_hol = df["holiday"].values.astype(float)
        out["f_holiday"] = np.where(np.isnan(raw_hol), 0.0, raw_hol)
    else:
        out["f_holiday"] = np.zeros(n)

    if "working_day" in df.columns:
        raw_wd = df["working_day"].values.astype(float)
        out["f_working_day"] = np.where(np.isnan(raw_wd), 1.0 - wknd, raw_wd)
    else:
        out["f_working_day"] = 1.0 - wknd

    # --- city encoding ---
    city_col = df["city"].astype(str) if "city" in df.columns else pd.Series(["unknown"] * n, index=df.index)
    out["f_city"] = city_col.map(artifacts["city_encoder"]).fillna(-1).values.astype(float)

    # --- station key for lookups ---
    if "start_station_id" in df.columns:
        sid = normalize_sid(df["start_station_id"].astype(str))
    else:
        sid = pd.Series(["__missing__"] * n, index=df.index)
    station_key = city_col + "_" + sid

    hour_s = hour.astype(str)
    weekday_s = weekday.astype(str)

    # --- historical demand stats with fallback chain ---
    # unknown station  → city-level stat
    # unknown city     → global mean
    global_mean = float(artifacts["global_mean"])

    c_mean = city_col.map(artifacts["city_stats"]).fillna(global_mean).values.astype(float)
    c_h = (city_col + "_" + hour_s).map(artifacts["city_hour_stats"]).fillna(global_mean).values.astype(float)
    c_wh = (city_col + "_" + weekday_s + "_" + hour_s).map(artifacts["city_weekday_hour_stats"]).fillna(global_mean).values.astype(float)

    s_mean_raw = station_key.map(artifacts["station_stats"]).values.astype(float)
    s_h_raw = (station_key + "_" + hour_s).map(artifacts["station_hour_stats"]).values.astype(float)
    s_wh_raw = (station_key + "_" + weekday_s + "_" + hour_s).map(artifacts["station_weekday_hour_stats"]).values.astype(float)

    s_mean = np.where(np.isnan(s_mean_raw), c_mean, s_mean_raw)
    s_h = np.where(np.isnan(s_h_raw), s_mean, s_h_raw)
    s_wh = np.where(np.isnan(s_wh_raw), s_h, s_wh_raw)

    out["f_station_mean"] = s_mean
    out["f_station_hour_mean"] = s_h
    out["f_station_weekday_hour_mean"] = s_wh
    out["f_city_mean"] = c_mean
    out["f_city_hour_mean"] = c_h
    out["f_city_weekday_hour_mean"] = c_wh

    # --- weather (already in test_df, may be NaN) ---
    for col in WEATHER_COLS:
        out[f"f_{col}"] = df[col].values.astype(float) if col in df.columns else np.full(n, np.nan)

    # --- station metadata (already in test_df) ---
    for col in STATION_META_COLS:
        out[f"f_{col}"] = df[col].values.astype(float) if col in df.columns else np.full(n, np.nan)

    return np.column_stack([out[c] for c in FEATURE_COLS])


class BikeDemandModel:
    """LightGBM-based bike demand predictor.

    Predicts hourly ride counts per station for arbitrary cities.
    For unseen stations/cities, falls back to city-level or global demand stats.
    """

    def __init__(self):
        self.artifacts = None

    def load_artifacts(self, artifacts: dict) -> None:
        self.artifacts = artifacts

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        if self.artifacts is None:
            raise RuntimeError("Call load_artifacts() first.")
        X = build_features(test_df, self.artifacts)
        preds = self.artifacts["model"].predict(X)
        return np.maximum(0.0, preds)
