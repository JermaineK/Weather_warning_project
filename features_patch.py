import numpy as np
import pandas as pd

REQ_BASE = ["time", "lat", "lon", "zeta", "div", "S", "agree"]  # minimal set from rebuild

def _ensure_datetime(df):
    if not np.issubdtype(df["time"].dtype, np.datetime64):
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")

def _sorted(df):
    # Ensure deterministic order for groupby-diff/rolling
    return df.sort_values(["lat", "lon", "time"], kind="mergesort").reset_index(drop=True)

def _pressure_tendency(df):
    """
    Returns two 1-D Series aligned to df.index:
      - msl_d1h: 1-hour backward difference
      - msl_d3h: 3-hour backward difference
    Units follow df['msl'] (hPa if you wrote Pa/100 in rebuild).
    """
    if "msl" not in df.columns:
        print("[augment] skip: 'msl' not present; tendencies unavailable.")
        return pd.Series(index=df.index, dtype="float64"), pd.Series(index=df.index, dtype="float64")

    g = df.groupby(["lat", "lon"], sort=False, group_keys=False)
    d1h = g["msl"].diff(1)    # msl(t) - msl(t-1)
    d3h = g["msl"].diff(3)    # msl(t) - msl(t-3)

    # Fill initial NaNs with 0 (or keep NaN if you prefer)
    d1h = d1h.fillna(0.0)
    d3h = d3h.fillna(0.0)
    return d1h.astype("float64"), d3h.astype("float64")

def _rolling_stats(df, col, win=3):
    """
    3-hour rolling mean/std within each (lat,lon) column.
    Aligned 1-D Series, NaNs filled forward then zeros for first rows.
    """
    if col not in df.columns:
        return (pd.Series(index=df.index, dtype="float64"),
                pd.Series(index=df.index, dtype="float64"))

    g = df.groupby(["lat", "lon"], sort=False, group_keys=False)[col]
    mean = g.rolling(win, min_periods=1).mean().reset_index(level=[0,1], drop=True)
    std  = g.rolling(win, min_periods=1).std().reset_index(level=[0,1], drop=True)
    return mean.fillna(method="ffill").fillna(0.0).astype("float64"), \
           std.fillna(method="ffill").fillna(0.0).astype("float64")

def _shear(df):
    """
    Simple horizontal shear proxy from zeta/div. If u10/v10 exist, you can replace by true shear later.
    """
    if ("zeta" not in df.columns) or ("div" not in df.columns):
        return pd.Series(index=df.index, dtype="float64")
    # A cheap anisotropy proxy: magnitude difference
    return (np.abs(df["zeta"]) - np.abs(df["div"])).astype("float64")

def add_all_feature_enhancements(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a new DataFrame with extra columns:
      msl_d1h, msl_d3h,
      zeta_mean3h, zeta_std3h, div_mean3h, div_std3h, S_mean3h, S_std3h,
      shear_proxy
    Will gracefully skip those requiring missing inputs.
    """
    missing = [c for c in ["time","lat","lon"] if c not in df_in.columns]
    if missing:
        raise ValueError(f"augment: required columns missing: {missing}")

    df = df_in.copy()
    _ensure_datetime(df)
    df = _sorted(df)

    # Pressure tendencies (safe 1-D aligned series)
    d1h, d3h = _pressure_tendency(df)
    df["msl_d1h"] = d1h
    df["msl_d3h"] = d3h

    # 3-hour rolling stats for zeta/div/S if present
    for col, pref in [("zeta","zeta"), ("div","div"), ("S","S")]:
        mean3, std3 = _rolling_stats(df, col, win=3)
        df[f"{pref}_mean3h"] = mean3
        df[f"{pref}_std3h"]  = std3

    # Shear proxy (zeta vs div magnitude)
    df["shear_proxy"] = _shear(df)

    # Keep dtypes compact where possible
    for c in ["msl_d1h","msl_d3h","zeta_mean3h","zeta_std3h","div_mean3h","div_std3h",
              "S_mean3h","S_std3h","shear_proxy"]:
        if c in df.columns:
            df[c] = df[c].astype("float32")

    return df