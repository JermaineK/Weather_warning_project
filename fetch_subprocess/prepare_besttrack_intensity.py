#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prepare best-track intensity CSV for intensity_analysis.py.

Inputs:
  --ibtracs     Path to IBTrACS file (CSV preferred; NetCDF supported if xarray is installed)

Output:
  --out         CSV with columns: obs_time, lat, lon, vmax, pmin, name

Options:
  --start, --end        Date filters (YYYY-MM-DD)
  --area                'latN,lonW,latS,lonE' crop (after lon normalization)
  --normalize-lon       none | -180..180 | 0..360  (default: -180..180)
  --wind-source         auto | USA | WMO           (choose which wind to prefer)
  --time-offset-hours   Shift all times (e.g., -10 for local→UTC corrections)
  --min-wind            Keep rows with vmax >= this (kt)
  --quiet               Reduce console output
"""

import argparse, sys
from typing import Optional, Tuple
import numpy as np
import pandas as pd

try:
    import xarray as xr
except Exception:
    xr = None


# ----------------------------- helpers -----------------------------

def _parse_area(s: Optional[str]) -> Optional[Tuple[float,float,float,float]]:
    if not s:
        return None
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 4:
        raise ValueError("--area must be 'latN,lonW,latS,lonE'")
    return tuple(map(float, parts))  # type: ignore


def _norm_lon(x: pd.Series, mode: str) -> pd.Series:
    v = pd.to_numeric(x, errors="coerce")
    # Clean common fill values that sometimes sneak into IBTrACS
    v = v.replace({999.0: np.nan, 999.9: np.nan, 9999.0: np.nan})
    if mode == "none":
        return v
    if mode == "0..360":
        v = (v % 360 + 360) % 360
    else:  # "-180..180"
        v = ((v + 180) % 360) - 180
    return v


def _to_naive_utc(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce").dt.tz_localize(None)


def _pick_first(df: pd.DataFrame, cands) -> Optional[str]:
    for c in cands:
        if c in df.columns:
            return c
        for cc in df.columns:
            if cc.lower() == str(c).lower():
                return cc
    return None


def _read_ibtracs_csv(path: str):
    df = pd.read_csv(path)
    # Probe common names (case-insensitive)
    time_col = _pick_first(df, ["iso_time", "iso_time_str", "time", "datetime"])
    lat_col  = _pick_first(df, ["lat", "latitude"])
    lon_col  = _pick_first(df, ["lon", "longitude"])
    name_col = _pick_first(df, ["name", "storm_name"])

    # wind/pressure candidate maps (strings, unit handling later)
    wind_cols_map = {
        "USA":  ["usa_wind", "usa_wind_min", "usa_wind_max"],
        "WMO":  ["wmo_wind", "wmo_wind_min", "wmo_wind_max"],
        "auto": ["usa_wind", "wmo_wind", "usa_wind_min", "wmo_wind_min", "wind", "max_wind"]
    }
    pres_cols = ["usa_pres", "wmo_pres", "min_slp", "central_pressure", "pmin", "pres"]

    return df, time_col, lat_col, lon_col, wind_cols_map, pres_cols, name_col


def _read_ibtracs_netcdf(path: str):
    if xr is None:
        raise RuntimeError("xarray not installed; cannot read NetCDF. Use an IBTrACS CSV instead.")
    ds = xr.open_dataset(path)
    # Pull a broad set and tidy to DataFrame; we'll probe column names later
    want = []
    for cand in ["time", "iso_time", "date_time"]:
        if cand in ds.variables:
            want.append(cand)
    for cand in ["lat", "latitude"]:
        if cand in ds.variables:
            want.append(cand)
    for cand in ["lon", "longitude"]:
        if cand in ds.variables:
            want.append(cand)
    for cand in ["wmo_wind", "usa_wind", "wind",
                 "wmo_pres", "usa_pres", "pres",
                 "name", "storm_name"]:
        if cand in ds.variables:
            want.append(cand)
    df = ds[want].to_dataframe().reset_index()
    # We’ll probe generically below
    return df, None, None, None, None, None, None


# ----------------------------- main -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ibtracs", required=True, help="IBTrACS CSV (preferred) or NetCDF")
    ap.add_argument("--out", required=True, help="Output CSV path (.gz will be compressed)")

    ap.add_argument("--start", default=None, help="YYYY-MM-DD (inclusive)")
    ap.add_argument("--end",   default=None, help="YYYY-MM-DD (inclusive)")
    ap.add_argument("--area",  default=None, help="latN,lonW,latS,lonE (after lon normalization)")

    ap.add_argument("--normalize-lon", default="-180..180", choices=["none", "-180..180", "0..360"])
    ap.add_argument("--wind-source",   default="auto", choices=["auto", "USA", "WMO"])
    ap.add_argument("--time-offset-hours", type=float, default=0.0)
    ap.add_argument("--min-wind", type=float, default=0.0, help="Keep rows with vmax >= this (kt)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    path = args.ibtracs
    if path.lower().endswith((".nc", ".nc4", ".netcdf")):
        df, time_col, lat_col, lon_col, wind_cols_map, pres_cols, name_col = _read_ibtracs_netcdf(path)
    else:
        df, time_col, lat_col, lon_col, wind_cols_map, pres_cols, name_col = _read_ibtracs_csv(path)

    # Probe missing key columns generically if needed (NetCDF path, quirky CSVs)
    if time_col is None:
        time_col = _pick_first(df, ["iso_time", "time", "datetime", "date_time"])
    if lat_col is None:
        lat_col  = _pick_first(df, ["lat", "latitude"])
    if lon_col is None:
        lon_col  = _pick_first(df, ["lon", "longitude"])
    if name_col is None:
        name_col = _pick_first(df, ["name", "storm_name"])

    if time_col is None or lat_col is None or lon_col is None:
        raise ValueError("Could not detect time/lat/lon in IBTrACS file.")

    # Time → tz-naive UTC (+ optional offset)
    t = _to_naive_utc(df[time_col])
    if args.time_offset_hours:
        t = t + pd.to_timedelta(args.time_offset_hours, unit="h")

    # Numeric coercion early (robust to 'NA', blanks)
    lat = pd.to_numeric(df[lat_col], errors="coerce")
    lon = _norm_lon(df[lon_col], args.normalize_lon)

    # Choose wind/pressure candidates
    # wind_cols_map may be None in some NetCDF variants; fall back to generic list
    if isinstance(wind_cols_map, dict):
        src = args.wind_source.upper()
        pref = wind_cols_map.get(src, wind_cols_map.get("auto", []))
    else:
        pref = ["usa_wind", "wmo_wind", "wind", "max_wind"]
    wind_col = _pick_first(df, pref)
    pres_col = _pick_first(df, pres_cols or ["usa_pres", "wmo_pres", "min_slp", "central_pressure", "pmin", "pres"])

    vmax = pd.to_numeric(df[wind_col], errors="coerce") if wind_col else pd.Series([np.nan] * len(df))
    pmin = pd.to_numeric(df[pres_col], errors="coerce") if pres_col else pd.Series([np.nan] * len(df))

    # Handle units (rough heuristic): if values look small, assume m/s → convert to kt
    # Typical kt winds for TCs are > 34 kt. If 95th pct < 60, it's probably m/s.
    if vmax.notna().sum() > 10:
        p95 = np.nanpercentile(vmax.to_numpy(dtype=float), 95)
        if p95 < 60:  # likely m/s
            vmax = vmax * 1.94384  # m/s → kt

    # Name as a vector (not a scalar) if missing
    if name_col and name_col in df.columns:
        name = df[name_col].astype(str)
    else:
        name = pd.Series([""] * len(df))

    # Build output and drop obviously broken locations/times
    out = pd.DataFrame({
        "obs_time": t,
        "lat": lat,
        "lon": lon,
        "vmax": vmax,
        "pmin": pmin,
        "name": name,
    }).dropna(subset=["obs_time", "lat", "lon"]).reset_index(drop=True)

    n0 = len(out)

    # Optional filters
    if args.start:
        out = out.loc[out["obs_time"] >= pd.Timestamp(args.start)]
    if args.end:
        # inclusive day
        out = out.loc[out["obs_time"] <= (pd.Timestamp(args.end) + pd.Timedelta(days=1))]
    if args.min_wind > 0:
        out = out.loc[pd.to_numeric(out["vmax"], errors="coerce").fillna(-np.inf) >= float(args.min_wind)]

    aoi = _parse_area(args.area)
    if aoi:
        n, w, s, e = aoi
        out = out.loc[(out["lat"] <= n) & (out["lat"] >= s)]
        if w <= e:
            out = out.loc[(out["lon"] >= w) & (out["lon"] <= e)]
        else:
            # crosses antimeridian
            out = out.loc[(out["lon"] >= w) | (out["lon"] <= e)]

    if not args.quiet:
        print(f"[besttrack] pre-drop rows: {n0:,}  → kept: {len(out):,}")

    # Write (auto-compress if .gz)
    compression = "gzip" if str(args.out).lower().endswith(".gz") else "infer"
    out.to_csv(args.out, index=False, compression=compression, date_format="%Y-%m-%d %H:%M:%S")

    if not args.quiet:
        tmin = out["obs_time"].min()
        tmax = out["obs_time"].max()
        print(f"[besttrack] wrote {args.out} | rows={len(out):,} | time: {tmin} → {tmax}")


if __name__ == "__main__":
    main()