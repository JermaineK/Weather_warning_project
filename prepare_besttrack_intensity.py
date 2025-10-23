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
"""

import argparse, sys, math
from typing import Optional, Tuple
import numpy as np
import pandas as pd

try:
    import xarray as xr
except Exception:
    xr = None

def _parse_area(s: Optional[str]) -> Optional[Tuple[float,float,float,float]]:
    if not s: return None
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 4:
        raise ValueError("--area must be 'latN,lonW,latS,lonE'")
    return tuple(map(float, parts))  # type: ignore

def _norm_lon(x: pd.Series, mode: str) -> pd.Series:
    v = pd.to_numeric(x, errors="coerce")
    if mode == "none": return v
    if mode == "0..360":
        v = (v % 360 + 360) % 360
    else:  # -180..180
        v = ((v + 180) % 360) - 180
    return v

def _to_naive_utc(s: pd.Series):
    return pd.to_datetime(s, utc=True, errors="coerce").dt.tz_localize(None)

def _read_ibtracs_csv(path: str) -> pd.DataFrame:
    # IBTrACS columns vary by release; we’ll probe common names.
    df = pd.read_csv(path)
    # canonicalize column names to lower for probing (keep original for values)
    lc = {c: c.lower() for c in df.columns}
    # time
    time_col = None
    for cand in ["iso_time","iso_time_str","time","datetime"]:
        if cand in lc.values():
            time_col = [k for k,v in lc.items() if v == cand][0]
            break
    if time_col is None:
        raise ValueError(f"{path}: could not find a time column like ISO_time/time")

    # lat/lon
    lat_col = next((k for k,v in lc.items() if v in ("lat","latitude")), None)
    lon_col = next((k for k,v in lc.items() if v in ("lon","longitude")), None)
    if lat_col is None or lon_col is None:
        raise ValueError(f"{path}: need lat/lon columns")

    # wind candidates (kt)
    wind_cols = {
        "USA":  ["usa_wind", "usa_wind_min", "usa_wind_max"],
        "WMO":  ["wmo_wind", "wmo_wind_min", "wmo_wind_max"],
        "auto": ["usa_wind", "wmo_wind", "usa_wind_min", "wmo_wind_min"],
    }
    # pressure candidates (hPa)
    pres_cols = ["usa_pres", "wmo_pres", "min_slp", "central_pressure", "pmin"]

    # name (optional)
    name_col = next((k for k,v in lc.items() if v in ("name","storm_name")), None)

    return df, time_col, lat_col, lon_col, wind_cols, pres_cols, name_col

def _read_ibtracs_netcdf(path: str) -> pd.DataFrame:
    if xr is None:
        raise RuntimeError("xarray not installed; cannot read NetCDF. Use IBTrACS CSV instead.")
    ds = xr.open_dataset(path)
    # IBTrACS NetCDF variables:
    # time (datetime64), lat, lon, wind, pres often have per-agency arrays.
    # We’ll try WMO then USA where available.
    # Convert to tidy long-form DataFrame.
    want = []
    for cand in ["time", "iso_time", "date_time"]:
        if cand in ds.variables: want.append(cand)
    for cand in ["lat","latitude"]:   want.append(cand) if cand in ds.variables else None
    for cand in ["lon","longitude"]:  want.append(cand) if cand in ds.variables else None
    # agency-specific:
    for cand in ["wmo_wind","usa_wind","wmo_pres","usa_pres","name"]:
        if cand in ds.variables: want.append(cand)
    df = ds[want].to_dataframe().reset_index()
    # Normalize columns similar to CSV reader
    return df, None, None, None, None, None, None  # We’ll probe generically below

def _pick_first(df: pd.DataFrame, cands):
    for c in cands:
        if c in df.columns: return c
        # try case-insensitive
        for cc in df.columns:
            if cc.lower() == c.lower(): return cc
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ibtracs", required=True, help="IBTrACS CSV (preferred) or NetCDF")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--start", default=None, help="YYYY-MM-DD (inclusive)")
    ap.add_argument("--end", default=None, help="YYYY-MM-DD (inclusive)")
    ap.add_argument("--area", default=None, help="latN,lonW,latS,lonE (after lon normalization)")
    ap.add_argument("--normalize-lon", default="-180..180", choices=["none","-180..180","0..360"])
    ap.add_argument("--wind-source", default="auto", choices=["auto","USA","WMO"])
    ap.add_argument("--time-offset-hours", type=float, default=0.0)
    ap.add_argument("--min-wind", type=float, default=0.0, help="Keep rows with vmax >= this (kt)")
    args = ap.parse_args()

    path = args.ibtracs
    if path.lower().endswith((".nc",".nc4",".netcdf")):
        df, time_col, lat_col, lon_col, wind_cols_map, pres_cols, name_col = _read_ibtracs_netcdf(path)
    else:
        df, time_col, lat_col, lon_col, wind_cols_map, pres_cols, name_col = _read_ibtracs_csv(path)

    # If NetCDF path, probe column names generically
    if time_col is None:
        time_col = _pick_first(df, ["iso_time","time","datetime","date_time"])
    if lat_col is None:
        lat_col  = _pick_first(df, ["lat","latitude"])
    if lon_col is None:
        lon_col  = _pick_first(df, ["lon","longitude"])
    if name_col is None:
        name_col = _pick_first(df, ["name","storm_name"])

    if time_col is None or lat_col is None or lon_col is None:
        raise ValueError("Could not detect time/lat/lon in IBTrACS file.")

    # Time → tz-naive UTC
    t = _to_naive_utc(df[time_col])
    if args.time_offset_hours:
        t = t + pd.to_timedelta(args.time_offset_hours, unit="h")

    lat = pd.to_numeric(df[lat_col], errors="coerce")
    lon = _norm_lon(df[lon_col], args.normalize_lon)

    # Wind/pressure selection
    wind_pref = wind_cols_map["auto"] if isinstance(wind_cols_map, dict) else ["usa_wind","wmo_wind"]
    if isinstance(wind_cols_map, dict) and args.wind_source in wind_cols_map:
        wind_pref = wind_cols_map[args.wind_source] + wind_pref  # ensure preferred first
    wind_col = _pick_first(df, wind_pref)
    pres_col = _pick_first(df, ["usa_pres","wmo_pres","min_slp","central_pressure","pmin"])

    vmax = pd.to_numeric(df[wind_col], errors="coerce") if wind_col else np.nan
    pmin = pd.to_numeric(df[pres_col], errors="coerce") if pres_col else np.nan
    name = df[name_col] if name_col in df.columns else ""

    out = pd.DataFrame({
        "obs_time": t,
        "lat": lat,
        "lon": lon,
        "vmax": vmax,
        "pmin": pmin,
        "name": name,
    }).dropna(subset=["obs_time","lat","lon"])

    # Optional filters
    if args.start:
        out = out.loc[out["obs_time"] >= pd.Timestamp(args.start)]
    if args.end:
        out = out.loc[out["obs_time"] <= pd.Timestamp(args.end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)]
    if args.min_wind > 0:
        out = out.loc[pd.to_numeric(out["vmax"], errors="coerce").fillna(-np.inf) >= float(args.min_wind)]

    aoi = _parse_area(args.area)
    if aoi:
        n, w, s, e = aoi
        out = out.loc[(out["lat"] <= n) & (out["lat"] >= s)]
        if w <= e:
            out = out.loc[(out["lon"] >= w) & (out["lon"] <= e)]
        else:
            out = out.loc[(out["lon"] >= w) | (out["lon"] <= e)]

    out = out.reset_index(drop=True)
    out.to_csv(args.out, index=False)
    print(f"[besttrack] wrote {args.out} | rows={len(out):,} "
          f"| time: {out['obs_time'].min()} → {out['obs_time'].max()}")

if __name__ == "__main__":
    main()