#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fetch_tracks_ibtracs.py — fast, robust subsetter for IBTrACS v04r01

Outputs tidy: time (UTC, tz-naive), lat, lon, vmax, pmin, name, basin, storm_id, source

Additions vs baseline:
  • usecols on read_csv for speed
  • multi-agency vmax/pmin fallback + (SID,time) dedup (max wind / min pressure)
  • dateline-aware AOI crop after lon normalization
  • --out supports .csv or .parquet
"""

import argparse, sys
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Set
import numpy as np
import pandas as pd

try:
    import yaml  # only needed if --yaml is used
except Exception:
    yaml = None

IBTRACS_URL_DEFAULT = (
    "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-"
    "stewardship-ibtracs/v04r01/access/csv/ibtracs.ALL.list.v04r01.csv"
)

# ---------------- utils ----------------

def _download_or_cache(url: str, cache_path: Path) -> Path:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and cache_path.stat().st_size > 10_000:
        print(f"[cache] {cache_path}")
        return cache_path
    try:
        import requests
        print(f"[download] {url}")
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        cache_path.write_bytes(r.content)
        print(f"[cache] saved → {cache_path}")
        return cache_path
    except Exception as e:
        print(f"[warn] download failed: {e}", file=sys.stderr)
        if cache_path.exists():
            print("[cache] falling back to existing cache.", file=sys.stderr)
            return cache_path
        raise

def _to_utc_naive(series) -> pd.Series:
    t = pd.to_datetime(series, utc=True, errors="coerce")
    # FIX: drop timezone to get naive UTC
    return t.dt.tz_localize(None)

def _norm_lon(x: pd.Series, frame: str) -> pd.Series:
    v = pd.to_numeric(x, errors="coerce")
    if frame == "0..360":
        return (v % 360 + 360) % 360
    return ((v + 180) % 360) - 180  # default −180..180

def _parse_area(aoi: str) -> Tuple[float, float, float, float]:
    latN, lonW, latS, lonE = [float(s.strip()) for s in aoi.split(",")]
    return latN, lonW, latS, lonE

def _crop_area_dateline(df: pd.DataFrame, area: Tuple[float,float,float,float]) -> pd.DataFrame:
    latN, lonW, latS, lonE = area
    lat_ok = (df["lat"] <= latN) & (df["lat"] >= latS)
    # if lon range doesn't cross dateline, simple box; else two boxes
    if lonW <= lonE:
        lon_ok = (df["lon"] >= lonW) & (df["lon"] <= lonE)
        return df.loc[lat_ok & lon_ok].copy()
    else:
        # crossing dateline (e.g., 170 .. -170 in −180..180)
        lon_ok = (df["lon"] >= lonW) | (df["lon"] <= lonE)
        return df.loc[lat_ok & lon_ok].copy()

def _read_yaml_defaults(yaml_path: str) -> Dict[str, str]:
    if yaml is None:
        raise RuntimeError("pyyaml not installed but --yaml was provided. pip install pyyaml")
    cfg = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8")) or {}
    d = cfg.get("defaults", {}) or {}
    out = {}
    for k in ("start","end","area","normalize_lon"):
        if d.get(k):
            out[k] = str(d[k])
    return out

def _preferred_wind(df: pd.DataFrame) -> pd.Series:
    # Prefer USA_WIND, WMO_WIND if present, else any *_WIND
    cand = [c for c in ["USA_WIND", "WMO_WIND"] if c in df.columns]
    if not cand:
        cand = [c for c in df.columns if c.endswith("_WIND")]
    if not cand:
        return pd.Series(np.nan, index=df.index)

    winds = {c: pd.to_numeric(df[c], errors="coerce") for c in cand}
    return pd.DataFrame(winds).max(axis=1)

def _preferred_pres(df: pd.DataFrame) -> pd.Series:
    cand = [c for c in ["USA_PRES", "WMO_PRES"] if c in df.columns]
    if not cand:
        cand = [c for c in df.columns if c.endswith("_PRES")]
    if not cand:
        return pd.Series(np.nan, index=df.index)

    pres = {c: pd.to_numeric(df[c], errors="coerce") for c in cand}
    return pd.DataFrame(pres).min(axis=1)

def _dedup_per_sid_time(df: pd.DataFrame) -> pd.DataFrame:
    # keep max wind, min pressure per (SID,time)
    key = ["SID","_time"]
    if not set(key).issubset(df.columns):
        return df
    df = df.sort_values(key, kind="mergesort")
    agg = {
        "lat":"first","lon":"first","NAME":"first","BASIN":"first",
        "vmax_kt":"max","pmin_hPa":"min"
    }
    return df.groupby(key, as_index=False).agg(agg)

def _write_any(path: str, df: pd.DataFrame):
    outp = Path(path); outp.parent.mkdir(parents=True, exist_ok=True)
    if outp.suffix.lower() in (".parquet",".parq",".pq"):
        df.to_parquet(outp, index=False)
    else:
        df.to_csv(outp, index=False, date_format="%Y-%m-%d %H:%M:%S")

# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser(description="Fetch & filter IBTrACS best tracks (config/time/area aware).")
    ap.add_argument("--out", required=True, help="Output path (.csv or .parquet).")
    ap.add_argument("--yaml", default=None, help="Pipeline YAML with defaults.start/end/area/normalize_lon.")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--area", default=None, help='latN,lonW,latS,lonE (dateline-aware)')
    ap.add_argument("--normalize-lon", choices=["-180..180","0..360"], default=None)
    ap.add_argument("--names", default="")
    ap.add_argument("--basins", default="")
    ap.add_argument("--min-wind", type=float, default=None, help="Minimum vmax (in chosen units).")
    ap.add_argument("--url", default=IBTRACS_URL_DEFAULT)
    ap.add_argument("--cache-file", default="data/tracks/ibtracs.ALL.list.v04r01.csv")
    ap.add_argument("--vmax-units", choices=["kt","mps"], default="kt")
    args = ap.parse_args()

    # YAML defaults
    y = _read_yaml_defaults(args.yaml) if args.yaml else {}
    start = args.start or y.get("start")
    end   = args.end   or y.get("end")
    if not start or not end:
        raise ValueError("Provide --start/--end or --yaml defaults.start/end.")
    area_str = args.area or y.get("area")
    lon_frame = (args.normalize_lon or y.get("normalize_lon") or "-180..180")

    csv_path = _download_or_cache(args.url, Path(args.cache_file))

    # Read only needed columns (case tolerant where possible)
    usecols = [
        "SID","NAME","BASIN","ISO_TIME",
        "LAT","LON",
        "USA_WIND","WMO_WIND","USA_PRES","WMO_PRES"
    ]
    try:
        df = pd.read_csv(csv_path, low_memory=False, usecols=usecols)
    except Exception:
        df = pd.read_csv(csv_path, low_memory=False)

    # Normalize essential cols (case variants)
    for req in ["SID","NAME","BASIN","ISO_TIME","LAT","LON"]:
        if req not in df.columns:
            alt = req.lower()
            if alt in df.columns:
                df[req] = df[alt]
    for req in ["SID","NAME","BASIN","ISO_TIME","LAT","LON"]:
        if req not in df.columns:
            raise ValueError(f"IBTrACS file missing expected column: {req}")

    # Time filter (inclusive end-day)
    df["_time"] = _to_utc_naive(df["ISO_TIME"])
    t0 = pd.to_datetime(start).tz_localize(None)
    t1 = pd.to_datetime(end).tz_localize(None) + pd.Timedelta(days=1)
    df = df.loc[(df["_time"] >= t0) & (df["_time"] < t1)].copy()

    # Filters
    if args.basins.strip():
        keep_basins: Set[str] = {b.strip().upper() for b in args.basins.split(",") if b.strip()}
        df = df.loc[df["BASIN"].astype(str).str.upper().isin(keep_basins)].copy()
    df["NAME_UP"] = df["NAME"].astype(str).str.upper().str.strip()
    if args.names.strip():
        keep_names: Set[str] = {n.strip().upper() for n in args.names.split(",") if n.strip()}
        df = df.loc[df["NAME_UP"].isin(keep_names)].copy()

    # Coords + lon normalization
    df["lat"] = pd.to_numeric(df["LAT"], errors="coerce")
    df["lon"] = _norm_lon(pd.to_numeric(df["LON"], errors="coerce"), lon_frame)

    # AOI crop (dateline-aware)
    if area_str:
        df = _crop_area_dateline(df, _parse_area(area_str))

    # Intensities with fallbacks
    df["vmax_kt"]  = _preferred_wind(df)
    df["pmin_hPa"] = _preferred_pres(df)

    # Deduplicate per (SID, time) taking strongest wind / lowest pressure
    df = _dedup_per_sid_time(df)

    # Units
    if args.vmax_units == "mps":
        df["vmax"] = pd.to_numeric(df["vmax_kt"], errors="coerce") * 0.514444
    else:
        df["vmax"] = pd.to_numeric(df["vmax_kt"], errors="coerce")
    df["pmin"] = pd.to_numeric(df["pmin_hPa"], errors="coerce")

    # Min wind filter (in output units)
    if args.min_wind is not None:
        df = df.loc[df["vmax"] >= float(args.min_wind)].copy()

    out = (pd.DataFrame({
        "time": df["_time"],
        "lat": df["lat"],
        "lon": df["lon"],
        "vmax": df["vmax"],
        "pmin": df["pmin"],
        "name": df["NAME_UP"],
        "basin": df["BASIN"],
        "storm_id": df["SID"],
        "source": "IBTrACS_v04r01",
    })
    .dropna(subset=["time","lat","lon"])
    .sort_values(["name","time"], kind="mergesort")
    .reset_index(drop=True))

    if out["vmax"].isna().all() and out["pmin"].isna().all():
        print("[warn] all intensity values are NaN in the chosen window/area.", file=sys.stderr)

    _write_any(args.out, out)
    print(f"[write] {args.out} (rows={len(out)})")

if __name__ == "__main__":
    main()