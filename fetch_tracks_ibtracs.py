#!/usr/bin/env python3
"""
fetch_tracks_ibtracs.py
Download (or reuse cached) IBTrACS best-track CSV, filter to time window / names / basins,
and save a tidy tracks file for intensity correlation.

Output columns:
  time (UTC, tz-naive ISO), lat (deg), lon (deg),
  vmax (kt by default), pmin (hPa),
  name (upper), basin, storm_id, source

Examples (PowerShell):
  python .\fetch_tracks_ibtracs.py `
    --start 2025-02-01 --end 2025-04-30 `
    --names ZELIA,ERROL `
    --basins AU,SI,SP `
    --out data\tracks\besttrack_zelia_errol.csv

  # If you want m/s instead of kt:
  python .\fetch_tracks_ibtracs.py `
    --start 2025-02-01 --end 2025-04-30 `
    --names ZELIA,ERROL `
    --out data\tracks\besttrack_zelia_errol_mps.csv `
    --vmax-units mps
"""

import argparse, os, sys, io
import pandas as pd
import numpy as np
from pathlib import Path

IBTRACS_URL = (
  "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-"
  "stewardship-ibtracs/v04r01/access/csv/ibtracs.ALL.list.v04r01.csv"
)

def _download_or_cache(url: str, cache_path: Path) -> Path:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and cache_path.stat().st_size > 10_000:
        print(f"[cache] using cached IBTrACS file → {cache_path}")
        return cache_path
    try:
        import requests
        print(f"[download] {url}")
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        cache_path.write_bytes(r.content)
        print(f"[cache] saved → {cache_path}")
        return cache_path
    except Exception as e:
        print(f"[warn] download failed ({e}). If you have the file locally, place it at:\n  {cache_path}", file=sys.stderr)
        if cache_path.exists():
            print("[cache] falling back to existing (possibly stale) cache.", file=sys.stderr)
            return cache_path
        raise

def _to_utc_naive(s):
    t = pd.to_datetime(s, utc=True, errors="coerce")
    return t.dt.tz_convert(None)

def main():
    ap = argparse.ArgumentParser(description="Fetch & filter IBTrACS best-track data.")
    ap.add_argument("--out", required=True, help="Output CSV path for filtered tracks.")
    ap.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    ap.add_argument("--end", required=True, help="End date (YYYY-MM-DD, inclusive)")
    ap.add_argument("--names", default="", help="Comma-separated storm names to keep (e.g., ZELIA,ERROL). Empty = keep all.")
    ap.add_argument("--basins", default="AU,SI,SP", help="Comma list of basins to keep (IBTrACS BASIN codes). Default AU,SI,SP.")
    ap.add_argument("--cache-file", default="data/tracks/ibtracs.ALL.list.v04r01.csv", help="Local cache path for the big CSV.")
    ap.add_argument("--vmax-units", choices=["kt","mps"], default="kt", help="Output units for vmax (default kt).")
    args = ap.parse_args()

    cache_path = Path(args.cache_file)
    csv_path = _download_or_cache(IBTRACS_URL, cache_path)

    print(f"[read] {csv_path}")
    # IBTrACS has a few metadata header rows; pandas handles them fine.
    df = pd.read_csv(csv_path, low_memory=False)

    # Column availability varies; prefer USA_*; fall back to WMO
    # Common columns:
    #   SID, NAME, BASIN, ISO_TIME
    #   LAT, LON
    #   USA_WIND (kt), USA_PRES (hPa)
    #   WMO_WIND (kt), WMO_PRES (hPa)
    for col in ["SID","NAME","BASIN","ISO_TIME","LAT","LON"]:
        if col not in df.columns:
            raise ValueError(f"IBTrACS file missing expected column: {col}")

    # Normalize time
    df["_time"] = _to_utc_naive(df["ISO_TIME"])
    # Filter time window
    t0 = pd.to_datetime(args.start).tz_localize(None)
    t1 = pd.to_datetime(args.end).tz_localize(None)
    df = df.loc[(df["_time"] >= t0) & (df["_time"] <= (t1 + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)))].copy()

    # Filter basin(s)
    keep_basins = {b.strip().upper() for b in args.basins.split(",") if b.strip()}
    if keep_basins:
        df = df.loc[df["BASIN"].astype(str).str.upper().isin(keep_basins)].copy()

    # Filter names (if provided)
    keep_names = {n.strip().upper() for n in args.names.split(",") if n.strip()}
    df["NAME_UP"] = df["NAME"].astype(str).str.upper().str.strip()
    if keep_names:
        df = df.loc[df["NAME_UP"].isin(keep_names)].copy()

    # Coerce coords
    df["lat"] = pd.to_numeric(df["LAT"], errors="coerce")
    df["lon"] = pd.to_numeric(df["LON"], errors="coerce")
    # Normalize lon to [-180, 180]
    df["lon"] = ((df["lon"] + 180) % 360) - 180

    # Choose intensity fields
    wind_col = "USA_WIND" if "USA_WIND" in df.columns else ("WMO_WIND" if "WMO_WIND" in df.columns else None)
    pres_col = "USA_PRES" if "USA_PRES" in df.columns else ("WMO_PRES" if "WMO_PRES" in df.columns else None)
    if wind_col is None and pres_col is None:
        print("[warn] No wind or pressure columns found; will emit NaNs.")

    if wind_col:
        df["vmax_kt"] = pd.to_numeric(df[wind_col], errors="coerce")
    else:
        df["vmax_kt"] = np.nan

    if pres_col:
        df["pmin_hPa"] = pd.to_numeric(df[pres_col], errors="coerce")
    else:
        df["pmin_hPa"] = np.nan

    # Convert units if requested
    if args.vmax_units == "mps":
        # 1 kt = 0.514444 m/s
        df["vmax"] = df["vmax_kt"] * 0.514444
    else:
        df["vmax"] = df["vmax_kt"]
    df["pmin"] = df["pmin_hPa"]

    # Final tidy frame
    out = pd.DataFrame({
        "time": df["_time"],
        "lat": df["lat"],
        "lon": df["lon"],
        "vmax": df["vmax"],
        "pmin": df["pmin"],
        "name": df["NAME_UP"],
        "basin": df["BASIN"],
        "storm_id": df["SID"],
        "source": "IBTrACS_v04r01"
    })

    # Drop rows with no coordinates
    out = out.dropna(subset=["time","lat","lon"]).reset_index(drop=True)

    # Optional: for robustness, keep only rows that have at least one intensity measure
    # (but allow missing one of vmax/pmin)
    if out["vmax"].isna().all() and out["pmin"].isna().all():
        print("[warn] All intensity values are NaN; check the source columns or date filters.")

    # Sort and write
    out.sort_values(["name","time"], inplace=True, ignore_index=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, date_format="%Y-%m-%d %H:%M:%S")
    print(f"[write] {out_path}  (rows={len(out)})")

if __name__ == "__main__":
    main()