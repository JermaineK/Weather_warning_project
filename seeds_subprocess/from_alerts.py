#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
seeds_from_alerts_v2.py
Build hourly "proto-seeds" from alert files, robust to column name variations
and file formats, with optional AOI crop, longitude normalization, and Parquet output.

Outputs (same shapes/names as v1 for CSV; Parquet optional):
  {out_dir}/{run_name}_union_byhour.csv[.parquet]
  {out_dir}/{run_name}_starts_byhour.csv[.parquet]

Union-by-hour columns:
  time, lat, lon, prob_max, any_alert, n_hits

Starts-by-hour columns:
  lat, lon, time_start

Key upgrades:
  • Supports CSV(.gz) and Parquet inputs intermixed.
  • Robust time parsing with optional --time-format.
  • Longitude frame control and AOI crop.
  • Prob/flag autodetection with explicit overrides.
  • Dedup + type coercion safety.
  • Optional chunked CSV reads for very large files.
  • Optional Parquet outputs alongside CSV.
"""

from __future__ import annotations

import argparse, glob, os
from pathlib import Path
from typing import Optional, Iterable

import numpy as np
import pandas as pd

# -------------------- helpers --------------------

TIME_CANDIDATES = ["time", "valid_time", "t", "datetime"]
LAT_CANDIDATES  = ["lat", "latitude", "Lat", "Latitude"]
LON_CANDIDATES  = ["lon", "longitude", "Lon", "Longitude"]
PROB_CANDIDATES = ["prob", "prob_max", "p", "score", "max_prob_hour"]
FLAG_CANDIDATES = ["alert_final", "alert", "flag", "is_event", "label"]

def to_utc_naive(series: pd.Series, fmt: Optional[str]) -> pd.Series:
    raw = series.astype(str).str.strip().str.replace("Z", "", regex=False)
    if fmt:
        t = pd.to_datetime(raw, format=fmt, utc=True, errors="coerce")
    else:
        t = pd.to_datetime(raw, utc=True, errors="coerce")
    return t.dt.tz_convert(None)

def norm_lon(s: pd.Series, mode: str) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    if mode == "none":
        return x
    if mode == "0..360":
        return (x % 360 + 360) % 360
    return ((x + 180) % 360) - 180  # default -180..180

def parse_area(aoi: Optional[str]):
    if not aoi:
        return None
    latN, lonW, latS, lonE = [float(z.strip()) for z in aoi.split(",")]
    return latN, lonW, latS, lonE

def crop_area(df: pd.DataFrame, aoi) -> pd.DataFrame:
    latN, lonW, latS, lonE = aoi
    return df.loc[(df["lat"] <= latN) & (df["lat"] >= latS) &
                  (df["lon"] >= lonW) & (df["lon"] <= lonE)].copy()

def _first_present(df_cols: Iterable[str], cands: list[str]) -> Optional[str]:
    for c in cands:
        if c in df_cols:
            return c
    return None

def _read_csv_chunked(path: str, chunksize: int) -> Iterable[pd.DataFrame]:
    if chunksize <= 0:
        yield pd.read_csv(path, low_memory=False, compression="infer")
        return
    for chunk in pd.read_csv(path, low_memory=False, compression="infer", chunksize=int(chunksize)):
        yield chunk

def _read_any(path: str, chunksize: int) -> Iterable[pd.DataFrame]:
    low = path.lower()
    if low.endswith((".parquet", ".parq", ".pq", ".pqt")):
        yield pd.read_parquet(path)
    else:
        yield from _read_csv_chunked(path, chunksize)

def _normalize_and_project(df: pd.DataFrame,
                           prob_col_hint: Optional[str],
                           flag_col_hint: Optional[str],
                           time_fmt: Optional[str],
                           lon_frame: str,
                           aoi) -> pd.DataFrame:
    cols = df.columns

    # time
    tcol = _first_present(cols, TIME_CANDIDATES)
    if tcol is None:
        raise ValueError("Could not find a time column in: " + ", ".join(TIME_CANDIDATES))
    t = to_utc_naive(df[tcol], time_fmt)

    # lat/lon
    latc = _first_present(cols, LAT_CANDIDATES)
    lonc = _first_present(cols, LON_CANDIDATES)
    if not latc or not lonc:
        raise ValueError("Missing lat/lon columns; looked for " +
                         f"{LAT_CANDIDATES} / {LON_CANDIDATES}")
    lat = pd.to_numeric(df[latc], errors="coerce")
    lon = norm_lon(pd.to_numeric(df[lonc], errors="coerce"), lon_frame)

    # prob
    pcand = [prob_col_hint] if prob_col_hint else PROB_CANDIDATES
    pc = _first_present(cols, pcand)
    prob = pd.to_numeric(df[pc], errors="coerce") if pc else np.nan

    # flag
    fcand = [flag_col_hint] if flag_col_hint else FLAG_CANDIDATES
    fc = _first_present(cols, fcand)
    if fc:
        flag = pd.to_numeric(df[fc], errors="coerce").fillna(0).astype(int).clip(0, 1)
    else:
        flag = pd.Series(0, index=df.index, dtype=int)

    out = pd.DataFrame({"time": t, "lat": lat, "lon": lon, "prob": prob, "alert": flag})
    out = out.dropna(subset=["time", "lat", "lon"]).reset_index(drop=True)

    # AOI crop if requested
    if aoi is not None and not out.empty:
        out = crop_area(out, aoi)

    return out

# -------------------- main --------------------

def main():
    ap = argparse.ArgumentParser(description="Build hourly proto-seeds from alert files.")
    ap.add_argument("--alerts", required=True,
                    help="Glob of alert files, e.g. results/alerts/alerts_*_thr*.csv.gz (CSV/Parquet supported)")
    ap.add_argument("--prob-col", default=None, help="Explicit probability column override")
    ap.add_argument("--flag-col", default=None, help="Explicit alert flag column override")
    ap.add_argument("--time-floor", default="H", help="Time floor (pandas offset alias); default: H")
    ap.add_argument("--thr", type=float, default=None,
                    help="Optional prob threshold; keep rows with prob_max>=thr OR any_alert>0")
    ap.add_argument("--normalize-lon", choices=["none", "-180..180", "0..360"], default="-180..180",
                    help="Longitude normalization for inputs (default -180..180)")
    ap.add_argument("--area", default=None,
                    help='Optional AOI "latN,lonW,latS,lonE" applied after lon normalization')
    ap.add_argument("--chunk-rows", type=int, default=0,
                    help="Chunk size for large CSVs (0 disables chunking)")
    ap.add_argument("--out-dir", default="results/seedmaps")
    ap.add_argument("--run-name", default="proto")
    ap.add_argument("--time-format", default=None, help="Optional strptime format for non-standard time strings")
    ap.add_argument("--write-parquet", action="store_true", help="Also write Parquet copies next to CSVs")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.alerts))
    if not paths:
        raise FileNotFoundError(f"No files matched: {args.alerts}")

    aoi = parse_area(args.area)

    # Read + normalize all inputs (supports mixed CSV/Parquet)
    pieces: list[pd.DataFrame] = []
    for p in paths:
        try:
            for chunk in _read_any(p, args.chunk_rows):
                if chunk is None or len(chunk) == 0:
                    continue
                norm = _normalize_and_project(chunk,
                                              prob_col_hint=args.prob_col,
                                              flag_col_hint=args.flag_col,
                                              time_fmt=args.time_format,
                                              lon_frame=args.normalize_lon,
                                              aoi=aoi)
                if not norm.empty:
                    pieces.append(norm)
        except Exception as e:
            print(f"[warn] failed to read/normalize {p}: {e}")

    if not pieces:
        # write empty outputs to keep pipeline deterministic
        out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        union_path  = out_dir / f"{args.run_name}_union_byhour.csv"
        starts_path = out_dir / f"{args.run_name}_starts_byhour.csv"
        pd.DataFrame(columns=["time","lat","lon","prob_max","any_alert","n_hits"]).to_csv(union_path, index=False)
        pd.DataFrame(columns=["lat","lon","time_start"]).to_csv(starts_path, index=False)
        if args.write_parquet:
            pd.DataFrame(columns=["time","lat","lon","prob_max","any_alert","n_hits"]).to_parquet(str(union_path)+".parquet", index=False)
            pd.DataFrame(columns=["lat","lon","time_start"]).to_parquet(str(starts_path)+".parquet", index=False)
        print(f"[write] {union_path} rows=0")
        print(f"[write] {starts_path} rows=0")
        return

    df = pd.concat(pieces, ignore_index=True)

    # Floor to requested cadence and aggregate to union-by-hour cells
    df["time_h"] = df["time"].dt.floor(args.time_floor)

    agg = (df.groupby(["time_h", "lat", "lon"], as_index=False)
             .agg(prob_max=("prob", "max"),
                  any_alert=("alert", "max"),
                  n_hits=("alert", "size")))

    if args.thr is not None:
        agg = agg.loc[(agg["prob_max"] >= float(args.thr)) | (agg["any_alert"] > 0)].reset_index(drop=True)

    # starts_byhour: first hour a given (lat,lon) appears in union set
    starts = (agg.sort_values("time_h")
                .groupby(["lat", "lon"], as_index=False)
                .first()
                .rename(columns={"time_h": "time_start"}))

    # Write outputs
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_union  = out_dir / f"{args.run_name}_union_byhour.csv"
    out_starts = out_dir / f"{args.run_name}_starts_byhour.csv"

    agg = agg.rename(columns={"time_h": "time"})
    agg.to_csv(out_union, index=False, date_format="%Y-%m-%d %H:%M:%S")
    starts.to_csv(out_starts, index=False, date_format="%Y-%m-%d %H:%M:%S")

    if args.write_parquet:
        agg.to_parquet(str(out_union) + ".parquet", index=False)
        starts.to_parquet(str(out_starts) + ".parquet", index=False)

    print(f"[write] {out_union} rows={len(agg)}")
    print(f"[write] {out_starts} rows={len(starts)}")

if __name__ == "__main__":
    main()