#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
features_join_features.py
Merge two feature tables on keys (default: "time, lat, lon").

Examples:
  python features_join_features.py \
    --left  data/features_eoi.parquet \
    --right data/features_bulk_shear.parquet \
    --on "time, lat, lon" \
    --out data/features_merged.parquet \
    --normalize-lon -180..180 --overwrite
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

def read_any(path: str, usecols=None):
    low = str(path).lower()
    if low.endswith((".parquet",".parq",".pq",".pqt")):
        return pd.read_parquet(path, columns=usecols if usecols else None)
    return pd.read_csv(path, compression="infer", low_memory=False, usecols=usecols if usecols else None)

def write_any(path: str, df: pd.DataFrame, overwrite=True):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists() and not overwrite:
        print(f"[skip] exists: {p}"); return
    low = p.name.lower()
    if low.endswith((".parquet",".parq",".pq",".pqt")):
        df.to_parquet(p, index=False)
    else:
        comp = "gzip" if (low.endswith(".gz") or p.suffix.lower()==".gz") else "infer"
        df.to_csv(p, index=False, compression=comp, date_format="%Y-%m-%d %H:%M:%S")

def to_naive_utc(s): 
    return pd.to_datetime(s, utc=True, errors="coerce").dt.tz_localize(None)

def wrap_lon(vals, mode):
    x = pd.to_numeric(vals, errors="coerce")
    m = str(mode or "none").replace(" ","")
    if m=="0..360":     return (x%360+360)%360
    if m=="-180..180":  return ((x+180)%360)-180
    return x

def parse_args():
    ap = argparse.ArgumentParser(description="Join two feature tables on keys")
    ap.add_argument("--left", required=True)
    ap.add_argument("--right", required=True)
    ap.add_argument("--on", default="time, lat, lon", help='comma-separated list of join keys')
    ap.add_argument("--out", required=True)
    ap.add_argument("--normalize-lon", default="-180..180")
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()

def main():
    args = parse_args()
    left  = read_any(args.left)
    right = read_any(args.right)

    if "time" in left:   left["time"]  = to_naive_utc(left["time"])
    if "time" in right:  right["time"] = to_naive_utc(right["time"])
    if "lon" in left:    left["lon"]   = wrap_lon(left["lon"], args.normalize_lon)
    if "lon" in right:   right["lon"]  = wrap_lon(right["lon"], args.normalize_lon)

    keys = [k.strip() for k in str(args.on).split(",") if k.strip()]
    out = left.merge(right, on=keys, how="left", suffixes=("","_r"))

    write_any(args.out, out, overwrite=args.overwrite)
    print(f"[join-features] wrote {args.out} rows={len(out):,}")

if __name__ == "__main__":
    main()