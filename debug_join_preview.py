#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import math
import numpy as np
import pandas as pd

TIME_CANDS = ["time","obs_time","ISO_TIME","datetime","date_time","valid_time"]
LAT_CANDS  = ["lat","LAT","latitude","y","lat_c"]
LON_CANDS  = ["lon","LON","longitude","x","lon_c"]

def read_any(path, usecols=None) -> pd.DataFrame:
    p = str(path).lower()
    if p.endswith((".parquet",".parq",".pq")):
        return pd.read_parquet(path, columns=usecols if usecols else None)
    return pd.read_csv(path, compression="infer", low_memory=False, usecols=usecols if usecols else None)

def pick_col(df: pd.DataFrame, explicit: str|None, cands: list[str], role: str) -> str:
    if explicit and explicit in df.columns: return explicit
    for c in cands:
        if c in df.columns: return c
    raise ValueError(f"{role}: none of {cands} found. Columns: {sorted(df.columns)[:20]} ...")

def to_utc_naive(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce").dt.tz_localize(None)

def norm_lon(vals, mode: str):
    x = pd.to_numeric(vals, errors="coerce")
    if mode == "none": return x
    if mode == "0..360": return (x % 360 + 360) % 360
    return ((x + 180) % 360) - 180

def parse_area(aoi: str|None):
    if not aoi: return None
    latN, lonW, latS, lonE = [float(x.strip()) for x in aoi.split(",")]
    return latN, lonW, latS, lonE

def crop_aoi(df, area):
    if not area: return df
    latN, lonW, latS, lonE = area
    return df.loc[(df["lat"]<=latN)&(df["lat"]>=latS)&(df["lon"]>=lonW)&(df["lon"]<=lonE)].copy()

def dup_keys(df: pd.DataFrame, tcol="time"):
    t = pd.to_datetime(df[tcol], utc=True, errors="coerce").dt.tz_localize(None)
    return int(df.assign(_t=t).duplicated(subset=["_t","lat","lon"]).sum())

def grid_footprint(df: pd.DataFrame, tcol="time"):
    t = pd.to_datetime(df[tcol], utc=True, errors="coerce").dt.tz_localize(None).dt.floor("h")
    H = df["lat"].nunique(); W = df["lon"].nunique(); T = t.nunique()
    return H,W,T

def top_hours(df: pd.DataFrame, flag_col=None, k=5):
    if "time" not in df.columns: return []
    t = pd.to_datetime(df["time"], utc=True, errors="coerce").dt.tz_localize(None).dt.floor("h")
    if flag_col and flag_col in df.columns:
        v = pd.to_numeric(df[flag_col], errors="coerce").fillna(0).astype(int)
    else:
        v = pd.Series(1, index=df.index)
    byh = pd.DataFrame({"t":t, "v":v}).groupby("t", sort=True)["v"].sum().sort_values(ascending=False)
    return [(str(i), int(v)) for i,v in byh.head(k).items()]

def main():
    ap = argparse.ArgumentParser(description="Preview features/labels alignment before join.")
    ap.add_argument("--features", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--normalize-lon", choices=["none","-180..180","0..360"], default="-180..180")
    ap.add_argument("--area", default=None, help='latN,lonW,latS,lonE')
    ap.add_argument("--labels-time-col", default=None)
    ap.add_argument("--labels-lat-col",  default=None)
    ap.add_argument("--labels-lon-col",  default=None)
    ap.add_argument("--flag-col", default=None, help="Optional binary flag in features (e.g., alert/label) for hour sums.")
    args = ap.parse_args()

    area = parse_area(args.area) if args.area else None

    # --- load
    F = read_any(args.features)
    L = read_any(args.labels)

    # --- pick columns
    ft = pick_col(F, None, TIME_CANDS, "features time")
    fla = pick_col(F, None, LAT_CANDS,  "features lat")
    flo = pick_col(F, None, LON_CANDS,  "features lon")

    lt = pick_col(L, args.labels_time_col, TIME_CANDS, "labels time")
    lla = pick_col(L, args.labels_lat_col,  LAT_CANDS,  "labels lat")
    llo = pick_col(L, args.labels_lon_col,  LON_CANDS,  "labels lon")

    # --- normalize + clean
    F = F.rename(columns={ft:"time",fla:"lat",flo:"lon"}).copy()
    L = L.rename(columns={lt:"time",lla:"lat",llo:"lon"}).copy()

    F["time"] = to_utc_naive(F["time"]); L["time"] = to_utc_naive(L["time"])
    F["lat"]  = pd.to_numeric(F["lat"], errors="coerce")
    F["lon"]  = norm_lon(F["lon"], args.normalize_lon)
    L["lat"]  = pd.to_numeric(L["lat"], errors="coerce")
    L["lon"]  = norm_lon(L["lon"], args.normalize_lon)

    F = F.dropna(subset=["time","lat","lon"]).reset_index(drop=True)
    L = L.dropna(subset=["time","lat","lon"]).reset_index(drop=True)

    if area:
        F = crop_aoi(F, area)
        L = crop_aoi(L, area)

    # --- basic ranges
    def rng(df, col): return (df[col].min(), df[col].max()) if not df.empty else (math.nan, math.nan)

    print("=== FEATURES ===")
    print("rows:", len(F))
    print("time:", *rng(F,"time"))
    print("lat :", *rng(F,"lat"))
    print("lon :", *rng(F,"lon"))
    try:
        H,W,T = grid_footprint(F)
        print(f"grid: H={H} × W={W} × T={T}")
    except Exception:
        pass
    dkF = dup_keys(F)
    if dkF: print("dup (time,lat,lon):", dkF)

    if args.flag_col and args.flag_col in F.columns:
        print("top hours (features flag sums):", top_hours(F, args.flag_col, k=5))

    print("\n=== LABELS ===")
    print("rows:", len(L))
    print("time:", *rng(L,"time"))
    print("lat :", *rng(L,"lat"))
    print("lon :", *rng(L,"lon"))

    # --- time overlap window
    tF = pd.to_datetime(F["time"], utc=True).dt.tz_localize(None)
    tL = pd.to_datetime(L["time"], utc=True).dt.tz_localize(None)
    if len(tF) and len(tL):
        tFmin,tFmax = tF.min(), tF.max()
        tLmin,tLmax = tL.min(), tL.max()
        lo = max(tFmin, tLmin)
        hi = min(tFmax, tLmax)
        has_overlap = lo <= hi
        print("\n=== TIME OVERLAP ===")
        print(f"features: {tFmin} → {tFmax}")
        print(f"labels  : {tLmin} → {tLmax}")
        print(f"overlap : {'YES' if has_overlap else 'NO'}"
              + (f"  ({lo} → {hi})" if has_overlap else ""))

    # --- heuristic warnings for join_labels_grid.py
    warn = []
    # old files with lat_c/lon_c
    if "lat_c" in F.columns or "lon_c" in F.columns:
        warn.append("features contain lat_c/lon_c; the join expects lat/lon (you’re fine after renaming above).")
    # sparse hours?
    perhF = pd.to_datetime(F["time"]).dt.floor("h").value_counts()
    if len(perhF) and perhF.min() == 0:
        warn.append("some feature hours empty after filters.")
    # label sparsity hint
    perhL = pd.to_datetime(L["time"]).dt.floor("h").value_counts()
    if len(perhL)==0:
        warn.append("no label hours parsed; check labels time column and format.")

    if warn:
        print("\n=== NOTES ===")
        for w in warn: print("-", w)

    # tiny samples
    print("\nSAMPLE rows (features):")
    print(F[["time","lat","lon"]].head(5).to_string(index=False))
    print("\nSAMPLE rows (labels):")
    print(L[["time","lat","lon"]].head(5).to_string(index=False))

if __name__ == "__main__":
    main()