#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_seeds.py
Quick, robust analysis of seed-cell outputs (e.g., seed_cells_H72.parquet).

See previous notes for details; this version fixes a KeyError on 'time_h'
by resetting index before previewing top hours and naming the index.
"""

from __future__ import annotations
import argparse, glob
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd

# -------------------- parsing helpers --------------------
CANDIDATE_TIME_COLS = ["time", "issue_time", "valid_time", "datetime"]
CANDIDATE_FLAG_COLS = ["seed", "alert", "alert_final", "alert_throttled"]
CANDIDATE_PROB_COLS = ["prob", "p", "pgeom", "p_seed"]

def read_any(path: str | Path, columns: Optional[List[str]] = None) -> pd.DataFrame:
    p = str(path).strip()
    low = p.lower()
    if low.endswith(".parquet") or low.endswith(".pq"):
        return pd.read_parquet(p, columns=columns)
    try:
        return pd.read_csv(p, compression="infer", low_memory=False, usecols=columns)
    except pd.errors.EmptyDataError:
        raise RuntimeError(f"{p} is empty (0 bytes). Was it created successfully?")
    except Exception:
        return pd.read_csv(p, compression="infer", engine="python",
                           on_bad_lines="warn", usecols=columns)

def _try_parse_time_raw(s: pd.Series, fmt: str | None) -> pd.Series:
    raw = s.astype(str).str.strip().str.replace("Z", "", regex=False)
    t1 = pd.to_datetime(raw, utc=True, errors="coerce")
    if t1.notna().mean() > 0.5: return t1.dt.tz_localize(None)
    if fmt:
        try:
            t2 = pd.to_datetime(raw, format=fmt, utc=True, errors="coerce")
            if t2.notna().mean() > 0.5: return t2.dt.tz_localize(None)
        except Exception: pass
    num = pd.to_numeric(raw, errors="coerce")
    if num.notna().any():
        mid = np.nanmedian(num)
        unit = "ms" if (pd.notna(mid) and mid > 1e11) else "s"
        t3 = pd.to_datetime(num, unit=unit, utc=True, errors="coerce")
        if t3.notna().mean() > 0.5: return t3.dt.tz_localize(None)
    t4 = pd.to_datetime(raw, utc=True, errors="coerce", infer_datetime_format=True)
    return t4.dt.tz_localize(None)

def _norm_lon(series: pd.Series, mode: str) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    if mode == "none": return x
    if mode == "0..360": return (x % 360 + 360) % 360
    return ((x + 180) % 360) - 180  # -180..180

def _parse_area(aoi: str | None):
    if not aoi: return None
    latN, lonW, latS, lonE = [float(x.strip()) for x in aoi.split(",")]
    return latN, lonW, latS, lonE

# -------------------- analysis helpers --------------------
def _pick_time_col(df: pd.DataFrame, prefer: str | None) -> str:
    if prefer and prefer in df.columns: return prefer
    for c in CANDIDATE_TIME_COLS:
        if c in df.columns: return c
    raise ValueError(f"Could not find a time column. Tried: { [prefer] + CANDIDATE_TIME_COLS }")

def _pick_flag_prob(df: pd.DataFrame, flag_col: Optional[str], prob_col: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    fcol = flag_col if (flag_col and flag_col in df.columns) else None
    if fcol is None:
        for c in CANDIDATE_FLAG_COLS:
            if c in df.columns: fcol = c; break
    pcol = prob_col if (prob_col and prob_col in df.columns) else None
    if pcol is None:
        for c in CANDIDATE_PROB_COLS:
            if c in df.columns: pcol = c; break
    return fcol, pcol

def _run_lengths_per_point(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    df = df.copy()
    df["time_h"] = pd.to_datetime(df[time_col], utc=True).dt.tz_localize(None).dt.floor("h")
    out = []
    for (lat, lon), g in df.groupby(["lat","lon"], sort=False):
        ts = g["time_h"].dropna().sort_values().unique()
        if len(ts) == 0: continue
        runs, run_len = [], 1
        for i in range(1, len(ts)):
            if (ts[i] - ts[i-1]).total_seconds() == 3600: run_len += 1
            else: runs.append(run_len); run_len = 1
        runs.append(run_len)
        u, c = np.unique(runs, return_counts=True)
        for L, cnt in zip(u, c):
            out.append(dict(lat=float(lat), lon=float(lon), run_len=int(L), runs=int(cnt)))
    if not out:
        return pd.DataFrame(columns=["lat","lon","run_len","runs"])
    return pd.DataFrame(out)

def _heatmap_bins(df: pd.DataFrame, nbins: int = 60) -> pd.DataFrame:
    lat = pd.to_numeric(df["lat"], errors="coerce").dropna().to_numpy()
    lon = pd.to_numeric(df["lon"], errors="coerce").dropna().to_numpy()
    if lat.size == 0:
        return pd.DataFrame(columns=["lat_c","lon_c","count"])
    H, xedges, yedges = np.histogram2d(lat, lon, bins=nbins)
    lat_c = 0.5 * (xedges[:-1] + xedges[1:])
    lon_c = 0.5 * (yedges[:-1] + yedges[1:])
    ii, jj = np.where(H > 0)
    rows = [dict(lat_c=float(lat_c[i]), lon_c=float(lon_c[j]), count=int(H[i,j])) for i,j in zip(ii,jj)]
    return pd.DataFrame(rows).sort_values("count", ascending=False)

# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser(description="Analyze seed cells and write a text report + small CSVs.")
    ap.add_argument("--seeds", required=True, help="Seed file (parquet/csv/csv.gz) or a glob.")
    ap.add_argument("--out-dir", default=None, help="Output folder (default: alongside the first input file).")
    ap.add_argument("--time-col", default=None)
    ap.add_argument("--prob-col", default=None)
    ap.add_argument("--flag-col", default=None)
    ap.add_argument("--time-format", default=None)
    ap.add_argument("--normalize-lon", choices=["none","-180..180","0..360"], default="none")
    ap.add_argument("--area", default=None, help='Optional crop "latN,lonW,latS,lonE" after lon normalization.')
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--bins", type=int, default=60)
    args = ap.parse_args()

    files = sorted(glob.glob(args.seeds))
    if not files:
        raise FileNotFoundError(f"No files matched: {args.seeds}")
    first = Path(files[0])
    out_dir = Path(args.out_dir) if args.out_dir else first.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    dfs = []
    for f in files:
        df = read_any(f)
        if df.empty:
            print(f"[warn] {f} empty; skipping."); continue
        df["__src__"] = Path(f).name
        dfs.append(df)
    if not dfs:
        raise RuntimeError("All inputs were empty.")
    df = pd.concat(dfs, ignore_index=True)

    time_col = _pick_time_col(df, args.time_col)
    flag_col, prob_col = _pick_flag_prob(df, args.flag_col, args.prob_col)

    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = _norm_lon(pd.to_numeric(df["lon"], errors="coerce"), args.normalize_lon)
    df = df.dropna(subset=["lat","lon"]).reset_index(drop=True)

    t = _try_parse_time_raw(df[time_col], args.time_format)
    bad = int(t.isna().sum())
    if bad: print(f"[info] dropping {bad:,} rows with invalid time.")
    df = df.loc[t.notna()].copy()
    df[time_col] = t[t.notna()]
    df["time_h"] = df[time_col].dt.floor("h")
    df["date"]   = df[time_col].dt.floor("D")

    aoi = _parse_area(args.area)
    if aoi:
        latN, lonW, latS, lonE = aoi
        before = len(df)
        df = df.loc[(df["lat"] <= latN) & (df["lat"] >= latS) &
                    (df["lon"] >= lonW) & (df["lon"] <= lonE)].reset_index(drop=True)
        print(f"[info] AOI crop: kept {len(df):,}/{before:,} rows.")

    if flag_col is None:
        if prob_col is not None:
            df["__flag__"] = (pd.to_numeric(df[prob_col], errors="coerce") > 0).astype(int)
            flag_col = "__flag__"
        else:
            df["__flag__"] = 1
            flag_col = "__flag__"
    else:
        df[flag_col] = pd.to_numeric(df[flag_col], errors="coerce").fillna(0).astype(int)

    rows, cols = len(df), len(df.columns)
    hours = df["time_h"].nunique()
    dates = df["date"].nunique()
    lat_min, lat_max = float(df["lat"].min()), float(df["lat"].max())
    lon_min, lon_max = float(df["lon"].min()), float(df["lon"].max())
    tmin, tmax = df[time_col].min(), df[time_col].max()
    uniq_pts = df[["lat","lon"]].drop_duplicates().shape[0]

    by_hour = df.groupby("time_h", sort=True)
    hourly_count = by_hour[flag_col].sum().rename("active_cells")
    hourly_count.index.name = "time_h"  # <-- ensure name for later
    if prob_col:
        hourly_prob_mean = by_hour[prob_col].mean().rename("prob_mean")
        hourly_prob_median = by_hour[prob_col].median().rename("prob_median")
        hourly = pd.concat([hourly_count, hourly_prob_mean, hourly_prob_median], axis=1)
    else:
        hourly = hourly_count.to_frame()

    per_day = df.groupby("date", sort=True)[flag_col].sum().rename("active_cells")

    north = df.loc[df["lat"] >= 0, flag_col].sum()
    south = df.loc[df["lat"]  < 0, flag_col].sum()

    hot = df.groupby(["lat","lon"], sort=False)[flag_col].sum().reset_index().rename(columns={flag_col:"hits"})
    hot = hot.sort_values("hits", ascending=False).reset_index(drop=True)
    topk = hot.head(int(args.topk))

    runs = _run_lengths_per_point(df[[time_col,"time_h","lat","lon",flag_col]].loc[df[flag_col] == 1], time_col=time_col)
    if not runs.empty:
        runs_summary = runs.groupby("run_len")["runs"].sum().sort_index().reset_index()
        longest = int(runs["run_len"].max())
        run50 = int(np.quantile(runs["run_len"], 0.50))
        run90 = int(np.quantile(runs["run_len"], 0.90))
    else:
        runs_summary = pd.DataFrame(columns=["run_len","runs"]); longest = run50 = run90 = 0

    bins_df = _heatmap_bins(df, nbins=int(args.bins))

    if len(files) == 1:
        stem = Path(files[0]).stem.replace(".csv","")
        report_path = out_dir / f"{stem}_analysis.txt"
    else:
        report_path = out_dir / "seeds_analysis.txt"

    hourly_path   = out_dir / "seeds_hourly.csv"
    day_path      = out_dir / "seeds_per_day.csv"
    hotspots_path = out_dir / "seeds_hotspots.csv"
    runs_path     = out_dir / "seeds_runs.csv"
    bins_path     = out_dir / "seeds_spatial_bins.csv"

    hourly.reset_index().to_csv(hourly_path, index=False)
    per_day.reset_index().to_csv(day_path, index=False)
    topk.to_csv(hotspots_path, index=False)
    runs_summary.to_csv(runs_path, index=False)
    bins_df.to_csv(bins_path, index=False)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Seed Analysis Report\n")
        f.write(f"{'-'*72}\n")
        f.write(f"Inputs matched : {len(files)} file(s)\n")
        for fpath in files[:5]: f.write(f"  - {Path(fpath).name}\n")
        if len(files) > 5: f.write(f"  … (+{len(files)-5} more)\n")
        f.write("\n")
        f.write(f"Rows          : {rows:,}\n")
        f.write(f"Columns       : {cols}\n")
        f.write(f"Unique points : {uniq_pts:,}\n")
        f.write(f"Hours         : {hours:,}   Days: {dates:,}\n")
        f.write(f"Time span     : {tmin} → {tmax}\n")
        f.write(f"Lat range     : {lat_min:.3f} .. {lat_max:.3f}\n")
        f.write(f"Lon range     : {lon_min:.3f} .. {lon_max:.3f}   (normalize_lon={args.normalize_lon})\n")
        f.write(f"AOI           : {args.area or '(none)'}\n")
        f.write(f"Time column   : {time_col}\n")
        f.write(f"Flag column   : {flag_col}\n")
        f.write(f"Prob column   : {prob_col or '(none)'}\n\n")

        if len(hourly) > 0:
            a = hourly["active_cells"].to_numpy()
            f.write("Hourly active cells:\n")
            f.write(f"  mean={float(np.mean(a)):.1f}  median={float(np.median(a)):.1f}  "
                    f"p5={float(np.quantile(a,0.05)):.1f}  p95={float(np.quantile(a,0.95)):.1f}  "
                    f"max={int(np.max(a))}\n")
            topH = hourly.reset_index().sort_values("active_cells", ascending=False).head(5)  # <-- fix
            f.write("  Top hours:\n")
            for _, r in topH.iterrows():
                line = f"    {r['time_h']} -> active={int(r['active_cells'])}"
                if 'prob_mean' in r and pd.notna(r['prob_mean']):   line += f", mean_p={r['prob_mean']:.3f}"
                if 'prob_median' in r and pd.notna(r['prob_median']): line += f", med_p={r['prob_median']:.3f}"
                f.write(line + "\n")
            f.write("\n")

        if len(per_day) > 0:
            dvals = per_day.values.astype(float)
            f.write("Per-day activity:\n")
            f.write(f"  mean={float(np.mean(dvals)):.1f}  median={float(np.median(dvals)):.1f}  "
                    f"min={int(np.min(dvals))}  max={int(np.max(dvals))}\n\n")

        f.write(f"Hemispheric split (sum of hits):  North={int(north):,}   South={int(south):,}\n\n")

        f.write("Persistence (consecutive hours at a grid point):\n")
        f.write(f"  longest_run={longest}h   median_run={run50}h   p90_run={run90}h\n")
        if not runs_summary.empty:
            preview = runs_summary.head(10)
            f.write("  run_len histogram (first 10 rows):\n")
            for _, r in preview.iterrows():
                f.write(f"    L={int(r['run_len'])}h : runs={int(r['runs'])}\n")
        f.write("\n")

        f.write(f"Top-{min(args.topk, len(topk))} hotspots (by hits):\n")
        for _, r in topk.iterrows():
            f.write(f"  lat={r['lat']:.3f}  lon={r['lon']:.3f}  hits={int(r['hits'])}\n")
        f.write("\n")

        f.write("Spatial density (binned) saved → seeds_spatial_bins.csv\n\n")
        f.write("Outputs written:\n")
        f.write(f"  • Report            : {report_path}\n")
        f.write(f"  • Hourly CSV        : {hourly_path}\n")
        f.write(f"  • Per-day CSV       : {day_path}\n")
        f.write(f"  • Hotspots CSV      : {hotspots_path}\n")
        f.write(f"  • Run-lengths CSV   : {runs_path}\n")
        f.write(f"  • Spatial bins CSV  : {bins_path}\n")

    print(f"[done] Wrote report → {report_path}")
    print(f"[done] Also wrote CSVs to {out_dir}")

if __name__ == "__main__":
    main()